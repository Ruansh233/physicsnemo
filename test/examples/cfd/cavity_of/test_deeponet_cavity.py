# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples.cfd.cavity_of.deeponet_cavity import (
    CavityCaseMetaData,
    CavityDeepONetConfig,
    CavityPhysicalLimits,
    build_deeponet,
    build_sample_tensors,
    compute_reynolds_number,
    ensure_openfoam_environment,
    run_case_for_viscosity,
    validate_viscosity_and_reynolds,
)


class _FakeTransportProperties(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeBoundaryValue:
    def __init__(self, value):
        self.value = value


class _FakeField:
    def __init__(self, internal_field, boundary_field=None):
        self.internal_field = internal_field
        self.boundary_field = boundary_field or {}


class _FakeTimeDirectory:
    def __init__(self, time, fields, centers):
        self.time = time
        self._fields = fields
        self._centers = centers

    def __getitem__(self, key):
        return self._fields[key]

    def cell_centers(self):
        return _FakeField(self._centers)


class _FakeFoamCase:
    source_instances = []
    clone_instances = []

    def __init__(self, _path: Path):
        self.path = Path(_path)
        moving_wall = {"movingWall": _FakeBoundaryValue(np.array([1.0, 0.0, 0.0]))}
        self._time0 = _FakeTimeDirectory(
            time=0.0,
            fields={
                "U": _FakeField(np.zeros((3, 3)), moving_wall),
                "p": _FakeField(np.zeros(3)),
            },
            centers=np.array(
                [
                    [0.10, 0.20, 0.0],
                    [0.25, 0.30, 0.0],
                    [0.70, 0.80, 0.0],
                ]
            ),
        )
        self._time1 = _FakeTimeDirectory(
            time=0.5,
            fields={
                "U": _FakeField(
                    np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.5, 0.1, 0.0],
                            [0.2, -0.1, 0.0],
                        ]
                    )
                ),
                "p": _FakeField(np.array([0.2, 0.1, -0.1])),
            },
            centers=np.array(
                [
                    [0.10, 0.20, 0.0],
                    [0.25, 0.30, 0.0],
                    [0.70, 0.80, 0.0],
                ]
            ),
        )
        self.block_mesh_dict = {"scale": 0.1}
        self.transport_properties = _FakeTransportProperties(nu=0.01)
        self.block_mesh_called = False
        self.run_calls = []
        self.clone_paths = []
        self.use_solved_time = True
        self.source_instances.append(self)

    def __getitem__(self, idx):
        if idx in (0, "0", 0.0):
            return self._time0
        if idx == -1 and not self.use_solved_time:
            return self._time0
        if idx in (-1, "0.5", 0.5):
            return self._time1
        raise KeyError(idx)

    def copy(self, _dst):
        raise AssertionError("run_case_for_viscosity should clone clean cases")

    def clone(self, dst):
        self.clone_paths.append(Path(dst))
        clone = _FakeFoamCase(dst)
        clone.source_instances.remove(clone)
        clone.clone_paths = []
        self.clone_instances.append(clone)
        return clone

    def block_mesh(self):
        self.block_mesh_called = True

    def run(self, cmd):
        self.run_calls.append(cmd)


def test_reynolds_and_viscosity_validation():
    metadata = CavityCaseMetaData(lid_velocity=1.0, length_scale=0.1)
    limits = CavityPhysicalLimits(
        nu_min=1.0e-3, nu_max=1.0e-2, reynolds_min=10.0, reynolds_max=100.0
    )

    assert compute_reynolds_number(1.0e-2, 1.0, 0.1) == pytest.approx(10.0)
    assert compute_reynolds_number(1.0e-3, 1.0, 0.1) == pytest.approx(100.0)
    assert validate_viscosity_and_reynolds(5.0e-3, metadata, limits) == pytest.approx(20.0)

    with pytest.raises(ValueError, match="Viscosity"):
        validate_viscosity_and_reynolds(5.0e-4, metadata, limits)


def test_build_deeponet_defaults():
    model = build_deeponet(CavityDeepONetConfig())
    assert model.activation_fn == "gelu"
    assert model.velocity_dim == 3
    assert model.out_features == 4
    assert model.output_channel_names == ("u", "v", "w", "p")


def test_build_sample_tensors_shapes_and_pressure_last():
    case = _FakeFoamCase(Path("."))
    sample = build_sample_tensors(case=case, nu=1.0e-3)

    assert sample.branch_input.shape == (3, 1)
    assert sample.trunk_input.shape == (3, 3)
    assert sample.target.shape == (3, 4)
    assert sample.target[:, -1].tolist() == pytest.approx([0.2, 0.1, -0.1])
    assert sample.trunk_input[:, 2].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_build_sample_tensors_rejects_unsolved_latest_time():
    case = _FakeFoamCase(Path("."))
    case.use_solved_time = False

    with pytest.raises(ValueError, match="latest output time"):
        build_sample_tensors(case=case, nu=1.0e-3)


def test_build_sample_tensors_rejects_velocity_length_mismatch():
    case = _FakeFoamCase(Path("."))
    case._time1._fields["U"] = _FakeField(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="Velocity field length"):
        build_sample_tensors(case=case, nu=1.0e-3)


def test_openfoam_environment_check(monkeypatch):
    def _fake_which(cmd):
        if cmd == "postProcess":
            return None
        return f"/usr/bin/{cmd}"

    monkeypatch.setattr("examples.cfd.cavity_of.deeponet_cavity.shutil.which", _fake_which)
    with pytest.raises(RuntimeError, match="Missing required OpenFOAM commands"):
        ensure_openfoam_environment()


def test_run_case_for_viscosity_with_mocked_foamlib(monkeypatch, tmp_path):
    _FakeFoamCase.source_instances = []
    _FakeFoamCase.clone_instances = []
    monkeypatch.setattr(
        "examples.cfd.cavity_of.deeponet_cavity._get_foam_case_cls", lambda: _FakeFoamCase
    )
    monkeypatch.setattr(
        "examples.cfd.cavity_of.deeponet_cavity.ensure_openfoam_environment", lambda: None
    )

    cfg = CavityDeepONetConfig(
        case_path=str(tmp_path / "cavity"),
        run_root=str(tmp_path / "runs"),
        run_openfoam=True,
        viscosity_values=(1.0e-3,),
    )
    sample = run_case_for_viscosity(nu=1.0e-3, config=cfg, sample_index=0)

    assert sample.target.shape[-1] == 4
    assert sample.reynolds_number == pytest.approx(100.0)
    source_case = _FakeFoamCase.source_instances[0]
    cloned_case = _FakeFoamCase.clone_instances[0]
    assert source_case.clone_paths == [tmp_path / "runs" / "nu_000_0.001"]
    assert cloned_case.transport_properties["nu"] == pytest.approx(1.0e-3)
    assert cloned_case.block_mesh_called
    assert cloned_case.run_calls == ["icoFoam"]
