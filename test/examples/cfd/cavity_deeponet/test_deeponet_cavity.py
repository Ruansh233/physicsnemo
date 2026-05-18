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
import torch

from examples.cfd.cavity_deeponet.cavity_deeponet.splits import build_split_manifest
from examples.cfd.cavity_deeponet.cavity_deeponet.modeling import (
    load_checkpoint,
    save_checkpoint,
    train_deeponet,
)
from examples.cfd.cavity_deeponet.deeponet_cavity import (
    CavityCaseMetaData,
    CavityDeepONetConfig,
    CavityPhysicalLimits,
    CavitySampleTensors,
    TensorNormalizer,
    build_deeponet,
    build_sample_tensors,
    compute_relative_l2,
    compute_reynolds_number,
    ensure_openfoam_environment,
    generate_split_datasets,
    run_case_for_viscosity,
    validate_viscosity_and_reynolds,
)


def test_cavity_deeponet_package_exports_match_wrapper_module():
    from examples.cfd.cavity_deeponet.cavity_deeponet import (
        CavityDeepONetConfig as PackageConfig,
    )

    assert PackageConfig().activation_fn == CavityDeepONetConfig().activation_fn


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
    assert validate_viscosity_and_reynolds(5.0e-3, metadata, limits) == pytest.approx(
        20.0
    )

    with pytest.raises(ValueError, match="Viscosity"):
        validate_viscosity_and_reynolds(5.0e-4, metadata, limits)


def test_split_manifest_samples_50_cases_reproducibly():
    manifest_a = build_split_manifest(
        seed=7,
        train_count=40,
        validate_count=5,
        test_count=5,
        nu_min=1.0e-3,
        nu_max=1.0e-2,
    )
    manifest_b = build_split_manifest(
        seed=7,
        train_count=40,
        validate_count=5,
        test_count=5,
        nu_min=1.0e-3,
        nu_max=1.0e-2,
    )

    assert manifest_a == manifest_b
    assert len(manifest_a["train"]) == 40
    assert len(manifest_a["validate"]) == 5
    assert len(manifest_a["test"]) == 5
    all_values = manifest_a["train"] + manifest_a["validate"] + manifest_a["test"]
    assert len(all_values) == 50
    assert len(set(all_values)) == 50

    train_min = min(manifest_a["train"])
    train_max = max(manifest_a["train"])
    for value in manifest_a["validate"] + manifest_a["test"]:
        assert train_min < value < train_max


def test_generate_split_datasets_writes_manifest_and_uses_all_splits(
    monkeypatch, tmp_path
):
    calls = []

    def _fake_run_case_for_viscosity(nu, config, *, sample_index):
        calls.append((nu, sample_index, config.run_root))
        value = float(sample_index)
        return CavitySampleTensors(
            branch_input=torch.tensor([[nu]], dtype=torch.float32),
            trunk_input=torch.tensor(
                [[value, value + 1.0, value + 2.0]], dtype=torch.float32
            ),
            target=torch.tensor(
                [[value, value + 0.1, value + 0.2, value + 0.3]], dtype=torch.float32
            ),
            reynolds_number=1.0,
        )

    monkeypatch.setattr(
        "examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data.run_case_for_viscosity",
        _fake_run_case_for_viscosity,
    )

    cfg = CavityDeepONetConfig(
        run_root=str(tmp_path / "runs"),
        train_case_count=40,
        validate_case_count=5,
        test_case_count=5,
        manifest_filename="split.json",
    )
    split_tensors, manifest = generate_split_datasets(cfg)

    assert len(calls) == 50
    assert [sample_index for _, sample_index, _ in calls] == list(range(50))
    assert {Path(run_root).name for _, _, run_root in calls} == {
        "train",
        "validate",
        "test",
    }
    assert len(manifest["train"]) == 40
    assert len(manifest["validate"]) == 5
    assert len(manifest["test"]) == 5
    assert (tmp_path / "runs" / "split.json").is_file()
    assert split_tensors["train"][0].shape == (40, 1)
    assert split_tensors["validate"][0].shape == (5, 1)
    assert split_tensors["test"][0].shape == (5, 1)


def test_split_manifest_raises_when_interpolation_cannot_be_satisfied():
    with pytest.raises(ValueError, match="train_count >= 2"):
        build_split_manifest(
            seed=3,
            train_count=1,
            validate_count=1,
            test_count=0,
            nu_min=1.0e-3,
            nu_max=1.0e-2,
        )


def test_build_deeponet_defaults():
    model = build_deeponet(CavityDeepONetConfig())
    assert model.activation_fn == "gelu"
    assert model.velocity_dim == 3
    assert model.out_features == 4
    assert model.output_channel_names == ("u", "v", "w", "p")


def test_checkpoint_save_and_load_round_trip(tmp_path):
    cfg = CavityDeepONetConfig()
    model = build_deeponet(cfg)
    branch = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    trunk = torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=torch.float32)
    target = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]], dtype=torch.float32
    )
    branch_norm = TensorNormalizer.fit(branch)
    trunk_norm = TensorNormalizer.fit(trunk)
    target_norm = TensorNormalizer.fit(target)

    checkpoint_path = tmp_path / "deeponet_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        branch_normalizer=branch_norm,
        trunk_normalizer=trunk_norm,
        target_normalizer=target_norm,
    )

    reloaded_model = build_deeponet(cfg)
    loaded_branch, loaded_trunk, loaded_target = load_checkpoint(
        checkpoint_path,
        reloaded_model,
        device="cpu",
        dtype=torch.float32,
    )
    assert torch.allclose(loaded_branch.mean, branch_norm.mean)
    assert torch.allclose(loaded_branch.std, branch_norm.std)
    assert torch.allclose(loaded_trunk.mean, trunk_norm.mean)
    assert torch.allclose(loaded_trunk.std, trunk_norm.std)
    assert torch.allclose(loaded_target.mean, target_norm.mean)
    assert torch.allclose(loaded_target.std, target_norm.std)


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


def test_tensor_normalizer_handles_constant_channels():
    import torch

    tensor = torch.tensor(
        [
            [1.0, 0.5],
            [2.0, 0.5],
            [3.0, 0.5],
        ]
    )
    normalizer = TensorNormalizer.fit(tensor)
    normalized = normalizer.transform(tensor)

    assert normalizer.std[:, 1].item() == pytest.approx(1.0)
    assert normalized[:, 1].tolist() == pytest.approx([0.0, 0.0, 0.0])
    torch.testing.assert_close(normalizer.inverse(normalized), tensor)


def test_compute_relative_l2_uses_absolute_error_for_zero_channels():
    import torch

    target = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    prediction = torch.tensor([[1.0, 0.1], [2.0, -0.1]])

    aggregate, channel_errors = compute_relative_l2(prediction, target)

    assert aggregate > 0.0
    assert channel_errors[0] == pytest.approx(0.0)
    assert channel_errors[1] == pytest.approx(
        float(torch.linalg.vector_norm(prediction[:, 1]))
    )


def test_train_deeponet_prints_step_loss_log(capsys):
    model = build_deeponet(CavityDeepONetConfig(latent_dim=8, layer_size=16))
    branch = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    trunk = torch.tensor(
        [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=torch.float32
    )
    target = torch.tensor(
        [[1.0, 0.0, 0.0, 0.2], [0.9, 0.1, 0.0, 0.3], [0.8, 0.2, 0.0, 0.4]],
        dtype=torch.float32,
    )

    train_deeponet(
        model,
        branch_input=branch,
        trunk_input=trunk,
        target=target,
        lr=1.0e-3,
        steps=1,
        log_steps=1,
    )
    out = capsys.readouterr().out
    assert "step=00001 branch_loss=" in out
    assert "trunk_loss=" in out


def test_train_deeponet_realtime_plot_is_configurable(monkeypatch):
    calls = {"build": 0, "update": 0, "close": 0}

    def _fake_builder():
        calls["build"] += 1

        def _update(_step, _loss):
            calls["update"] += 1

        def _close():
            calls["close"] += 1

        return _update, _close

    monkeypatch.setattr(
        "examples.cfd.cavity_deeponet.cavity_deeponet.modeling._build_realtime_loss_plotter",
        _fake_builder,
    )

    model = build_deeponet(CavityDeepONetConfig(latent_dim=8, layer_size=16))
    branch = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    trunk = torch.tensor(
        [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=torch.float32
    )
    target = torch.tensor(
        [[1.0, 0.0, 0.0, 0.2], [0.9, 0.1, 0.0, 0.3], [0.8, 0.2, 0.0, 0.4]],
        dtype=torch.float32,
    )

    train_deeponet(
        model,
        branch_input=branch,
        trunk_input=trunk,
        target=target,
        lr=1.0e-3,
        steps=2,
        enable_realtime_loss_plot=False,
    )
    assert calls == {"build": 0, "update": 0, "close": 0}

    train_deeponet(
        model,
        branch_input=branch,
        trunk_input=trunk,
        target=target,
        lr=1.0e-3,
        steps=2,
        enable_realtime_loss_plot=True,
    )
    assert calls["build"] == 1
    assert calls["update"] == 2
    assert calls["close"] == 1


def test_openfoam_environment_check(monkeypatch):
    def _fake_which(cmd):
        if cmd == "postProcess":
            return None
        return f"/usr/bin/{cmd}"

    monkeypatch.setattr(
        "examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data.shutil.which",
        _fake_which,
    )
    with pytest.raises(RuntimeError, match="Missing required OpenFOAM commands"):
        ensure_openfoam_environment()


def test_run_case_for_viscosity_with_mocked_foamlib(monkeypatch, tmp_path):
    _FakeFoamCase.source_instances = []
    _FakeFoamCase.clone_instances = []
    mesh_reuse_calls = []
    monkeypatch.setattr(
        "examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data.get_foam_case_cls",
        lambda: _FakeFoamCase,
    )
    monkeypatch.setattr(
        "examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data.ensure_openfoam_environment",
        lambda: None,
    )
    monkeypatch.setattr(
        "examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data._reuse_reference_poly_mesh",
        lambda source_case, cloned_case: mesh_reuse_calls.append(
            (source_case.path, cloned_case.path)
        ),
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
    assert mesh_reuse_calls == [
        (tmp_path / "cavity", tmp_path / "runs" / "nu_000_0.001")
    ]
    assert not cloned_case.block_mesh_called
    assert cloned_case.run_calls == ["icoFoam"]


def test_run_case_for_viscosity_reuses_reference_poly_mesh(monkeypatch, tmp_path):
    class _FilesystemFoamCase(_FakeFoamCase):
        def __init__(self, path: Path):
            super().__init__(path)
            poly_mesh_dir = self.path / "constant" / "polyMesh"
            poly_mesh_dir.mkdir(parents=True, exist_ok=True)
            (poly_mesh_dir / "marker.txt").write_text("mesh", encoding="utf-8")

        def clone(self, dst):
            self.clone_paths.append(Path(dst))
            clone = _FilesystemFoamCase(dst)
            clone.source_instances.remove(clone)
            clone.clone_paths = []
            self.clone_instances.append(clone)
            return clone

    _FilesystemFoamCase.source_instances = []
    _FilesystemFoamCase.clone_instances = []
    monkeypatch.setattr(
        "examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data.get_foam_case_cls",
        lambda: _FilesystemFoamCase,
    )
    monkeypatch.setattr(
        "examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data.ensure_openfoam_environment",
        lambda: None,
    )

    cfg = CavityDeepONetConfig(
        case_path=str(tmp_path / "cavity"),
        run_root=str(tmp_path / "runs"),
        run_openfoam=True,
        viscosity_values=(1.0e-3,),
    )
    sample = run_case_for_viscosity(nu=1.0e-3, config=cfg, sample_index=0)

    source_case = _FilesystemFoamCase.source_instances[0]
    cloned_case = _FilesystemFoamCase.clone_instances[0]

    assert sample.reynolds_number == pytest.approx(100.0)
    assert not cloned_case.block_mesh_called
    assert cloned_case.run_calls == ["icoFoam"]
    poly_mesh_path = cloned_case.path / "constant" / "polyMesh"
    assert poly_mesh_path.is_symlink()
    assert poly_mesh_path.resolve() == (source_case.path / "constant" / "polyMesh").resolve()
    assert (poly_mesh_path / "marker.txt").read_text(encoding="utf-8") == "mesh"
