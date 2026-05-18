# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.cfd.cavity_pinns.cavity_pinns.config import (
    CavityCaseMetaData,
    CavityPINNConfig,
)
from examples.cfd.cavity_pinns.cavity_pinns.modeling import CavityPINN
from examples.cfd.cavity_pinns.cavity_pinns.openfoam_data import (
    CavityCaseTensors,
    build_case_tensors,
    ensure_openfoam_environment,
    get_foam_case_cls,
    run_case_for_viscosity,
)
from examples.cfd.cavity_pinns.cavity_pinns.physics import (
    compute_physics_residuals,
    make_physics_informer,
)
from examples.cfd.cavity_pinns.cavity_pinns.splits import (
    build_split_manifest,
    derive_viscosity_bounds,
)
from examples.cfd.cavity_pinns.cavity_pinns.visualization import (
    save_split_triptych_visualizations,
)
from examples.cfd.cavity_pinns.cavity_pinns.workflow import (
    _build_boundary_condition_tensors,
    _build_normalization_stats,
    _load_trained_checkpoint,
    _predict_physical_fields,
    _save_trained_checkpoint,
)


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
    def __init__(self, _path: Path):
        moving_wall = {"movingWall": _FakeBoundaryValue(np.array([1.0, 0.0, 0.0]))}
        self._time0 = _FakeTimeDirectory(
            time=0.0,
            fields={
                "U": _FakeField(np.zeros((3, 3)), moving_wall),
                "p": _FakeField(np.zeros(3)),
            },
            centers=np.array(
                [
                    [0.1, 0.2, 0.0],
                    [0.3, 0.4, 0.0],
                    [0.7, 0.8, 0.0],
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
                            [0.4, 0.2, 0.0],
                            [0.1, -0.1, 0.0],
                        ]
                    )
                ),
                "p": _FakeField(np.array([0.2, 0.1, -0.3])),
            },
            centers=np.array(
                [
                    [0.1, 0.2, 0.0],
                    [0.3, 0.4, 0.0],
                    [0.7, 0.8, 0.0],
                ]
            ),
        )
        self.block_mesh_dict = {"scale": 0.1}

    def __getitem__(self, idx):
        if idx in (0, "0", 0.0):
            return self._time0
        if idx in (-1, "0.5", 0.5):
            return self._time1
        raise KeyError(idx)


def test_run_case_for_viscosity_reuses_reference_mesh_symlink(tmp_path, monkeypatch):
    source_case_path = tmp_path / "source_case"
    source_mesh_path = source_case_path / "constant" / "polyMesh"
    source_mesh_path.mkdir(parents=True)
    (source_mesh_path / "points").write_text("mesh", encoding="utf-8")

    class _FakeTransportProperties:
        def __init__(self):
            self.data = {"nu": None}

        def __enter__(self):
            return self.data

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeRunCase:
        def __init__(self, path: Path):
            self.path = Path(path)
            self.block_mesh_dict = {"scale": 0.1}
            self.transport_properties = _FakeTransportProperties()
            moving_wall = {"movingWall": _FakeBoundaryValue(np.array([1.0, 0.0, 0.0]))}
            self._time0 = _FakeTimeDirectory(
                time=0.0,
                fields={
                    "U": _FakeField(np.zeros((1, 3)), moving_wall),
                    "p": _FakeField(np.zeros(1)),
                },
                centers=np.array([[0.0, 0.0, 0.0]]),
            )

        def clone(self, dst):
            shutil.copytree(self.path, dst, symlinks=True)
            return type(self)(dst)

        def __getitem__(self, idx):
            if idx in (0, "0", 0.0):
                return self._time0
            raise KeyError(idx)

        def block_mesh(self):
            raise AssertionError("blockMesh should not run for each generated case")

        def run(self, command):
            self.last_command = command

    expected_sample = CavityCaseTensors(
        coordinates=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        viscosity=torch.tensor([[1.0e-3]], dtype=torch.float32),
        target=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        reynolds_number=100.0,
        metadata=CavityCaseMetaData(lid_velocity=1.0, length_scale=0.1),
    )

    monkeypatch.setattr(
        "examples.cfd.cavity_pinns.cavity_pinns.openfoam_data.get_foam_case_cls",
        lambda: _FakeRunCase,
    )
    monkeypatch.setattr(
        "examples.cfd.cavity_pinns.cavity_pinns.openfoam_data.ensure_openfoam_environment",
        lambda: None,
    )
    monkeypatch.setattr(
        "examples.cfd.cavity_pinns.cavity_pinns.openfoam_data.build_case_tensors",
        lambda **kwargs: expected_sample,
    )

    config = CavityPINNConfig(
        case_path=str(source_case_path),
        run_root=str(tmp_path / "runs"),
        run_openfoam=True,
        spatial_dim=2,
        device="cpu",
    )

    result = run_case_for_viscosity(nu=1.0e-3, config=config, sample_index=0)

    mesh_path = Path(config.run_root) / "nu_000_0.001" / "constant" / "polyMesh"
    assert mesh_path.is_symlink()
    assert mesh_path.resolve() == source_mesh_path.resolve()
    torch.testing.assert_close(result.coordinates, expected_sample.coordinates)


def test_openfoam_environment_check(monkeypatch):
    def _fake_which(cmd: str):
        return None if cmd == "postProcess" else f"/usr/bin/{cmd}"

    monkeypatch.setattr(
        "examples.cfd.cavity_pinns.cavity_pinns.openfoam_data.shutil.which", _fake_which
    )
    with pytest.raises(RuntimeError, match="of2312"):
        ensure_openfoam_environment()


def test_get_foam_case_cls_reports_missing_foamlib(monkeypatch):
    def _fake_import(name: str):
        if name == "foamlib":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(
        "examples.cfd.cavity_pinns.cavity_pinns.openfoam_data.importlib.import_module",
        _fake_import,
    )
    with pytest.raises(ImportError, match="foamlib is required"):
        get_foam_case_cls()


def test_reynolds_bounds_and_split_sampling_are_reproducible():
    metadata = CavityCaseMetaData(lid_velocity=1.0, length_scale=0.1)
    nu_min, nu_max = derive_viscosity_bounds(
        metadata=metadata,
        reynolds_min=10.0,
        reynolds_max=100.0,
    )
    assert nu_min == pytest.approx(1.0e-3)
    assert nu_max == pytest.approx(1.0e-2)

    manifest_a = build_split_manifest(
        seed=7,
        train_count=3,
        val_count=2,
        test_count=2,
        unseen_count=2,
        nu_min=nu_min,
        nu_max=nu_max,
    )
    manifest_b = build_split_manifest(
        seed=7,
        train_count=3,
        val_count=2,
        test_count=2,
        unseen_count=2,
        nu_min=nu_min,
        nu_max=nu_max,
    )

    assert manifest_a == manifest_b
    seen = set(manifest_a["train"] + manifest_a["validate"] + manifest_a["test"])
    unseen = set(manifest_a["unseen"])
    assert seen.isdisjoint(unseen)
    train_min = min(manifest_a["train"])
    train_max = max(manifest_a["train"])
    heldout = manifest_a["validate"] + manifest_a["test"]
    assert heldout
    for value in heldout:
        assert train_min < value < train_max


def test_split_manifest_raises_when_interpolation_cannot_be_satisfied():
    with pytest.raises(ValueError, match="train_count >= 2"):
        build_split_manifest(
            seed=3,
            train_count=1,
            val_count=1,
            test_count=0,
            unseen_count=0,
            nu_min=1.0e-3,
            nu_max=1.0e-2,
        )


def test_build_case_tensors_2d_shapes_and_channels():
    case = _FakeFoamCase(Path("."))
    sample = build_case_tensors(
        case=case, nu=1.0e-3, spatial_dim=2, dtype=torch.float32
    )

    assert sample.coordinates.shape == (3, 2)
    assert sample.viscosity.shape == (3, 1)
    assert sample.target.shape == (3, 3)
    assert sample.target[:, -1].tolist() == pytest.approx([0.2, 0.1, -0.3])


def test_boundary_condition_tensor_sampling():
    case = _FakeFoamCase(Path("."))
    sample = build_case_tensors(
        case=case, nu=1.0e-3, spatial_dim=2, dtype=torch.float32
    )
    boundary = _build_boundary_condition_tensors([sample], points_per_wall=4)

    assert boundary.coordinates.shape == (16, 2)
    assert boundary.viscosity.shape == (16, 1)
    assert boundary.target_velocity.shape == (16, 2)
    assert boundary.target_velocity[:4, 0].tolist() == pytest.approx([1.0] * 4)
    assert torch.count_nonzero(boundary.target_velocity[4:]).item() == 0


def test_boundary_condition_tensors_use_physical_walls_not_cell_centers():
    sample = CavityCaseTensors(
        coordinates=torch.tensor(
            [
                [0.025, 0.025],
                [0.075, 0.025],
                [0.025, 0.075],
                [0.075, 0.075],
            ],
            dtype=torch.float32,
        ),
        viscosity=torch.full((4, 1), 1.0e-3),
        target=torch.zeros((4, 3)),
        reynolds_number=100.0,
        metadata=CavityCaseMetaData(lid_velocity=1.0, length_scale=0.1),
    )

    boundary = _build_boundary_condition_tensors([sample], points_per_wall=3)

    assert boundary.coordinates[:, 0].min().item() == pytest.approx(0.0)
    assert boundary.coordinates[:, 0].max().item() == pytest.approx(0.1)
    assert boundary.coordinates[:, 1].min().item() == pytest.approx(0.0)
    assert boundary.coordinates[:, 1].max().item() == pytest.approx(0.1)
    assert boundary.target_velocity[:3, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert torch.count_nonzero(boundary.target_velocity[3:]).item() == 0


def test_normalized_prediction_returns_physical_field_values():
    coordinates = torch.tensor([[0.0, 0.0], [0.1, 0.1]], dtype=torch.float32)
    viscosity = torch.tensor([[1.0e-3], [1.0e-2]], dtype=torch.float32)
    target = torch.tensor([[1.0, -1.0, 10.0], [3.0, 1.0, 20.0]], dtype=torch.float32)
    normalizer = _build_normalization_stats(coordinates, viscosity, target)

    class _ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)

    prediction = _predict_physical_fields(
        model=_ZeroModel(),
        normalizer=normalizer,
        coordinates=coordinates,
        viscosity=viscosity,
    )

    torch.testing.assert_close(
        prediction,
        torch.tensor([[2.0, 0.0, 15.0], [2.0, 0.0, 15.0]], dtype=torch.float32),
    )


def test_physics_residual_api_keys_and_shapes():
    informer = make_physics_informer(spatial_dim=2, device="cpu")
    coords = torch.rand(8, 2, requires_grad=True)
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    prediction = torch.cat((x + y, x * y, x - y), dim=1)
    viscosity = torch.full((8, 1), 1.0e-2)

    residuals = compute_physics_residuals(
        physics_informer=informer,
        coordinates=coords,
        prediction=prediction,
        viscosity=viscosity,
        spatial_dim=2,
    )

    assert set(residuals.keys()) == {"continuity", "momentum_x", "momentum_y"}
    for value in residuals.values():
        assert value.shape == (8, 1)


def test_visualization_writes_triptych_outputs(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    case = _FakeFoamCase(Path("."))
    sample = build_case_tensors(
        case=case, nu=1.0e-3, spatial_dim=2, dtype=torch.float32
    )
    model = CavityPINN(
        in_features=3,
        out_features=3,
        hidden_layers=1,
        hidden_size=8,
        activation_fn="tanh",
    )
    config = CavityPINNConfig(
        save_visualizations=True,
        spatial_dim=2,
        visualization_max_cases=1,
        visualization_dir=str(tmp_path),
        device="cpu",
    )

    save_split_triptych_visualizations(
        model=model,
        cases=[sample],
        config=config,
        split_name="test",
    )

    generated = sorted((tmp_path / "test").glob("*.png"))
    assert len(generated) == 3


def test_checkpoint_save_and_load_round_trip(tmp_path):
    config = CavityPINNConfig(model_layers=1, model_layer_size=8, spatial_dim=2, device="cpu")
    model = CavityPINN(
        in_features=3,
        out_features=3,
        hidden_layers=1,
        hidden_size=8,
        activation_fn="tanh",
    )
    normalizer = _build_normalization_stats(
        coordinates=torch.tensor([[0.0, 0.0], [0.1, 0.1]], dtype=torch.float32),
        viscosity=torch.tensor([[1.0e-3], [1.0e-2]], dtype=torch.float32),
        target=torch.tensor([[1.0, -1.0, 10.0], [3.0, 1.0, 20.0]], dtype=torch.float32),
    )
    ckpt = tmp_path / "model.pt"

    _save_trained_checkpoint(
        checkpoint_path=ckpt,
        model=model,
        normalizer=normalizer,
        config=config,
    )

    loaded_model, loaded_normalizer = _load_trained_checkpoint(
        checkpoint_path=ckpt,
        config=config,
    )
    assert isinstance(loaded_model, CavityPINN)
    torch.testing.assert_close(normalizer.input_mean, loaded_normalizer.input_mean)
    torch.testing.assert_close(normalizer.input_scale, loaded_normalizer.input_scale)
    torch.testing.assert_close(normalizer.output_mean, loaded_normalizer.output_mean)
    torch.testing.assert_close(normalizer.output_scale, loaded_normalizer.output_scale)
