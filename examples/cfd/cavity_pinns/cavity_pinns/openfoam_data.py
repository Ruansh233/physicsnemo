# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import CavityCaseMetaData, CavityPINNConfig

REQUIRED_OPENFOAM_COMMANDS = ("blockMesh", "icoFoam", "postProcess")


@dataclass
class CavityCaseTensors:
    coordinates: Tensor
    viscosity: Tensor
    target: Tensor
    reynolds_number: float
    metadata: CavityCaseMetaData


def get_foam_case_cls() -> Any:
    try:
        foamlib = importlib.import_module("foamlib")
    except ImportError as exc:
        raise ImportError(
            "foamlib is required for examples/cfd/cavity_pinns. "
            "Install with `pip install -r examples/cfd/cavity_pinns/requirements.txt`."
        ) from exc
    if not hasattr(foamlib, "FoamCase"):
        raise ImportError("foamlib.FoamCase is not available in the installed foamlib package.")
    return foamlib.FoamCase


def ensure_openfoam_environment(commands: Sequence[str] = REQUIRED_OPENFOAM_COMMANDS) -> None:
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing required OpenFOAM commands: {missing_str}. "
            "Activate OpenFOAM first (for this workspace, run `of2312`)."
        )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_lid_velocity(moving_wall_value: Any) -> float:
    arr = _to_numpy(moving_wall_value).astype(float)
    if arr.ndim == 0:
        return abs(float(arr))
    if arr.ndim == 1 and arr.shape[0] >= 1:
        return float(np.linalg.norm(arr[:3]))
    raise ValueError(
        f"Unsupported movingWall velocity shape {arr.shape}; expected scalar or vector."
    )


def extract_case_metadata(case: Any) -> CavityCaseMetaData:
    moving_wall = case[0]["U"].boundary_field["movingWall"].value
    lid_velocity = _extract_lid_velocity(moving_wall)
    length_scale = float(case.block_mesh_dict["scale"])
    if lid_velocity <= 0.0 or length_scale <= 0.0:
        raise ValueError("Expected positive lid velocity and length scale.")
    return CavityCaseMetaData(lid_velocity=lid_velocity, length_scale=length_scale)


def compute_reynolds_number(nu: float, metadata: CavityCaseMetaData) -> float:
    if nu <= 0.0:
        raise ValueError(f"Expected positive viscosity, got {nu}.")
    return (metadata.lid_velocity * metadata.length_scale) / nu


def _as_vector_field(value: Any, num_cells: int) -> np.ndarray:
    arr = _to_numpy(value).astype(float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    if arr.ndim == 1 and arr.shape[0] == 3:
        return np.repeat(arr[None, :], num_cells, axis=0)
    raise ValueError(
        f"Unsupported velocity field shape {arr.shape}; expected (N,3) or (3,)."
    )


def _as_scalar_field(value: Any, num_cells: int) -> np.ndarray:
    arr = _to_numpy(value).astype(float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 0:
        return np.full((num_cells,), float(arr), dtype=float)
    raise ValueError(
        f"Unsupported pressure field shape {arr.shape}; expected (N,) or scalar."
    )


def build_case_tensors(
    *,
    case: Any,
    nu: float,
    spatial_dim: int,
    dtype: torch.dtype,
) -> CavityCaseTensors:
    if spatial_dim not in (2, 3):
        raise ValueError(f"Unsupported spatial_dim={spatial_dim}; expected 2 or 3.")

    metadata = extract_case_metadata(case)
    reynolds = compute_reynolds_number(nu=nu, metadata=metadata)

    last_time = case[-1]
    query_time = float(last_time.time)
    if query_time <= 0.0:
        raise ValueError("Expected latest output time > 0 before building training tensors.")

    centers = _to_numpy(last_time.cell_centers().internal_field).astype(float)
    if centers.ndim != 2 or centers.shape[1] < 3:
        raise ValueError(f"Unsupported cell center shape {centers.shape}; expected (N,3).")
    num_cells = centers.shape[0]
    velocity = _as_vector_field(last_time["U"].internal_field, num_cells)
    pressure = _as_scalar_field(last_time["p"].internal_field, num_cells)
    if velocity.shape[0] != num_cells or pressure.shape[0] != num_cells:
        raise ValueError("Velocity and pressure sizes must match number of cells.")

    if spatial_dim == 2:
        coordinates_np = centers[:, :2]
        target_np = np.column_stack((velocity[:, 0], velocity[:, 1], pressure))
    else:
        coordinates_np = centers[:, :3]
        target_np = np.column_stack((velocity[:, 0], velocity[:, 1], velocity[:, 2], pressure))
    viscosity_np = np.full((num_cells, 1), float(nu), dtype=float)

    return CavityCaseTensors(
        coordinates=torch.tensor(coordinates_np, dtype=dtype),
        viscosity=torch.tensor(viscosity_np, dtype=dtype),
        target=torch.tensor(target_np, dtype=dtype),
        reynolds_number=reynolds,
        metadata=metadata,
    )


def run_case_for_viscosity(
    *,
    nu: float,
    config: CavityPINNConfig,
    sample_index: int,
) -> CavityCaseTensors:
    foam_case_cls = get_foam_case_cls()
    source_case = foam_case_cls(Path(config.case_path))
    metadata = extract_case_metadata(source_case)
    reynolds = compute_reynolds_number(nu=nu, metadata=metadata)
    if not config.reynolds_min <= reynolds <= config.reynolds_max:
        raise ValueError(
            f"Derived Reynolds number {reynolds} is outside [{config.reynolds_min}, {config.reynolds_max}] for nu={nu}."
        )

    run_root = Path(config.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    run_path = run_root / f"nu_{sample_index:03d}_{nu:.6g}"
    if run_path.exists():
        shutil.rmtree(run_path)

    cloned_case = source_case.clone(run_path)
    with cloned_case.transport_properties as transport_properties:
        transport_properties["nu"] = float(nu)

    if config.run_openfoam:
        ensure_openfoam_environment()
        cloned_case.block_mesh()
        cloned_case.run("icoFoam")

    sample = build_case_tensors(
        case=cloned_case,
        nu=nu,
        spatial_dim=config.spatial_dim,
        dtype=getattr(torch, config.dtype),
    )
    if not math.isclose(sample.reynolds_number, reynolds, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Unexpected Reynolds mismatch between sampled tensors and metadata.")
    return sample


def generate_cases_for_viscosities(
    *,
    config: CavityPINNConfig,
    viscosity_values: Sequence[float],
) -> list[CavityCaseTensors]:
    cases: list[CavityCaseTensors] = []
    for idx, nu in enumerate(viscosity_values):
        cases.append(run_case_for_viscosity(nu=float(nu), config=config, sample_index=idx))
    return cases
