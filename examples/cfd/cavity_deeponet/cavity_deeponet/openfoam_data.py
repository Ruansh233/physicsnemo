# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import CavityCaseMetaData, CavityDeepONetConfig, CavityPhysicalLimits
from .splits import build_split_manifest, save_split_manifest

REQUIRED_OPENFOAM_COMMANDS = ("blockMesh", "icoFoam", "postProcess")


@dataclass
class CavitySampleTensors:
    branch_input: Tensor
    trunk_input: Tensor
    target: Tensor
    reynolds_number: float


def get_foam_case_cls() -> Any:
    try:
        from foamlib import FoamCase
    except ImportError as exc:
        raise ImportError(
            "foamlib is required for examples/cfd/cavity_deeponet. "
            "Install with `pip install -r examples/cfd/cavity_deeponet/requirements.txt`."
        ) from exc
    return FoamCase


def ensure_openfoam_environment(
    commands: Sequence[str] = REQUIRED_OPENFOAM_COMMANDS,
) -> None:
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing required OpenFOAM commands: {missing_str}. "
            "Activate OpenFOAM first, for example with "
            "`source /home/shenhui_ruan/OpenFOAM/OpenFOAM-v2312/etc/bashrc`."
        )


def compute_reynolds_number(
    nu: float, lid_velocity: float, length_scale: float
) -> float:
    if nu <= 0.0:
        raise ValueError(f"Expected nu > 0, but got {nu}")
    if lid_velocity <= 0.0:
        raise ValueError(f"Expected lid_velocity > 0, but got {lid_velocity}")
    if length_scale <= 0.0:
        raise ValueError(f"Expected length_scale > 0, but got {length_scale}")
    return (lid_velocity * length_scale) / nu


def validate_viscosity_and_reynolds(
    nu: float,
    metadata: CavityCaseMetaData,
    limits: CavityPhysicalLimits,
) -> float:
    if not limits.nu_min <= nu <= limits.nu_max:
        raise ValueError(
            f"Viscosity nu={nu} is outside valid range [{limits.nu_min}, {limits.nu_max}]."
        )

    reynolds = compute_reynolds_number(
        nu=nu,
        lid_velocity=metadata.lid_velocity,
        length_scale=metadata.length_scale,
    )
    if not limits.reynolds_min <= reynolds <= limits.reynolds_max:
        raise ValueError(
            "Derived Reynolds number "
            f"Re={reynolds} is outside valid range [{limits.reynolds_min}, {limits.reynolds_max}] "
            f"for nu={nu}."
        )
    return reynolds


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

    if lid_velocity <= 0.0:
        raise ValueError(f"Expected positive movingWall speed, but got {lid_velocity}.")
    if length_scale <= 0.0:
        raise ValueError(f"Expected positive blockMesh scale, but got {length_scale}.")

    return CavityCaseMetaData(lid_velocity=lid_velocity, length_scale=length_scale)


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


def _shared_poly_mesh_path(case: Any) -> Path:
    return Path(case.path) / "constant" / "polyMesh"


def _reuse_reference_poly_mesh(source_case: Any, cloned_case: Any) -> None:
    source_poly_mesh = _shared_poly_mesh_path(source_case)
    if not source_poly_mesh.is_dir():
        raise FileNotFoundError(
            "Expected reference mesh directory at "
            f"{source_poly_mesh}, but it was not found."
        )

    cloned_poly_mesh = _shared_poly_mesh_path(cloned_case)
    if cloned_poly_mesh.is_symlink() or cloned_poly_mesh.is_file():
        cloned_poly_mesh.unlink()
    elif cloned_poly_mesh.exists():
        shutil.rmtree(cloned_poly_mesh)

    cloned_poly_mesh.parent.mkdir(parents=True, exist_ok=True)
    cloned_poly_mesh.symlink_to(source_poly_mesh, target_is_directory=True)


def build_sample_tensors(
    case: Any, nu: float, dtype: torch.dtype = torch.float32
) -> CavitySampleTensors:
    last_time = case[-1]
    query_time = float(last_time.time)
    if query_time <= 0.0:
        raise ValueError(
            "Expected latest output time to be greater than 0.0 before building "
            f"training tensors, but got latest output time {query_time}."
        )

    centers = _to_numpy(last_time.cell_centers().internal_field).astype(float)
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError(
            f"Unsupported cell center shape {centers.shape}; expected (N,3)-like array."
        )

    num_cells = centers.shape[0]
    velocity = _as_vector_field(last_time["U"].internal_field, num_cells)
    pressure = _as_scalar_field(last_time["p"].internal_field, num_cells)
    if velocity.shape[0] != num_cells:
        raise ValueError(
            f"Velocity field length {velocity.shape[0]} does not match cell count {num_cells}."
        )
    if pressure.shape[0] != num_cells:
        raise ValueError(
            f"Pressure field length {pressure.shape[0]} does not match cell count {num_cells}."
        )

    trunk_np = np.column_stack(
        (centers[:, 0], centers[:, 1], np.full((num_cells,), query_time, dtype=float))
    )
    branch_np = np.full((num_cells, 1), nu, dtype=float)
    target_np = np.column_stack(
        (velocity[:, 0], velocity[:, 1], velocity[:, 2], pressure)
    )

    return CavitySampleTensors(
        branch_input=torch.tensor(branch_np, dtype=dtype),
        trunk_input=torch.tensor(trunk_np, dtype=dtype),
        target=torch.tensor(target_np, dtype=dtype),
        reynolds_number=math.nan,
    )


def run_case_for_viscosity(
    nu: float,
    config: CavityDeepONetConfig,
    *,
    sample_index: int,
) -> CavitySampleTensors:
    limits = CavityPhysicalLimits(
        nu_min=config.nu_min,
        nu_max=config.nu_max,
        reynolds_min=config.reynolds_min,
        reynolds_max=config.reynolds_max,
    )

    FoamCase = get_foam_case_cls()
    source_case = FoamCase(Path(config.case_path))
    metadata = extract_case_metadata(source_case)
    reynolds = validate_viscosity_and_reynolds(nu=nu, metadata=metadata, limits=limits)

    run_root = Path(config.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    run_path = run_root / f"nu_{sample_index:03d}_{nu:.6g}"
    if run_path.exists():
        shutil.rmtree(run_path)

    cloned_case = source_case.clone(run_path)
    with cloned_case.transport_properties as transport_properties:
        transport_properties["nu"] = float(nu)

    _reuse_reference_poly_mesh(source_case, cloned_case)

    if config.run_openfoam:
        ensure_openfoam_environment()
        cloned_case.run("icoFoam")

    sample = build_sample_tensors(
        cloned_case, nu=nu, dtype=getattr(torch, config.dtype)
    )
    sample.reynolds_number = reynolds
    return sample


def generate_dataset(config: CavityDeepONetConfig) -> tuple[Tensor, Tensor, Tensor]:
    if config.clean_run_root:
        run_root = Path(config.run_root)
        if run_root.exists():
            shutil.rmtree(run_root)

    branch_batches: list[Tensor] = []
    trunk_batches: list[Tensor] = []
    target_batches: list[Tensor] = []
    for idx, nu in enumerate(config.viscosity_values):
        sample = run_case_for_viscosity(nu=nu, config=config, sample_index=idx)
        branch_batches.append(sample.branch_input)
        trunk_batches.append(sample.trunk_input)
        target_batches.append(sample.target)

    return (
        torch.cat(branch_batches, dim=0),
        torch.cat(trunk_batches, dim=0),
        torch.cat(target_batches, dim=0),
    )


def generate_split_datasets(
    config: CavityDeepONetConfig,
) -> tuple[dict[str, tuple[Tensor, Tensor, Tensor]], dict[str, object]]:
    if config.clean_run_root:
        run_root = Path(config.run_root)
        if run_root.exists():
            shutil.rmtree(run_root)

    manifest = build_split_manifest(
        seed=config.viscosity_seed,
        train_count=config.train_case_count,
        validate_count=config.validate_case_count,
        test_count=config.test_case_count,
        nu_min=config.nu_min,
        nu_max=config.nu_max,
    )
    manifest_path = Path(config.run_root) / config.manifest_filename
    save_split_manifest(manifest_path, manifest)
    print(f"Saved viscosity split manifest to: {manifest_path}")

    split_tensors: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    sample_index = 0
    for split_name in ("train", "validate", "test"):
        split_config = replace(config, run_root=str(Path(config.run_root) / split_name))
        branch_batches: list[Tensor] = []
        trunk_batches: list[Tensor] = []
        target_batches: list[Tensor] = []
        for nu in manifest[split_name]:
            sample = run_case_for_viscosity(
                nu=float(nu),
                config=split_config,
                sample_index=sample_index,
            )
            sample_index += 1
            branch_batches.append(sample.branch_input)
            trunk_batches.append(sample.trunk_input)
            target_batches.append(sample.target)
        if not branch_batches:
            raise ValueError(f"Split '{split_name}' is empty; increase case counts.")
        split_tensors[split_name] = (
            torch.cat(branch_batches, dim=0),
            torch.cat(trunk_batches, dim=0),
            torch.cat(target_batches, dim=0),
        )

    return split_tensors, manifest
