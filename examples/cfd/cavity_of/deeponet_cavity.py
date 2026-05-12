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

r"""DeepONet workflow for the OpenFOAM lid-driven cavity case.

This module uses ``foamlib`` for case manipulation and execution, and produces
training tensors for a DeepONet model that predicts velocity and pressure with
channel order ``(u, v, w, p)``.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from physicsnemo.experimental.models.deeponet import DeepONet

REQUIRED_OPENFOAM_COMMANDS = ("blockMesh", "icoFoam", "postProcess")


@dataclass(frozen=True)
class CavityPhysicalLimits:
    r"""Physical bounds for viscosity and Reynolds number."""

    nu_min: float = 1.0e-3
    nu_max: float = 1.0e-2
    reynolds_min: float = 10.0
    reynolds_max: float = 100.0


@dataclass(frozen=True)
class CavityCaseMetaData:
    r"""Cavity case metadata required for Reynolds-number computation."""

    lid_velocity: float
    length_scale: float


@dataclass
class CavityDeepONetConfig:
    r"""Configuration for cavity DeepONet data generation and model setup."""

    case_path: str = str(Path(__file__).resolve().parent / "cavity")
    run_root: str = str(Path(__file__).resolve().parent / ".foamlib_runs")
    run_openfoam: bool = True
    clean_run_root: bool = False
    viscosity_values: tuple[float, ...] = (1.0e-3, 5.0e-3, 1.0e-2)
    nu_min: float = 1.0e-3
    nu_max: float = 1.0e-2
    reynolds_min: float = 10.0
    reynolds_max: float = 100.0
    latent_dim: int = 128
    branch_layers: int = 4
    trunk_layers: int = 4
    layer_size: int = 128
    activation_fn: str = "gelu"
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-6
    train_steps: int = 1000
    lbfgs_steps: int = 200
    lbfgs_lr: float = 0.3
    normalize_inputs: bool = True
    normalize_targets: bool = True
    device: str = "cpu"
    dtype: str = "float32"
    save_visualizations: bool = True
    visualization_dir: str = str(Path(__file__).resolve().parent / "outputs" / "figures")
    visualization_max_cases: int = 3


@dataclass
class CavitySampleTensors:
    r"""Tensor bundle for one cavity sample."""

    branch_input: Tensor
    trunk_input: Tensor
    target: Tensor
    reynolds_number: float


@dataclass(frozen=True)
class TensorNormalizer:
    r"""Per-channel affine normalizer for example tensors."""

    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, tensor: Tensor) -> "TensorNormalizer":
        r"""Fit a channel-wise normalizer with safe handling of constant channels."""
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        std = torch.where(std < 1.0e-8, torch.ones_like(std), std)
        return cls(mean=mean, std=std)

    @classmethod
    def identity(cls, tensor: Tensor) -> "TensorNormalizer":
        r"""Build a no-op normalizer matching the tensor feature dimension."""
        feature_shape = (1, tensor.shape[-1])
        return cls(
            mean=torch.zeros(feature_shape, dtype=tensor.dtype, device=tensor.device),
            std=torch.ones(feature_shape, dtype=tensor.dtype, device=tensor.device),
        )

    def transform(self, tensor: Tensor) -> Tensor:
        r"""Normalize a tensor."""
        return (tensor - self.mean) / self.std

    def inverse(self, tensor: Tensor) -> Tensor:
        r"""Map a normalized tensor back to physical units."""
        return tensor * self.std + self.mean


def _get_foam_case_cls() -> Any:
    try:
        from foamlib import FoamCase
    except ImportError as exc:
        raise ImportError(
            "foamlib is required for examples/cfd/cavity_of. "
            "Install with `pip install -r examples/cfd/cavity_of/requirements.txt`."
        ) from exc
    return FoamCase


def ensure_openfoam_environment(
    commands: Sequence[str] = REQUIRED_OPENFOAM_COMMANDS,
) -> None:
    r"""Ensure OpenFOAM commands are available in the current environment."""
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing required OpenFOAM commands: {missing_str}. "
            "Activate OpenFOAM first (for this workspace, run `of2312`)."
        )


def compute_reynolds_number(nu: float, lid_velocity: float, length_scale: float) -> float:
    r"""Compute Reynolds number :math:`Re = U L / \nu`."""
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
    r"""Validate viscosity and derived Reynolds number against configured bounds."""
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
    r"""Extract lid velocity and length scale from an OpenFOAM cavity case."""
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


def build_sample_tensors(
    case: Any,
    nu: float,
    dtype: torch.dtype = torch.float32,
) -> CavitySampleTensors:
    r"""Build branch/trunk/target tensors from the latest case output time."""
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
    target_np = np.column_stack((velocity[:, 0], velocity[:, 1], velocity[:, 2], pressure))

    return CavitySampleTensors(
        branch_input=torch.tensor(branch_np, dtype=dtype),
        trunk_input=torch.tensor(trunk_np, dtype=dtype),
        target=torch.tensor(target_np, dtype=dtype),
        reynolds_number=math.nan,
    )


def build_deeponet(config: CavityDeepONetConfig) -> DeepONet:
    r"""Build DeepONet model for cavity output channels ``(u,v,w,p)``."""
    return DeepONet(
        branch_in_features=1,
        trunk_in_features=3,
        latent_dim=config.latent_dim,
        velocity_dim=3,
        branch_layers=config.branch_layers,
        trunk_layers=config.trunk_layers,
        layer_size=config.layer_size,
        activation_fn=config.activation_fn,
    )


def run_case_for_viscosity(
    nu: float,
    config: CavityDeepONetConfig,
    *,
    sample_index: int,
) -> CavitySampleTensors:
    r"""Run a cavity case at one viscosity and convert solver results to tensors."""
    limits = CavityPhysicalLimits(
        nu_min=config.nu_min,
        nu_max=config.nu_max,
        reynolds_min=config.reynolds_min,
        reynolds_max=config.reynolds_max,
    )

    FoamCase = _get_foam_case_cls()
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

    if config.run_openfoam:
        ensure_openfoam_environment()
        cloned_case.block_mesh()
        cloned_case.run("icoFoam")

    sample = build_sample_tensors(cloned_case, nu=nu, dtype=getattr(torch, config.dtype))
    sample.reynolds_number = reynolds
    return sample


def generate_dataset(config: CavityDeepONetConfig) -> tuple[Tensor, Tensor, Tensor]:
    r"""Generate concatenated branch/trunk/target tensors for all viscosities."""
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


def train_deeponet(
    model: DeepONet,
    branch_input: Tensor,
    trunk_input: Tensor,
    target: Tensor,
    *,
    lr: float,
    steps: int,
    weight_decay: float = 0.0,
    lbfgs_steps: int = 0,
    lbfgs_lr: float = 1.0,
) -> float:
    r"""Run a small supervised training loop and return final loss."""
    if steps <= 0 and lbfgs_steps <= 0:
        return float("nan")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    model.train()
    final_loss = float("nan")
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(branch_input, trunk_input)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    if lbfgs_steps > 0:
        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs_optimizer.zero_grad(set_to_none=True)
            pred = model(branch_input, trunk_input)
            loss = criterion(pred, target)
            loss.backward()
            return loss

        lbfgs_optimizer.step(closure)
        with torch.no_grad():
            final_loss = float(
                criterion(model(branch_input, trunk_input), target).detach().cpu().item()
            )
    return final_loss


def compute_relative_l2(prediction: Tensor, target: Tensor) -> tuple[float, tuple[float, ...]]:
    r"""Compute aggregate and channel-wise relative :math:`L^2` errors."""
    aggregate = torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target)
    channel_errors: list[float] = []
    for channel_idx in range(target.shape[-1]):
        denominator = torch.linalg.vector_norm(target[:, channel_idx])
        numerator = torch.linalg.vector_norm(prediction[:, channel_idx] - target[:, channel_idx])
        if denominator <= 1.0e-12:
            channel_errors.append(float(numerator.detach().cpu().item()))
        else:
            channel_errors.append(float((numerator / denominator).detach().cpu().item()))
    return float(aggregate.detach().cpu().item()), tuple(channel_errors)


def visualize_predictions(
    model: DeepONet,
    branch_input: Tensor,
    trunk_input: Tensor,
    target: Tensor,
    output_dir: Path,
    max_cases: int = 3,
    branch_normalizer: TensorNormalizer | None = None,
    trunk_normalizer: TensorNormalizer | None = None,
    target_normalizer: TensorNormalizer | None = None,
) -> None:
    r"""Save side-by-side true/pred/error plots for ``(u,v,w,p)`` fields."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install with "
            "`pip install matplotlib` or add it to your example environment."
        ) from exc

    if max_cases <= 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    branch_model_input = (
        branch_normalizer.transform(branch_input)
        if branch_normalizer is not None
        else branch_input
    )
    trunk_model_input = (
        trunk_normalizer.transform(trunk_input) if trunk_normalizer is not None else trunk_input
    )
    model.eval()
    with torch.no_grad():
        pred = model(branch_model_input, trunk_model_input)
        if target_normalizer is not None:
            pred = target_normalizer.inverse(pred)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    trunk_np = trunk_input.detach().cpu().numpy()
    branch_np = branch_input.detach().cpu().numpy().reshape(-1)

    # Group samples by viscosity value; each group corresponds to one OpenFOAM run.
    unique_nu = np.unique(branch_np)
    channel_names = model.output_channel_names
    for case_idx, nu in enumerate(unique_nu[:max_cases]):
        mask = np.isclose(branch_np, nu)
        x = trunk_np[mask, 0]
        y = trunk_np[mask, 1]
        tri = mtri.Triangulation(x, y)

        for channel_idx, channel_name in enumerate(channel_names):
            truth_field = target_np[mask, channel_idx]
            pred_field = pred_np[mask, channel_idx]
            diff_field = np.abs(pred_field - truth_field)

            fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            vmin = float(np.min(truth_field))
            vmax = float(np.max(truth_field))

            true_plot = ax[0].tricontourf(tri, truth_field, levels=50, vmin=vmin, vmax=vmax)
            pred_plot = ax[1].tricontourf(tri, pred_field, levels=50, vmin=vmin, vmax=vmax)
            diff_plot = ax[2].tricontourf(tri, diff_field, levels=50)
            fig.colorbar(true_plot, ax=ax[0])
            fig.colorbar(pred_plot, ax=ax[1])
            fig.colorbar(diff_plot, ax=ax[2])

            ax[0].set_title("True")
            ax[1].set_title("Pred")
            ax[2].set_title("Difference")
            for axis in ax:
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.set_aspect("equal", adjustable="box")

            fig.suptitle(f"nu={nu:.6g}, field={channel_name}")
            out_file = output_dir / f"case_{case_idx:03d}_nu_{nu:.6g}_{channel_name}.png"
            fig.savefig(out_file, dpi=200)
            plt.close(fig)


def _load_config(config_path: Path | None) -> CavityDeepONetConfig:
    default_cfg = OmegaConf.structured(CavityDeepONetConfig())
    if config_path is None:
        merged = default_cfg
    else:
        user_cfg = OmegaConf.load(config_path)
        merged = OmegaConf.merge(default_cfg, user_cfg)
    return CavityDeepONetConfig(**OmegaConf.to_container(merged, resolve=True))


def main() -> None:
    r"""CLI entrypoint for cavity DeepONet data generation and optional training."""
    parser = argparse.ArgumentParser(description="Cavity DeepONet with foamlib-driven OpenFOAM.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if cfg.train_steps <= 0 and cfg.save_visualizations:
        raise ValueError(
            "train_steps must be > 0 when save_visualizations=True. "
            "Otherwise visualized predictions are from an untrained model."
        )

    model = build_deeponet(cfg).to(cfg.device)
    branch_input, trunk_input, target = generate_dataset(cfg)
    branch_input = branch_input.to(cfg.device)
    trunk_input = trunk_input.to(cfg.device)
    target = target.to(cfg.device)

    branch_normalizer = (
        TensorNormalizer.fit(branch_input)
        if cfg.normalize_inputs
        else TensorNormalizer.identity(branch_input)
    )
    trunk_normalizer = (
        TensorNormalizer.fit(trunk_input)
        if cfg.normalize_inputs
        else TensorNormalizer.identity(trunk_input)
    )
    target_normalizer = (
        TensorNormalizer.fit(target)
        if cfg.normalize_targets
        else TensorNormalizer.identity(target)
    )
    model_branch_input = branch_normalizer.transform(branch_input)
    model_trunk_input = trunk_normalizer.transform(trunk_input)
    model_target = target_normalizer.transform(target)

    final_loss = train_deeponet(
        model,
        branch_input=model_branch_input,
        trunk_input=model_trunk_input,
        target=model_target,
        lr=cfg.learning_rate,
        steps=cfg.train_steps,
        weight_decay=cfg.weight_decay,
        lbfgs_steps=cfg.lbfgs_steps,
        lbfgs_lr=cfg.lbfgs_lr,
    )

    model.eval()
    with torch.no_grad():
        normalized_prediction = model(model_branch_input, model_trunk_input)
        prediction = target_normalizer.inverse(normalized_prediction)
    relative_l2, channel_relative_l2 = compute_relative_l2(prediction, target)

    print(f"Branch tensor shape: {tuple(branch_input.shape)}")
    print(f"Trunk tensor shape:  {tuple(trunk_input.shape)}")
    print(f"Target tensor shape: {tuple(target.shape)}")
    print(f"Output channel names: {model.output_channel_names}")
    if cfg.train_steps > 0 or cfg.lbfgs_steps > 0:
        print(
            "Final normalized training loss after "
            f"{cfg.train_steps} AdamW steps and {cfg.lbfgs_steps} LBFGS steps: {final_loss:.6e}"
        )
        print(f"Physical aggregate rel-L2: {relative_l2:.6e}")
        for channel_name, channel_error in zip(model.output_channel_names, channel_relative_l2):
            print(f"Physical rel-L2[{channel_name}]: {channel_error:.6e}")
    if cfg.save_visualizations:
        visualize_predictions(
            model=model,
            branch_input=branch_input,
            trunk_input=trunk_input,
            target=target,
            output_dir=Path(cfg.visualization_dir),
            max_cases=cfg.visualization_max_cases,
            branch_normalizer=branch_normalizer,
            trunk_normalizer=trunk_normalizer,
            target_normalizer=target_normalizer,
        )
        print(f"Saved visualization figures to: {cfg.visualization_dir}")


if __name__ == "__main__":
    main()
