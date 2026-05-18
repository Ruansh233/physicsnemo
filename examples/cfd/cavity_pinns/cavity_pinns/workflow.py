# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from .config import CavityPINNConfig
from .modeling import build_pinn_model
from .openfoam_data import (
    CavityCaseTensors,
    extract_case_metadata,
    generate_cases_for_viscosities,
    get_foam_case_cls,
)
from .physics import compute_physics_residuals, make_physics_informer
from .splits import build_split_manifest, derive_viscosity_bounds, save_split_manifest
from .visualization import save_split_triptych_visualizations


@dataclass
class BoundaryConditionTensors:
    coordinates: Tensor
    viscosity: Tensor
    target_velocity: Tensor


@dataclass
class NormalizationStats:
    """Affine normalization stats for PINN inputs and field outputs."""

    input_mean: Tensor
    input_scale: Tensor
    output_mean: Tensor
    output_scale: Tensor

    def to(self, device: torch.device | str) -> "NormalizationStats":
        return NormalizationStats(
            input_mean=self.input_mean.to(device),
            input_scale=self.input_scale.to(device),
            output_mean=self.output_mean.to(device),
            output_scale=self.output_scale.to(device),
        )


def _stack_cases(cases: list[CavityCaseTensors]) -> tuple[Tensor, Tensor, Tensor]:
    if not cases:
        raise ValueError("Expected at least one case for stacking.")
    coordinates = torch.cat([case.coordinates for case in cases], dim=0)
    viscosity = torch.cat([case.viscosity for case in cases], dim=0)
    target = torch.cat([case.target for case in cases], dim=0)
    return coordinates, viscosity, target


def _stable_scale(values: Tensor) -> Tensor:
    scale = values.std(dim=0, keepdim=True, unbiased=False)
    return torch.where(scale > 1.0e-12, scale, torch.ones_like(scale))


def _build_normalization_stats(
    coordinates: Tensor,
    viscosity: Tensor,
    target: Tensor,
) -> NormalizationStats:
    model_input = torch.cat((coordinates, viscosity), dim=1)
    return NormalizationStats(
        input_mean=model_input.mean(dim=0, keepdim=True),
        input_scale=_stable_scale(model_input),
        output_mean=target.mean(dim=0, keepdim=True),
        output_scale=_stable_scale(target),
    )


def _normalize_inputs(
    *,
    normalizer: NormalizationStats,
    coordinates: Tensor,
    viscosity: Tensor,
) -> Tensor:
    model_input = torch.cat((coordinates, viscosity), dim=1)
    return (model_input - normalizer.input_mean) / normalizer.input_scale


def _normalize_outputs(*, normalizer: NormalizationStats, fields: Tensor) -> Tensor:
    return (fields - normalizer.output_mean) / normalizer.output_scale


def _denormalize_outputs(*, normalizer: NormalizationStats, fields: Tensor) -> Tensor:
    return fields * normalizer.output_scale + normalizer.output_mean


def _predict_normalized_fields(
    *,
    model: torch.nn.Module,
    normalizer: NormalizationStats,
    coordinates: Tensor,
    viscosity: Tensor,
) -> Tensor:
    return model(
        _normalize_inputs(
            normalizer=normalizer, coordinates=coordinates, viscosity=viscosity
        )
    )


def _predict_physical_fields(
    *,
    model: torch.nn.Module,
    normalizer: NormalizationStats,
    coordinates: Tensor,
    viscosity: Tensor,
) -> Tensor:
    prediction = _predict_normalized_fields(
        model=model,
        normalizer=normalizer,
        coordinates=coordinates,
        viscosity=viscosity,
    )
    return _denormalize_outputs(normalizer=normalizer, fields=prediction)


def _compute_boundary_loss(
    *,
    model: torch.nn.Module,
    normalizer: NormalizationStats,
    boundary: BoundaryConditionTensors,
) -> Tensor:
    if boundary.coordinates.numel() == 0:
        return torch.zeros(
            (),
            dtype=boundary.target_velocity.dtype,
            device=boundary.target_velocity.device,
        )
    prediction = _predict_physical_fields(
        model=model,
        normalizer=normalizer,
        coordinates=boundary.coordinates,
        viscosity=boundary.viscosity,
    )
    return torch.mean((prediction[:, :2] - boundary.target_velocity) ** 2)


def _build_boundary_condition_tensors(
    cases: list[CavityCaseTensors],
    *,
    points_per_wall: int,
) -> BoundaryConditionTensors:
    if points_per_wall < 0:
        raise ValueError("Expected boundary_points_per_wall >= 0.")
    if not cases or points_per_wall == 0:
        return BoundaryConditionTensors(
            coordinates=torch.empty((0, 2)),
            viscosity=torch.empty((0, 1)),
            target_velocity=torch.empty((0, 2)),
        )

    coordinate_batches: list[Tensor] = []
    viscosity_batches: list[Tensor] = []
    target_batches: list[Tensor] = []
    for case in cases:
        coords = case.coordinates
        dtype = coords.dtype
        device = coords.device
        x_min = torch.zeros((), dtype=dtype, device=device)
        y_min = torch.zeros((), dtype=dtype, device=device)
        x_max = torch.as_tensor(case.metadata.length_scale, dtype=dtype, device=device)
        y_max = torch.as_tensor(case.metadata.length_scale, dtype=dtype, device=device)
        xs = torch.linspace(x_min, x_max, points_per_wall, dtype=dtype, device=device)
        ys = torch.linspace(y_min, y_max, points_per_wall, dtype=dtype, device=device)

        top = torch.stack((xs, torch.full_like(xs, y_max)), dim=1)
        bottom = torch.stack((xs, torch.full_like(xs, y_min)), dim=1)
        left = torch.stack((torch.full_like(ys, x_min), ys), dim=1)
        right = torch.stack((torch.full_like(ys, x_max), ys), dim=1)
        boundary_coords = torch.cat((top, bottom, left, right), dim=0)

        top_target = torch.zeros((points_per_wall, 2), dtype=dtype, device=device)
        top_target[:, 0] = case.metadata.lid_velocity
        no_slip_target = torch.zeros(
            (3 * points_per_wall, 2), dtype=dtype, device=device
        )
        target_velocity = torch.cat((top_target, no_slip_target), dim=0)

        coordinate_batches.append(boundary_coords)
        viscosity_batches.append(
            case.viscosity[:1].expand(boundary_coords.shape[0], -1)
        )
        target_batches.append(target_velocity)

    return BoundaryConditionTensors(
        coordinates=torch.cat(coordinate_batches, dim=0),
        viscosity=torch.cat(viscosity_batches, dim=0),
        target_velocity=torch.cat(target_batches, dim=0),
    )


def _prepare_split_config(
    config: CavityPINNConfig, split_name: str
) -> CavityPINNConfig:
    split_run_root = Path(config.run_root) / split_name
    return replace(config, run_root=str(split_run_root))


def _prepare_split_manifest(config: CavityPINNConfig) -> dict[str, object]:
    foam_case_cls = get_foam_case_cls()
    source_case = foam_case_cls(Path(config.case_path))
    metadata = extract_case_metadata(source_case)
    nu_min, nu_max = derive_viscosity_bounds(
        metadata=metadata,
        reynolds_min=config.reynolds_min,
        reynolds_max=config.reynolds_max,
    )
    manifest = build_split_manifest(
        seed=config.viscosity_seed,
        train_count=config.train_case_count,
        val_count=config.validate_case_count,
        test_count=config.test_case_count,
        unseen_count=config.unseen_case_count,
        nu_min=nu_min,
        nu_max=nu_max,
    )
    manifest_path = Path(config.run_root) / config.manifest_filename
    save_split_manifest(manifest_path, manifest)
    print(f"Saved viscosity split manifest to: {manifest_path}")
    return manifest


def _evaluate_split(
    *,
    model: torch.nn.Module,
    physics_informer,
    normalizer: NormalizationStats,
    cases: list[CavityCaseTensors],
    config: CavityPINNConfig,
    split_name: str,
) -> dict[str, float]:
    coordinates, viscosity, target = _stack_cases(cases)
    coordinates = coordinates.to(config.device).clone().detach().requires_grad_(True)
    viscosity = viscosity.to(config.device)
    target = target.to(config.device)
    prediction = _predict_physical_fields(
        model=model,
        normalizer=normalizer,
        coordinates=coordinates,
        viscosity=viscosity,
    )
    boundary = _build_boundary_condition_tensors(
        cases,
        points_per_wall=config.boundary_points_per_wall,
    )
    boundary = BoundaryConditionTensors(
        coordinates=boundary.coordinates.to(config.device),
        viscosity=boundary.viscosity.to(config.device),
        target_velocity=boundary.target_velocity.to(config.device),
    )

    supervised_mse = torch.mean((prediction - target) ** 2)
    rel_l2 = torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(
        target
    )
    boundary_loss = _compute_boundary_loss(
        model=model, normalizer=normalizer, boundary=boundary
    )
    residuals = compute_physics_residuals(
        physics_informer=physics_informer,
        coordinates=coordinates,
        prediction=prediction,
        viscosity=viscosity,
        spatial_dim=config.spatial_dim,
    )
    residual_mse = torch.mean(
        torch.cat([value**2 for value in residuals.values()], dim=1)
    )

    metrics = {
        "supervised_mse": float(supervised_mse.detach().cpu().item()),
        "relative_l2": float(rel_l2.detach().cpu().item()),
        "boundary_loss": float(boundary_loss.detach().cpu().item()),
        "physics_residual_mse": float(residual_mse.detach().cpu().item()),
        "num_samples": float(target.shape[0]),
    }
    for idx, name in enumerate((*config.velocity_names, config.pressure_name)):
        channel_error = torch.linalg.vector_norm(prediction[:, idx] - target[:, idx])
        channel_norm = torch.linalg.vector_norm(target[:, idx]).clamp_min(1.0e-12)
        metrics[f"{name}_relative_l2"] = float(
            (channel_error / channel_norm).detach().cpu().item()
        )
    print(
        f"{split_name}: mse={metrics['supervised_mse']:.6e}, "
        f"rel_l2={metrics['relative_l2']:.6e}, "
        f"u={metrics['u_relative_l2']:.6e}, "
        f"v={metrics['v_relative_l2']:.6e}, "
        f"p={metrics['p_relative_l2']:.6e}, "
        f"bc={metrics['boundary_loss']:.6e}, "
        f"physics={metrics['physics_residual_mse']:.6e}"
    )
    save_split_triptych_visualizations(
        model=model,
        cases=cases,
        config=config,
        split_name=split_name,
        predict_fn=lambda model, coords, visc: _predict_physical_fields(
            model=model,
            normalizer=normalizer,
            coordinates=coords,
            viscosity=visc,
        ),
    )
    return metrics


def _ramped_weight(*, base_weight: float, step: int, ramp_steps: int) -> float:
    if base_weight <= 0.0:
        return 0.0
    if ramp_steps <= 0:
        return base_weight
    return base_weight * min(1.0, step / ramp_steps)


def _train_model(
    *,
    config: CavityPINNConfig,
    train_cases: list[CavityCaseTensors],
) -> tuple[torch.nn.Module, object, NormalizationStats]:
    model = build_pinn_model(config).to(config.device)
    physics_informer = make_physics_informer(
        spatial_dim=config.spatial_dim, device=config.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_coordinates, train_viscosity, train_target = _stack_cases(train_cases)
    train_viscosity = train_viscosity.to(config.device)
    train_target = train_target.to(config.device)
    normalizer = _build_normalization_stats(
        train_coordinates,
        train_viscosity.cpu(),
        train_target.cpu(),
    ).to(config.device)
    normalized_train_target = _normalize_outputs(
        normalizer=normalizer, fields=train_target
    )
    boundary = _build_boundary_condition_tensors(
        train_cases,
        points_per_wall=config.boundary_points_per_wall,
    )
    boundary = BoundaryConditionTensors(
        coordinates=boundary.coordinates.to(config.device),
        viscosity=boundary.viscosity.to(config.device),
        target_velocity=boundary.target_velocity.to(config.device),
    )

    for step in range(1, config.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        coordinates = (
            train_coordinates.to(config.device).clone().detach().requires_grad_(True)
        )
        normalized_prediction = _predict_normalized_fields(
            model=model,
            normalizer=normalizer,
            coordinates=coordinates,
            viscosity=train_viscosity,
        )
        prediction = _denormalize_outputs(
            normalizer=normalizer, fields=normalized_prediction
        )

        supervised_loss = torch.mean(
            (normalized_prediction - normalized_train_target) ** 2
        )
        boundary_loss = _compute_boundary_loss(
            model=model, normalizer=normalizer, boundary=boundary
        )
        residuals = compute_physics_residuals(
            physics_informer=physics_informer,
            coordinates=coordinates,
            prediction=prediction,
            viscosity=train_viscosity,
            spatial_dim=config.spatial_dim,
        )
        physics_loss = torch.mean(
            torch.cat([value**2 for value in residuals.values()], dim=1)
        )
        boundary_weight = _ramped_weight(
            base_weight=config.boundary_loss_weight,
            step=step,
            ramp_steps=config.boundary_loss_ramp_steps,
        )
        physics_weight = _ramped_weight(
            base_weight=config.physics_loss_weight,
            step=step,
            ramp_steps=config.physics_loss_ramp_steps,
        )

        loss = (
            config.supervised_loss_weight * supervised_loss
            + boundary_weight * boundary_loss
            + physics_weight * physics_loss
        )
        loss.backward()
        optimizer.step()

        if (
            step % max(config.log_every, 1) == 0
            or step == 1
            or step == config.train_steps
        ):
            print(
                f"step={step:05d} total={loss.detach().cpu().item():.6e} "
                f"supervised={supervised_loss.detach().cpu().item():.6e} "
                f"bc={boundary_loss.detach().cpu().item():.6e} "
                f"physics={physics_loss.detach().cpu().item():.6e} "
                f"bc_w={boundary_weight:.3e} physics_w={physics_weight:.3e}"
            )
    return model, physics_informer, normalizer


def _load_split_cases(
    config: CavityPINNConfig,
    split_values: list[float],
    split_name: str,
) -> list[CavityCaseTensors]:
    split_cfg = _prepare_split_config(config, split_name)
    return generate_cases_for_viscosities(
        config=split_cfg, viscosity_values=split_values
    )


def run_training_workflow(config: CavityPINNConfig) -> None:
    if config.spatial_dim != 2:
        raise NotImplementedError(
            "Current cavity PINN workflow supports spatial_dim=2 only."
        )
    if config.clean_run_root:
        run_root = Path(config.run_root)
        if run_root.exists():
            shutil.rmtree(run_root)

    manifest = _prepare_split_manifest(config)
    train_cases = _load_split_cases(config, manifest["train"], "train")
    validate_cases = _load_split_cases(config, manifest["validate"], "validate")
    test_cases = _load_split_cases(config, manifest["test"], "test")

    model, physics_informer, normalizer = _train_model(
        config=config, train_cases=train_cases
    )
    _evaluate_split(
        model=model,
        physics_informer=physics_informer,
        normalizer=normalizer,
        cases=train_cases,
        config=config,
        split_name="train",
    )
    _evaluate_split(
        model=model,
        physics_informer=physics_informer,
        normalizer=normalizer,
        cases=validate_cases,
        config=config,
        split_name="validate",
    )
    _evaluate_split(
        model=model,
        physics_informer=physics_informer,
        normalizer=normalizer,
        cases=test_cases,
        config=config,
        split_name="test",
    )


def run_unseen_viscosity_workflow(config: CavityPINNConfig) -> None:
    if config.spatial_dim != 2:
        raise NotImplementedError(
            "Current cavity PINN workflow supports spatial_dim=2 only."
        )
    if config.clean_run_root:
        run_root = Path(config.run_root)
        if run_root.exists():
            shutil.rmtree(run_root)

    manifest = _prepare_split_manifest(config)
    train_cases = _load_split_cases(config, manifest["train"], "train")
    unseen_cases = _load_split_cases(config, manifest["unseen"], "unseen")

    model, physics_informer, normalizer = _train_model(
        config=config, train_cases=train_cases
    )
    _evaluate_split(
        model=model,
        physics_informer=physics_informer,
        normalizer=normalizer,
        cases=train_cases,
        config=config,
        split_name="train",
    )
    _evaluate_split(
        model=model,
        physics_informer=physics_informer,
        normalizer=normalizer,
        cases=unseen_cases,
        config=config,
        split_name="unseen",
    )
