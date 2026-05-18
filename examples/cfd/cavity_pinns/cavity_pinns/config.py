# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class CavityCaseMetaData:
    lid_velocity: float
    length_scale: float


@dataclass
class CavityPINNConfig:
    case_path: str = str(
        Path(__file__).resolve().parents[2] / "cavity_deeponet" / "cavity"
    )
    run_root: str = str(Path(__file__).resolve().parent.parent / ".foamlib_runs_pinn")
    run_openfoam: bool = True
    clean_run_root: bool = False

    spatial_dim: int = 2
    coordinate_names: tuple[str, ...] = ("x", "y")
    velocity_names: tuple[str, ...] = ("u", "v")
    pressure_name: str = "p"

    device: str = "cpu"
    dtype: str = "float32"

    reynolds_min: float = 10.0
    reynolds_max: float = 100.0
    viscosity_seed: int = 7
    train_case_count: int = 40
    validate_case_count: int = 5
    test_case_count: int = 5
    unseen_case_count: int = 0
    manifest_filename: str = "viscosity_split_manifest.json"

    model_layers: int = 4
    model_layer_size: int = 128
    activation_fn: str = "tanh"

    train_steps: int = 5000
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    supervised_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    boundary_loss_ramp_steps: int = 500
    physics_loss_weight: float = 1.0e-1
    physics_loss_ramp_steps: int = 1000
    boundary_points_per_wall: int = 64
    log_every: int = 200

    save_visualizations: bool = False
    visualization_max_cases: int = 3
    visualization_dir: str = str(
        Path(__file__).resolve().parent.parent / "outputs" / "figures"
    )
    model_checkpoint_path: str = str(
        Path(__file__).resolve().parent.parent / "outputs" / "checkpoints" / "cavity_pinn.pt"
    )
    use_saved_model_for_unseen: bool = False


def load_config(config_path: Path | None) -> CavityPINNConfig:
    default_cfg = OmegaConf.structured(CavityPINNConfig())
    if config_path is None:
        merged = default_cfg
    else:
        user_cfg = OmegaConf.load(config_path)
        merged = OmegaConf.merge(default_cfg, user_cfg)
    return CavityPINNConfig(**OmegaConf.to_container(merged, resolve=True))
