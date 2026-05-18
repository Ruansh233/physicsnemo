# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class CavityPhysicalLimits:
    nu_min: float = 1.0e-3
    nu_max: float = 1.0e-2
    reynolds_min: float = 10.0
    reynolds_max: float = 100.0


@dataclass(frozen=True)
class CavityCaseMetaData:
    lid_velocity: float
    length_scale: float


@dataclass
class CavityDeepONetConfig:
    case_path: str = str(Path(__file__).resolve().parent.parent / "cavity")
    run_root: str = str(Path(__file__).resolve().parent.parent / ".foamlib_runs")
    run_openfoam: bool = True
    clean_run_root: bool = False
    viscosity_values: tuple[float, ...] = (1.0e-3, 5.0e-3, 1.0e-2)
    nu_min: float = 1.0e-3
    nu_max: float = 1.0e-2
    reynolds_min: float = 10.0
    reynolds_max: float = 100.0
    viscosity_seed: int = 7
    train_case_count: int = 40
    validate_case_count: int = 5
    test_case_count: int = 5
    manifest_filename: str = "viscosity_split_manifest.json"
    latent_dim: int = 128
    branch_layers: int = 4
    trunk_layers: int = 4
    layer_size: int = 128
    activation_fn: str = "gelu"
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-6
    train_steps: int = 1000
    log_steps: int = 50
    enable_realtime_loss_plot: bool = False
    lbfgs_steps: int = 200
    lbfgs_lr: float = 0.3
    normalize_inputs: bool = True
    normalize_targets: bool = True
    device: str = "cpu"
    dtype: str = "float32"
    save_visualizations: bool = True
    visualization_dir: str = str(
        Path(__file__).resolve().parent.parent / "outputs" / "figures"
    )
    visualization_max_cases: int = 3
    model_checkpoint_path: str = str(
        Path(__file__).resolve().parent.parent / "outputs" / "deeponet_model.pt"
    )
    save_trained_model: bool = True
    use_saved_model_for_unseen: bool = False


def load_config(config_path: Path | None) -> CavityDeepONetConfig:
    default_cfg = OmegaConf.structured(CavityDeepONetConfig())
    if config_path is None:
        merged = default_cfg
    else:
        user_cfg = OmegaConf.load(config_path)
        merged = OmegaConf.merge(default_cfg, user_cfg)
    return CavityDeepONetConfig(**OmegaConf.to_container(merged, resolve=True))
