# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


def _default_pod_modes() -> dict[str, int]:
    return {"u": 4, "v": 4, "w": 2, "p": 4}


def _default_autoencoder_latent_dims() -> dict[str, int]:
    return {"u": 8, "v": 8, "w": 4, "p": 8}


@dataclass
class CavityLatentDeepONetConfig:
    case_path: str = str(
        Path(__file__).resolve().parent.parent.parent / "cavity_deeponet" / "cavity"
    )
    run_root: str = str(Path(__file__).resolve().parent.parent / ".foamlib_runs")
    run_openfoam: bool = True
    clean_run_root: bool = False
    nu_min: float = 1.0e-3
    nu_max: float = 1.0e-2
    reynolds_min: float = 10.0
    reynolds_max: float = 100.0
    viscosity_seed: int = 7
    train_case_count: int = 40
    validate_case_count: int = 5
    test_case_count: int = 5
    manifest_filename: str = "viscosity_split_manifest.json"
    reconstruction_mode: str = "pod"
    field_order: tuple[str, ...] = ("u", "v", "w", "p")
    pod_modes: dict[str, int] = field(default_factory=_default_pod_modes)
    autoencoder_latent_dims: dict[str, int] = field(
        default_factory=_default_autoencoder_latent_dims
    )
    branch_layers: int = 4
    trunk_layers: int = 4
    decoder_layers: int = 2
    layer_size: int = 128
    decoder_layer_size: int = 128
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
        Path(__file__).resolve().parent.parent / "outputs" / "latent_deeponet_model.pt"
    )
    save_trained_model: bool = True


def _validate_field_mapping(
    mapping_name: str, mapping: dict[str, int], field_order: tuple[str, ...]
) -> None:
    expected = set(field_order)
    actual = set(mapping.keys())
    if actual != expected:
        raise ValueError(
            f"{mapping_name} must define exactly {field_order}, but got {tuple(sorted(actual))}."
        )
    for field_name, value in mapping.items():
        if value < 1:
            raise ValueError(
                f"{mapping_name}['{field_name}'] must be >= 1, but got {value}."
            )


def normalized_reconstruction_mode(reconstruction_mode: str) -> str:
    mode = reconstruction_mode.lower()
    if mode == "autoencoder":
        return "decoder"
    if mode in ("pod", "decoder"):
        return mode
    raise ValueError(
        "reconstruction_mode must be one of {'pod', 'autoencoder', 'decoder'}."
    )


def validate_config(config: CavityLatentDeepONetConfig) -> None:
    if len(config.field_order) < 1:
        raise ValueError("field_order must define at least one field.")
    mode = normalized_reconstruction_mode(config.reconstruction_mode)
    if mode == "pod":
        _validate_field_mapping("pod_modes", config.pod_modes, config.field_order)
    else:
        _validate_field_mapping(
            "autoencoder_latent_dims",
            config.autoencoder_latent_dims,
            config.field_order,
        )


def load_config(config_path: Path | None) -> CavityLatentDeepONetConfig:
    default_cfg = OmegaConf.structured(CavityLatentDeepONetConfig())
    if config_path is None:
        merged = default_cfg
    else:
        user_cfg = OmegaConf.load(config_path)
        merged = OmegaConf.merge(default_cfg, user_cfg)
    config = CavityLatentDeepONetConfig(**OmegaConf.to_container(merged, resolve=True))
    validate_config(config)
    return config
