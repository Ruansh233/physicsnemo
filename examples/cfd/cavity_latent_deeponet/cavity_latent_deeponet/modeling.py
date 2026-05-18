# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from examples.cfd.cavity_deeponet.cavity_deeponet.modeling import TensorNormalizer

from .config import CavityLatentDeepONetConfig, normalized_reconstruction_mode


def _build_realtime_loss_plotter() -> tuple[Callable[[int, float], None], Callable[[], None]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for real-time loss plotting. "
            "Install with `pip install matplotlib` or add it to your example environment."
        ) from exc

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    steps: list[int] = []
    losses: list[float] = []
    (line,) = ax.plot([], [], label="training_loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("LatentDeepONet Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    def _update(step: int, loss: float) -> None:
        steps.append(step)
        losses.append(loss)
        line.set_data(steps, losses)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

    def _close() -> None:
        plt.ioff()
        plt.close(fig)

    return _update, _close


def _get_activation(activation_fn: str) -> nn.Module:
    activation_name = activation_fn.lower()
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "silu":
        return nn.SiLU()
    if activation_name == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"Unsupported activation_fn '{activation_fn}'. Supported values are gelu, relu, silu, tanh."
    )


def _build_mlp(
    in_features: int,
    out_features: int,
    num_layers: int,
    hidden_size: int,
    activation_fn: str,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError(f"Expected num_layers >= 1, but got {num_layers}")

    layers: list[nn.Module] = []
    input_dim = in_features
    for _ in range(num_layers):
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(_get_activation(activation_fn))
        input_dim = hidden_size
    layers.append(nn.Linear(input_dim, out_features))
    return nn.Sequential(*layers)


class FieldwiseLatentDeepONet(nn.Module):
    r"""Shared-backbone latent DeepONet with field-specific reconstruction heads."""

    def __init__(
        self,
        field_latent_dims: Mapping[str, int],
        output_channel_names: tuple[str, ...],
        reconstruction_mode: str,
        branch_layers: int,
        trunk_layers: int,
        decoder_layers: int,
        layer_size: int,
        decoder_layer_size: int,
        activation_fn: str,
        pod_bases: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.output_channel_names = output_channel_names
        self.reconstruction_mode = reconstruction_mode
        self.field_latent_dims = dict(field_latent_dims)
        self.field_models = nn.ModuleDict()
        self.branch_encoder = _build_mlp(1, layer_size, branch_layers, layer_size, activation_fn)
        self.trunk_encoder = _build_mlp(3, layer_size, trunk_layers, layer_size, activation_fn)
        self.num_points: int | None = None

        self.branch_heads = nn.ModuleDict()
        self.trunk_heads = nn.ModuleDict()
        self.decoder_heads = nn.ModuleDict()
        self.latent_biases = nn.ParameterDict()
        self.output_biases = nn.ParameterDict()

        if self.reconstruction_mode == "pod":
            if pod_bases is None:
                raise ValueError("pod_bases is required when reconstruction_mode='pod'.")
            self._init_pod_heads(pod_bases)
        elif self.reconstruction_mode == "decoder":
            self._init_decoder_heads(
                decoder_layers=decoder_layers,
                decoder_layer_size=decoder_layer_size,
                activation_fn=activation_fn,
            )
        else:
            raise ValueError(
                "reconstruction_mode must be one of {'pod', 'decoder'}, "
                f"but got {self.reconstruction_mode}."
            )

    def _init_pod_heads(self, pod_bases: Mapping[str, Tensor]) -> None:
        point_count: int | None = None
        for field_name in self.output_channel_names:
            latent_dim = self.field_latent_dims[field_name]
            if field_name not in pod_bases:
                raise ValueError(f"Missing POD basis for field '{field_name}'.")
            basis = torch.as_tensor(pod_bases[field_name], dtype=torch.float32)
            if basis.ndim != 2 or basis.shape[1] != latent_dim:
                raise ValueError(
                    f"Expected POD basis for field '{field_name}' with shape "
                    f"(num_points, {latent_dim}), but got {tuple(basis.shape)}."
                )
            if point_count is None:
                point_count = int(basis.shape[0])
            elif basis.shape[0] != point_count:
                raise ValueError("All field POD bases must use the same num_points.")
            self.register_buffer(f"pod_basis_{field_name}", basis)
            self.branch_heads[field_name] = nn.Linear(
                self.branch_encoder[-1].out_features, latent_dim
            )
            self.latent_biases[field_name] = nn.Parameter(torch.zeros(latent_dim))
            self.output_biases[field_name] = nn.Parameter(torch.zeros(1))

        self.num_points = point_count

    def _init_decoder_heads(
        self,
        *,
        decoder_layers: int,
        decoder_layer_size: int,
        activation_fn: str,
    ) -> None:
        for field_name in self.output_channel_names:
            latent_dim = self.field_latent_dims[field_name]
            self.branch_heads[field_name] = nn.Linear(
                self.branch_encoder[-1].out_features, latent_dim
            )
            self.trunk_heads[field_name] = nn.Linear(
                self.trunk_encoder[-1].out_features, latent_dim
            )
            self.latent_biases[field_name] = nn.Parameter(torch.zeros(latent_dim))
            self.decoder_heads[field_name] = _build_mlp(
                latent_dim, 1, decoder_layers, decoder_layer_size, activation_fn
            )

    def forward(self, branch_input: Tensor, trunk_input: Tensor) -> Tensor:
        if self.reconstruction_mode == "pod":
            return self._forward_pod(branch_input, trunk_input)
        return self._forward_decoder(branch_input, trunk_input)

    @staticmethod
    def _validate_branch_input(branch_input: Tensor) -> None:
        if branch_input.ndim != 2 or branch_input.shape[-1] != 1:
            raise ValueError(
                f"Expected branch_input with shape (B, 1), got {tuple(branch_input.shape)}."
            )

    @staticmethod
    def _validate_trunk_input(trunk_input: Tensor) -> None:
        if trunk_input.ndim not in (2, 3) or trunk_input.shape[-1] != 3:
            raise ValueError(
                "Expected trunk_input with shape (B, 3) or (B, Q, 3), "
                f"got {tuple(trunk_input.shape)}."
            )

    def _forward_pod(self, branch_input: Tensor, trunk_input: Tensor) -> Tensor:
        self._validate_branch_input(branch_input)
        self._validate_trunk_input(trunk_input)
        if trunk_input.ndim == 3:
            case_count, point_count = trunk_input.shape[:2]
            if branch_input.shape[0] != case_count:
                raise ValueError(
                    "For rank-3 trunk_input, branch_input batch size must match "
                    f"case count {case_count}, but got {branch_input.shape[0]}."
                )
            if point_count != self.num_points:
                raise ValueError(
                    f"Expected trunk_input query count {self.num_points}, got {point_count}."
                )
            branch_cases = branch_input
            output_rank3 = True
        elif trunk_input.ndim == 2:
            row_count = trunk_input.shape[0]
            if branch_input.shape[0] != row_count:
                raise ValueError(
                    "For rank-2 trunk_input, branch_input and trunk_input must have "
                    f"matching rows, got {branch_input.shape[0]} and {row_count}."
                )
            if self.num_points is None or row_count % self.num_points != 0:
                raise ValueError(
                    f"Rank-2 POD inputs must contain a multiple of num_points={self.num_points} rows."
                )
            case_count = row_count // self.num_points
            branch_cases = branch_input.reshape(case_count, self.num_points, -1)[:, 0, :]
            output_rank3 = False

        branch_features = self.branch_encoder(branch_cases)
        field_outputs: list[Tensor] = []
        for field_name in self.output_channel_names:
            coeff = self.branch_heads[field_name](branch_features)
            coeff = coeff + self.latent_biases[field_name]
            basis = getattr(self, f"pod_basis_{field_name}")
            field = torch.einsum("cr,nr->nc", basis.to(dtype=coeff.dtype), coeff)
            field = field + self.output_biases[field_name].to(dtype=field.dtype)
            field_outputs.append(field.unsqueeze(-1))
        output = torch.cat(field_outputs, dim=-1)
        if output_rank3:
            return output
        return output.reshape(-1, len(self.output_channel_names))

    def _forward_decoder(self, branch_input: Tensor, trunk_input: Tensor) -> Tensor:
        self._validate_branch_input(branch_input)
        self._validate_trunk_input(trunk_input)
        if trunk_input.ndim == 3:
            batch_size, query_count = trunk_input.shape[:2]
            if branch_input.shape[0] != batch_size:
                raise ValueError(
                    "For rank-3 trunk_input, branch_input batch size must match "
                    f"case count {batch_size}, but got {branch_input.shape[0]}."
                )
            branch_model_input = (
                branch_input[:, None, :].expand(-1, query_count, -1).reshape(-1, 1)
            )
            trunk_model_input = trunk_input.reshape(-1, trunk_input.shape[-1])
            output_shape = (batch_size, query_count, len(self.output_channel_names))
        elif trunk_input.ndim == 2:
            if branch_input.shape[0] != trunk_input.shape[0]:
                raise ValueError(
                    "For rank-2 trunk_input, branch_input and trunk_input must have "
                    f"matching rows, got {branch_input.shape[0]} and {trunk_input.shape[0]}."
                )
            branch_model_input = branch_input
            trunk_model_input = trunk_input
            output_shape = (branch_input.shape[0], len(self.output_channel_names))

        branch_features = self.branch_encoder(branch_model_input)
        trunk_features = self.trunk_encoder(trunk_model_input)
        field_outputs: list[Tensor] = []
        for field_name in self.output_channel_names:
            latent = (
                self.branch_heads[field_name](branch_features)
                * self.trunk_heads[field_name](trunk_features)
                + self.latent_biases[field_name]
            )
            field_outputs.append(self.decoder_heads[field_name](latent))
        output = torch.cat(field_outputs, dim=-1)
        return output.reshape(output_shape)


def compute_field_pod_bases(
    target: Tensor,
    *,
    case_count: int,
    field_order: tuple[str, ...],
    pod_modes: Mapping[str, int],
) -> dict[str, Tensor]:
    if target.ndim != 2:
        raise ValueError(f"Expected target with shape (N, C), got {tuple(target.shape)}.")
    if case_count < 1:
        raise ValueError(f"Expected case_count >= 1, got {case_count}.")
    if target.shape[0] % case_count != 0:
        raise ValueError(
            f"Target row count {target.shape[0]} must be divisible by case_count={case_count}."
        )
    if target.shape[1] != len(field_order):
        raise ValueError(
            f"Expected target channel count {len(field_order)}, got {target.shape[1]}."
        )

    num_points = target.shape[0] // case_count
    case_fields = target.reshape(case_count, num_points, len(field_order))
    bases: dict[str, Tensor] = {}
    max_rank = min(case_count, num_points)
    for channel_idx, field_name in enumerate(field_order):
        if field_name not in pod_modes:
            raise ValueError(f"Missing pod_modes entry for field '{field_name}'.")
        rank = int(pod_modes[field_name])
        if rank < 1:
            raise ValueError(
                f"pod_modes['{field_name}'] must be >= 1, but got {rank}."
            )
        if rank > max_rank:
            raise ValueError(
                f"pod_modes['{field_name}']={rank} exceeds the maximum rank "
                f"{max_rank} for {case_count} cases and {num_points} points."
            )
        snapshots = case_fields[:, :, channel_idx].transpose(0, 1)
        basis, _, _ = torch.linalg.svd(snapshots, full_matrices=False)
        bases[field_name] = basis[:, :rank].contiguous()
    return bases


def build_latent_deeponet(
    config: CavityLatentDeepONetConfig,
    pod_bases: Mapping[str, Tensor] | None = None,
) -> FieldwiseLatentDeepONet:
    mode = normalized_reconstruction_mode(config.reconstruction_mode)
    latent_dims = (
        config.pod_modes if mode == "pod" else config.autoencoder_latent_dims
    )
    return FieldwiseLatentDeepONet(
        field_latent_dims={field_name: int(latent_dims[field_name]) for field_name in config.field_order},
        output_channel_names=config.field_order,
        reconstruction_mode=mode,
        branch_layers=config.branch_layers,
        trunk_layers=config.trunk_layers,
        decoder_layers=config.decoder_layers,
        layer_size=config.layer_size,
        decoder_layer_size=config.decoder_layer_size,
        activation_fn=config.activation_fn,
        pod_bases=pod_bases,
    )


def train_latent_deeponet(
    model: FieldwiseLatentDeepONet,
    branch_input: Tensor,
    trunk_input: Tensor,
    target: Tensor,
    *,
    lr: float,
    steps: int,
    weight_decay: float = 0.0,
    log_steps: int = 0,
    enable_realtime_loss_plot: bool = False,
    lbfgs_steps: int = 0,
    lbfgs_lr: float = 1.0,
) -> float:
    if steps <= 0 and lbfgs_steps <= 0:
        return float("nan")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    plot_update: Callable[[int, float], None] | None = None
    plot_close: Callable[[], None] | None = None
    if enable_realtime_loss_plot:
        plot_update, plot_close = _build_realtime_loss_plotter()

    model.train()
    final_loss = float("nan")
    try:
        for step_idx in range(steps):
            optimizer.zero_grad(set_to_none=True)
            pred = model(branch_input, trunk_input)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
            if plot_update is not None:
                plot_update(step_idx + 1, final_loss)
            if log_steps > 0 and ((step_idx + 1) % log_steps == 0):
                print(f"step={step_idx + 1:05d} loss={final_loss:.6e}")

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
                    criterion(model(branch_input, trunk_input), target)
                    .detach()
                    .cpu()
                    .item()
                )
    finally:
        if plot_close is not None:
            plot_close()
    return final_loss


def _config_to_state(config: CavityLatentDeepONetConfig) -> dict[str, object]:
    return {
        "case_path": config.case_path,
        "run_root": config.run_root,
        "run_openfoam": config.run_openfoam,
        "clean_run_root": config.clean_run_root,
        "nu_min": config.nu_min,
        "nu_max": config.nu_max,
        "reynolds_min": config.reynolds_min,
        "reynolds_max": config.reynolds_max,
        "viscosity_seed": config.viscosity_seed,
        "train_case_count": config.train_case_count,
        "validate_case_count": config.validate_case_count,
        "test_case_count": config.test_case_count,
        "manifest_filename": config.manifest_filename,
        "reconstruction_mode": config.reconstruction_mode,
        "field_order": tuple(config.field_order),
        "pod_modes": dict(config.pod_modes),
        "autoencoder_latent_dims": dict(config.autoencoder_latent_dims),
        "branch_layers": config.branch_layers,
        "trunk_layers": config.trunk_layers,
        "decoder_layers": config.decoder_layers,
        "layer_size": config.layer_size,
        "decoder_layer_size": config.decoder_layer_size,
        "activation_fn": config.activation_fn,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "train_steps": config.train_steps,
        "log_steps": config.log_steps,
        "enable_realtime_loss_plot": config.enable_realtime_loss_plot,
        "lbfgs_steps": config.lbfgs_steps,
        "lbfgs_lr": config.lbfgs_lr,
        "normalize_inputs": config.normalize_inputs,
        "normalize_targets": config.normalize_targets,
        "device": config.device,
        "dtype": config.dtype,
        "save_visualizations": config.save_visualizations,
        "visualization_dir": config.visualization_dir,
        "visualization_max_cases": config.visualization_max_cases,
        "model_checkpoint_path": config.model_checkpoint_path,
        "save_trained_model": config.save_trained_model,
    }


def save_checkpoint(
    path: Path,
    model: FieldwiseLatentDeepONet,
    *,
    config: CavityLatentDeepONetConfig,
    branch_normalizer: TensorNormalizer,
    trunk_normalizer: TensorNormalizer,
    target_normalizer: TensorNormalizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pod_bases = {
        field_name: getattr(model, f"pod_basis_{field_name}").detach().cpu()
        for field_name in model.output_channel_names
        if hasattr(model, f"pod_basis_{field_name}")
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": _config_to_state(config),
            "pod_bases": pod_bases,
            "branch_normalizer": branch_normalizer.to_state(),
            "trunk_normalizer": trunk_normalizer.to_state(),
            "target_normalizer": target_normalizer.to_state(),
        },
        path,
    )


def _load_checkpoint_payload(path: Path, device: torch.device | str) -> dict[str, object]:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint at {path} must be a mapping, got {type(state)}")
    return dict(state)


def _load_config_from_payload(payload: Mapping[str, object]) -> CavityLatentDeepONetConfig:
    config_state = payload.get("config")
    if not isinstance(config_state, dict):
        raise KeyError("Checkpoint missing `config`.")
    return CavityLatentDeepONetConfig(**config_state)


def _load_pod_bases_from_payload(
    payload: Mapping[str, object],
    config: CavityLatentDeepONetConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Tensor] | None:
    if normalized_reconstruction_mode(config.reconstruction_mode) != "pod":
        return None

    pod_bases = payload.get("pod_bases")
    if isinstance(pod_bases, dict):
        return {
            field_name: torch.as_tensor(pod_bases[field_name], device=device, dtype=dtype)
            for field_name in config.field_order
        }

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError("Checkpoint missing `model_state_dict`.")
    bases: dict[str, Tensor] = {}
    for field_name in config.field_order:
        key = f"pod_basis_{field_name}"
        if key not in model_state:
            raise KeyError(
                f"Checkpoint missing POD basis buffer `{key}` required to rebuild model."
            )
        bases[field_name] = torch.as_tensor(model_state[key], device=device, dtype=dtype)
    return bases


def load_checkpoint(
    path: Path,
    model: FieldwiseLatentDeepONet,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[
    CavityLatentDeepONetConfig,
    TensorNormalizer,
    TensorNormalizer,
    TensorNormalizer,
]:
    payload = _load_checkpoint_payload(path, device)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError("Checkpoint missing `model_state_dict`.")
    model.load_state_dict(model_state)

    config = _load_config_from_payload(payload)

    branch_state = payload.get("branch_normalizer")
    trunk_state = payload.get("trunk_normalizer")
    target_state = payload.get("target_normalizer")
    if not isinstance(branch_state, dict):
        raise KeyError("Checkpoint missing `branch_normalizer`.")
    if not isinstance(trunk_state, dict):
        raise KeyError("Checkpoint missing `trunk_normalizer`.")
    if not isinstance(target_state, dict):
        raise KeyError("Checkpoint missing `target_normalizer`.")

    return (
        config,
        TensorNormalizer.from_state(branch_state, device=device, dtype=dtype),
        TensorNormalizer.from_state(trunk_state, device=device, dtype=dtype),
        TensorNormalizer.from_state(target_state, device=device, dtype=dtype),
    )


def load_model_from_checkpoint(
    path: Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[
    FieldwiseLatentDeepONet,
    CavityLatentDeepONetConfig,
    TensorNormalizer,
    TensorNormalizer,
    TensorNormalizer,
]:
    payload = _load_checkpoint_payload(path, device)
    config = _load_config_from_payload(payload)
    pod_bases = _load_pod_bases_from_payload(
        payload,
        config,
        device=device,
        dtype=dtype,
    )
    model = build_latent_deeponet(config, pod_bases=pod_bases).to(
        device=device,
        dtype=dtype,
    )
    loaded_config, branch_normalizer, trunk_normalizer, target_normalizer = load_checkpoint(
        path,
        model,
        device=device,
        dtype=dtype,
    )
    return model, loaded_config, branch_normalizer, trunk_normalizer, target_normalizer
