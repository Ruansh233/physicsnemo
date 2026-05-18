# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from physicsnemo.experimental.models.deeponet import DeepONet

from .config import CavityDeepONetConfig


@dataclass(frozen=True)
class TensorNormalizer:
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, tensor: Tensor) -> "TensorNormalizer":
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        std = torch.where(std < 1.0e-8, torch.ones_like(std), std)
        return cls(mean=mean, std=std)

    @classmethod
    def identity(cls, tensor: Tensor) -> "TensorNormalizer":
        feature_shape = (1, tensor.shape[-1])
        return cls(
            mean=torch.zeros(feature_shape, dtype=tensor.dtype, device=tensor.device),
            std=torch.ones(feature_shape, dtype=tensor.dtype, device=tensor.device),
        )

    def transform(self, tensor: Tensor) -> Tensor:
        return (tensor - self.mean) / self.std

    def inverse(self, tensor: Tensor) -> Tensor:
        return tensor * self.std + self.mean

    def to_state(self) -> dict[str, Tensor]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def from_state(
        cls,
        state: dict[str, Tensor],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "TensorNormalizer":
        return cls(
            mean=state["mean"].to(device=device, dtype=dtype),
            std=state["std"].to(device=device, dtype=dtype),
        )


def build_deeponet(config: CavityDeepONetConfig) -> DeepONet:
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


def save_checkpoint(
    path: Path,
    model: DeepONet,
    *,
    branch_normalizer: TensorNormalizer,
    trunk_normalizer: TensorNormalizer,
    target_normalizer: TensorNormalizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "branch_normalizer": branch_normalizer.to_state(),
            "trunk_normalizer": trunk_normalizer.to_state(),
            "target_normalizer": target_normalizer.to_state(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: DeepONet,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[TensorNormalizer, TensorNormalizer, TensorNormalizer]:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint at {path} must be a mapping, got {type(state)}")
    payload = dict(state)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError("Checkpoint missing `model_state_dict`.")
    model.load_state_dict(model_state)
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
        TensorNormalizer.from_state(branch_state, device=device, dtype=dtype),
        TensorNormalizer.from_state(trunk_state, device=device, dtype=dtype),
        TensorNormalizer.from_state(target_state, device=device, dtype=dtype),
    )
