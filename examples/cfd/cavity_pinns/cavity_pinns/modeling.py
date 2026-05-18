# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from physicsnemo.core import ModelMetaData, Module

from .config import CavityPINNConfig


@dataclass
class CavityPINNMetaData(ModelMetaData):
    """Metadata flags for the example-local cavity PINN."""

    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = True
    func_torch: bool = True
    auto_grad: bool = True


def _make_activation(activation_fn: str) -> nn.Module:
    activation_map: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if activation_fn not in activation_map:
        raise ValueError(f"Unsupported activation_fn={activation_fn!r}.")
    return activation_map[activation_fn]()


class CavityPINN(Module):
    """Example-local MLP for viscosity-conditioned cavity PINN predictions."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_size: int,
        activation_fn: str,
    ) -> None:
        super().__init__(meta=CavityPINNMetaData())
        self.activation_fn = activation_fn
        self.input_layer = nn.Linear(in_features, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(max(hidden_layers, 0))]
        )
        self.output_layer = nn.Linear(hidden_size, out_features)
        self.activation = _make_activation(activation_fn)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        def init_hidden(weight: Tensor) -> None:
            if self.activation_fn == "tanh":
                gain = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(weight, gain=gain)
            else:
                nn.init.kaiming_uniform_(weight, nonlinearity="relu")

        init_hidden(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        for layer in self.hidden_layers:
            init_hidden(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.5)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            hidden = hidden + self.activation(layer(hidden))
        return self.output_layer(hidden)


def build_pinn_model(config: CavityPINNConfig) -> CavityPINN:
    if config.spatial_dim != 2:
        raise NotImplementedError("Current cavity PINN model wiring supports spatial_dim=2 only.")
    return CavityPINN(
        in_features=config.spatial_dim + 1,
        out_features=config.spatial_dim + 1,
        hidden_layers=config.model_layers,
        hidden_size=config.model_layer_size,
        activation_fn=config.activation_fn,
    )


def predict_fields(model: nn.Module, coordinates: Tensor, viscosity: Tensor) -> Tensor:
    model_input = torch.cat((coordinates, viscosity), dim=1)
    return model(model_input)
