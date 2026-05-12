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

r"""Deep operator network (DeepONet) for field operator learning.

This implementation uses branch and trunk subnetworks whose outputs are fused
with a channel-wise dot product to predict velocity and pressure fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core import ModelMetaData, Module


@dataclass
class DeepONetMetaData(ModelMetaData):
    r"""Metadata flags for DeepONet support."""

    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = True
    # Inference
    onnx: bool = True
    onnx_runtime: bool = True
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class DeepONet(Module):
    r"""DeepONet model for Navier-Stokes operator prediction.

    The model predicts velocity components and pressure channels from a branch
    input representing sensor/function values and a trunk input representing
    query coordinates.

    Parameters
    ----------
    branch_in_features : int
        Number of branch-input features :math:`D_b`.
    trunk_in_features : int
        Number of trunk-input features :math:`D_t`.
    latent_dim : int, optional, default=64
        Latent basis size :math:`L` used per output channel.
    velocity_dim : int, optional, default=3
        Number of velocity components. Output channels are
        :math:`velocity\_dim + 1`, with pressure as the final channel.
    branch_layers : int, optional, default=2
        Number of hidden layers in the default branch network.
    trunk_layers : int, optional, default=2
        Number of hidden layers in the default trunk network.
    layer_size : int, optional, default=64
        Hidden layer width for default branch/trunk MLPs.
    activation_fn : str, optional, default="gelu"
        Activation function used by default branch/trunk MLPs.
    branch_net : Module | None, optional, default=None
        Custom branch network. Must map to :math:`C_{out} \cdot L`.
    trunk_net : Module | None, optional, default=None
        Custom trunk network. Must map to :math:`C_{out} \cdot L`.

    Forward
    -------
    branch_input : torch.Tensor
        Branch tensor of shape :math:`(B, D_b)`.
    trunk_input : torch.Tensor
        Trunk tensor of shape :math:`(B, D_t)` or :math:`(B, Q, D_t)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out})` or
        :math:`(B, Q, C_{out})`, where :math:`C_{out} = velocity\_dim + 1`.
        Pressure is always the last output channel.
    """

    def __init__(
        self,
        branch_in_features: int,
        trunk_in_features: int,
        latent_dim: int = 64,
        velocity_dim: int = 3,
        branch_layers: int = 2,
        trunk_layers: int = 2,
        layer_size: int = 64,
        activation_fn: str = "gelu",
        branch_net: Module | None = None,
        trunk_net: Module | None = None,
    ) -> None:
        super().__init__(meta=DeepONetMetaData())

        if velocity_dim < 1:
            raise ValueError(f"Expected velocity_dim >= 1, but got {velocity_dim}")
        if latent_dim < 1:
            raise ValueError(f"Expected latent_dim >= 1, but got {latent_dim}")

        self.branch_in_features = branch_in_features
        self.trunk_in_features = trunk_in_features
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.out_features = velocity_dim + 1
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.layer_size = layer_size
        self.activation_fn = activation_fn

        latent_features = self.out_features * self.latent_dim
        self._validate_custom_network("branch_net", branch_net)
        self._validate_custom_network("trunk_net", trunk_net)

        self.branch_net = branch_net or self._build_mlp(
            in_features=self.branch_in_features,
            out_features=latent_features,
            num_layers=self.branch_layers,
            hidden_size=self.layer_size,
            activation_fn=self.activation_fn,
        )
        self.trunk_net = trunk_net or self._build_mlp(
            in_features=self.trunk_in_features,
            out_features=latent_features,
            num_layers=self.trunk_layers,
            hidden_size=self.layer_size,
            activation_fn=self.activation_fn,
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features))

    @staticmethod
    def _validate_custom_network(name: str, network: Module | None) -> None:
        if network is not None and not isinstance(network, Module):
            raise TypeError(
                f"Expected {name} to be a physicsnemo.Module or None, but got "
                f"{network.__class__.__name__}. Convert PyTorch modules with "
                "physicsnemo.Module.from_torch before passing them to DeepONet."
            )

    @staticmethod
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

    @classmethod
    def _build_mlp(
        cls,
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
            layers.append(cls._get_activation(activation_fn))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, out_features))
        return nn.Sequential(*layers)

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        r"""Output channel names with pressure as the final channel."""
        if self.velocity_dim == 3:
            velocity_names: Sequence[str] = ("u", "v", "w")
        else:
            velocity_names = tuple(f"u{i}" for i in range(self.velocity_dim))
        return tuple(velocity_names) + ("p",)

    def _validate_inputs(self, branch_input: Tensor, trunk_input: Tensor) -> None:
        if branch_input.ndim != 2:
            raise ValueError(
                f"Expected branch_input with shape (B, {self.branch_in_features}) but got tensor with shape {tuple(branch_input.shape)}"
            )
        if branch_input.shape[-1] != self.branch_in_features:
            raise ValueError(
                f"Expected branch_input with shape (B, {self.branch_in_features}) but got tensor with shape {tuple(branch_input.shape)}"
            )

        if trunk_input.ndim not in (2, 3):
            raise ValueError(
                f"Expected trunk_input with shape (B, {self.trunk_in_features}) or (B, Q, {self.trunk_in_features}) but got tensor with shape {tuple(trunk_input.shape)}"
            )
        if trunk_input.shape[-1] != self.trunk_in_features:
            raise ValueError(
                f"Expected trunk_input with shape (B, {self.trunk_in_features}) or (B, Q, {self.trunk_in_features}) but got tensor with shape {tuple(trunk_input.shape)}"
            )
        if branch_input.shape[0] != trunk_input.shape[0]:
            raise ValueError(
                f"Expected branch and trunk batch sizes to match but got {branch_input.shape[0]} and {trunk_input.shape[0]}"
            )

    def _validate_latent_shapes(
        self, branch_latent: Tensor, trunk_latent: Tensor, trunk_rank: int
    ) -> None:
        if branch_latent.ndim != 2:
            raise ValueError(
                f"Expected branch network output shape (B, {self.out_features * self.latent_dim}) but got tensor with shape {tuple(branch_latent.shape)}"
            )
        if branch_latent.shape[-1] != self.out_features * self.latent_dim:
            raise ValueError(
                f"Expected branch network output shape (B, {self.out_features * self.latent_dim}) but got tensor with shape {tuple(branch_latent.shape)}"
            )

        if trunk_rank == 2:
            expected_shape_msg = (
                f"(B, {self.out_features * self.latent_dim}) from trunk network"
            )
            if trunk_latent.ndim != 2 or trunk_latent.shape[-1] != self.out_features * self.latent_dim:
                raise ValueError(
                    f"Expected trunk network output shape {expected_shape_msg} but got tensor with shape {tuple(trunk_latent.shape)}"
                )
        else:
            expected_shape_msg = (
                f"(B*Q, {self.out_features * self.latent_dim}) from trunk network"
            )
            if trunk_latent.ndim != 2 or trunk_latent.shape[-1] != self.out_features * self.latent_dim:
                raise ValueError(
                    f"Expected trunk network output shape {expected_shape_msg} but got tensor with shape {tuple(trunk_latent.shape)}"
                )

    def forward(
        self,
        branch_input: Float[Tensor, "batch branch_in_features"],
        trunk_input: Float[Tensor, "batch trunk_in_features"]
        | Float[Tensor, "batch query trunk_in_features"],
    ) -> Float[Tensor, "batch out_features"] | Float[Tensor, "batch query out_features"]:
        r"""Predict velocity and pressure channels at query locations."""
        if not torch.compiler.is_compiling():
            self._validate_inputs(branch_input, trunk_input)

        branch_latent = self.branch_net(branch_input)

        trunk_rank = trunk_input.ndim
        if trunk_rank == 2:
            flat_trunk_input = trunk_input
        else:
            batch_size, num_query, _ = trunk_input.shape
            flat_trunk_input = trunk_input.reshape(batch_size * num_query, -1)

        trunk_latent = self.trunk_net(flat_trunk_input)

        if not torch.compiler.is_compiling():
            self._validate_latent_shapes(branch_latent, trunk_latent, trunk_rank)

        batch_size = branch_input.shape[0]
        branch_basis = branch_latent.reshape(batch_size, self.out_features, self.latent_dim)

        if trunk_rank == 2:
            trunk_basis = trunk_latent.reshape(batch_size, self.out_features, self.latent_dim)
            output = torch.einsum("bol,bol->bo", branch_basis, trunk_basis)
            output = output + self.bias
        else:
            trunk_basis = trunk_latent.reshape(
                batch_size, num_query, self.out_features, self.latent_dim
            )
            output = torch.einsum("bol,bqol->bqo", branch_basis, trunk_basis)
            output = output + self.bias.view(1, 1, -1)

        return output
