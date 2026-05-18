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

r"""Latent DeepONet with POD or learned decoder reconstruction backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core import ModelMetaData, Module


@dataclass
class LatentDeepONetMetaData(ModelMetaData):
    r"""Metadata flags for LatentDeepONet support."""

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


class LatentDeepONet(Module):
    r"""Latent DeepONet with backend-selectable output reconstruction.

    The model predicts latent coefficients from branch and trunk inputs, then
    reconstructs physical output channels with either a fixed POD basis or a
    learned decoder network.

    Parameters
    ----------
    branch_in_features : int
        Number of branch-input features :math:`D_b`.
    trunk_in_features : int
        Number of trunk-input features :math:`D_t`.
    out_features : int
        Number of reconstructed output channels :math:`C_{out}`.
    latent_dim : int
        Latent coefficient width :math:`L`.
    reconstruction_mode : Literal["pod", "decoder"]
        Reconstruction backend selector.
    branch_layers : int, optional, default=2
        Number of hidden layers in the default branch network.
    trunk_layers : int, optional, default=2
        Number of hidden layers in the default trunk network.
    decoder_layers : int, optional, default=2
        Number of hidden layers in the default decoder network.
    layer_size : int, optional, default=64
        Hidden layer width for default branch/trunk MLPs.
    decoder_layer_size : int, optional, default=64
        Hidden layer width for default decoder MLP.
    activation_fn : str, optional, default="gelu"
        Activation function used by default branch/trunk/decoder MLPs.
    branch_net : Module | None, optional, default=None
        Custom branch network. Must map to :math:`L`.
    trunk_net : Module | None, optional, default=None
        Custom trunk network. Must map to :math:`L`.
    decoder_net : Module | None, optional, default=None
        Custom decoder for decoder mode. Must map :math:`L \rightarrow C_{out}`.
    pod_basis : Tensor | list[list[float]] | tuple[tuple[float, ...], ...] | None, optional, default=None
        POD basis with shape :math:`(C_{out}, L)` used in POD mode.

    Forward
    -------
    branch_input : torch.Tensor
        Branch tensor of shape :math:`(B, D_b)`.
    trunk_input : torch.Tensor
        Trunk tensor of shape :math:`(B, D_t)` or :math:`(B, Q, D_t)`.

    Outputs
    -------
    torch.Tensor
        Reconstructed output tensor of shape :math:`(B, C_{out})` or
        :math:`(B, Q, C_{out})`.
    """

    def __init__(
        self,
        branch_in_features: int,
        trunk_in_features: int,
        out_features: int,
        latent_dim: int,
        reconstruction_mode: Literal["pod", "decoder"],
        branch_layers: int = 2,
        trunk_layers: int = 2,
        decoder_layers: int = 2,
        layer_size: int = 64,
        decoder_layer_size: int = 64,
        activation_fn: str = "gelu",
        branch_net: Module | None = None,
        trunk_net: Module | None = None,
        decoder_net: Module | None = None,
        pod_basis: Tensor
        | list[list[float]]
        | tuple[tuple[float, ...], ...]
        | None = None,
    ) -> None:
        super().__init__(meta=LatentDeepONetMetaData())

        if out_features < 1:
            raise ValueError(f"Expected out_features >= 1, but got {out_features}")
        if latent_dim < 1:
            raise ValueError(f"Expected latent_dim >= 1, but got {latent_dim}")
        if reconstruction_mode not in ("pod", "decoder"):
            raise ValueError(
                f"Expected reconstruction_mode to be one of ('pod', 'decoder') but got {reconstruction_mode}"
            )

        self.branch_in_features = branch_in_features
        self.trunk_in_features = trunk_in_features
        self.out_features = out_features
        self.latent_dim = latent_dim
        self.reconstruction_mode = reconstruction_mode
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.decoder_layers = decoder_layers
        self.layer_size = layer_size
        self.decoder_layer_size = decoder_layer_size
        self.activation_fn = activation_fn

        self._validate_custom_network("branch_net", branch_net)
        self._validate_custom_network("trunk_net", trunk_net)
        self._validate_custom_network("decoder_net", decoder_net)

        self.branch_net = branch_net or self._build_mlp(
            in_features=self.branch_in_features,
            out_features=self.latent_dim,
            num_layers=self.branch_layers,
            hidden_size=self.layer_size,
            activation_fn=self.activation_fn,
        )
        self.trunk_net = trunk_net or self._build_mlp(
            in_features=self.trunk_in_features,
            out_features=self.latent_dim,
            num_layers=self.trunk_layers,
            hidden_size=self.layer_size,
            activation_fn=self.activation_fn,
        )
        self.latent_bias = nn.Parameter(torch.zeros(self.latent_dim))
        self.output_bias = nn.Parameter(torch.zeros(self.out_features))

        if self.reconstruction_mode == "pod":
            if decoder_net is not None:
                raise ValueError(
                    "decoder_net must be None when reconstruction_mode='pod'"
                )
            if pod_basis is None:
                raise ValueError("pod_basis is required when reconstruction_mode='pod'")
            basis_tensor = torch.as_tensor(pod_basis, dtype=torch.float32)
            self._validate_pod_basis(basis_tensor)
            self.register_buffer("pod_basis", basis_tensor)
            self.decoder_net = None
        else:
            if pod_basis is not None:
                raise ValueError(
                    "pod_basis must be None when reconstruction_mode='decoder'"
                )
            self.pod_basis = None
            self.decoder_net = decoder_net or self._build_mlp(
                in_features=self.latent_dim,
                out_features=self.out_features,
                num_layers=self.decoder_layers,
                hidden_size=self.decoder_layer_size,
                activation_fn=self.activation_fn,
            )

        self._args = {
            "__name__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "__args__": {
                "branch_in_features": self.branch_in_features,
                "trunk_in_features": self.trunk_in_features,
                "out_features": self.out_features,
                "latent_dim": self.latent_dim,
                "reconstruction_mode": self.reconstruction_mode,
                "branch_layers": self.branch_layers,
                "trunk_layers": self.trunk_layers,
                "decoder_layers": self.decoder_layers,
                "layer_size": self.layer_size,
                "decoder_layer_size": self.decoder_layer_size,
                "activation_fn": self.activation_fn,
                "branch_net": branch_net,
                "trunk_net": trunk_net,
                "decoder_net": decoder_net,
                "pod_basis": self.pod_basis.detach().cpu().tolist()
                if self.pod_basis is not None
                else None,
            },
        }

    @staticmethod
    def _validate_custom_network(name: str, network: Module | None) -> None:
        if network is not None and not isinstance(network, Module):
            raise TypeError(
                f"Expected {name} to be a physicsnemo.Module or None, but got "
                f"{network.__class__.__name__}. Convert PyTorch modules with "
                "physicsnemo.Module.from_torch before passing them to LatentDeepONet."
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
        if branch_latent.ndim != 2 or branch_latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected branch network output shape (B, {self.latent_dim}) but got tensor with shape {tuple(branch_latent.shape)}"
            )

        if trunk_rank == 2:
            if trunk_latent.ndim != 2 or trunk_latent.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"Expected trunk network output shape (B, {self.latent_dim}) but got tensor with shape {tuple(trunk_latent.shape)}"
                )
        else:
            if trunk_latent.ndim != 2 or trunk_latent.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"Expected trunk network output shape (B*Q, {self.latent_dim}) but got tensor with shape {tuple(trunk_latent.shape)}"
                )

    def _validate_pod_basis(self, pod_basis: Tensor) -> None:
        if pod_basis.ndim != 2 or pod_basis.shape != (
            self.out_features,
            self.latent_dim,
        ):
            raise ValueError(
                f"Expected pod_basis with shape ({self.out_features}, {self.latent_dim}) but got tensor with shape {tuple(pod_basis.shape)}"
            )

    def predict_latent(
        self,
        branch_input: Float[Tensor, "batch branch_in_features"],
        trunk_input: Float[Tensor, "batch trunk_in_features"]
        | Float[Tensor, "batch query trunk_in_features"],
    ) -> Float[Tensor, "batch latent_dim"] | Float[Tensor, "batch query latent_dim"]:
        r"""Predict latent coefficients from branch and trunk inputs."""
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

        if trunk_rank == 2:
            return branch_latent * trunk_latent + self.latent_bias

        trunk_coeff = trunk_latent.reshape(batch_size, num_query, self.latent_dim)
        return branch_latent.unsqueeze(1) * trunk_coeff + self.latent_bias.view(
            1, 1, -1
        )

    def decode_latent(
        self,
        latent_coefficients: Float[Tensor, "batch latent_dim"]
        | Float[Tensor, "batch query latent_dim"],
    ) -> (
        Float[Tensor, "batch out_features"] | Float[Tensor, "batch query out_features"]
    ):
        r"""Decode latent coefficients into reconstructed physical outputs."""
        if latent_coefficients.ndim not in (2, 3):
            raise ValueError(
                f"Expected latent_coefficients with shape (B, {self.latent_dim}) or (B, Q, {self.latent_dim}) but got tensor with shape {tuple(latent_coefficients.shape)}"
            )
        if latent_coefficients.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_coefficients with shape (B, {self.latent_dim}) or (B, Q, {self.latent_dim}) but got tensor with shape {tuple(latent_coefficients.shape)}"
            )

        if self.reconstruction_mode == "pod":
            if latent_coefficients.ndim == 2:
                output = torch.einsum("bl,ol->bo", latent_coefficients, self.pod_basis)
                return output + self.output_bias
            output = torch.einsum("bql,ol->bqo", latent_coefficients, self.pod_basis)
            return output + self.output_bias.view(1, 1, -1)

        if latent_coefficients.ndim == 2:
            decoded = self.decoder_net(latent_coefficients)
            if decoded.ndim != 2 or decoded.shape[-1] != self.out_features:
                raise ValueError(
                    f"Expected decoder output shape (B, {self.out_features}) but got tensor with shape {tuple(decoded.shape)}"
                )
            return decoded + self.output_bias

        batch_size, num_query, _ = latent_coefficients.shape
        flat_coeff = latent_coefficients.reshape(batch_size * num_query, -1)
        flat_decoded = self.decoder_net(flat_coeff)
        if flat_decoded.ndim != 2 or flat_decoded.shape[-1] != self.out_features:
            raise ValueError(
                f"Expected decoder output shape (B*Q, {self.out_features}) but got tensor with shape {tuple(flat_decoded.shape)}"
            )
        return flat_decoded.reshape(
            batch_size, num_query, self.out_features
        ) + self.output_bias.view(1, 1, -1)

    def forward(
        self,
        branch_input: Float[Tensor, "batch branch_in_features"],
        trunk_input: Float[Tensor, "batch trunk_in_features"]
        | Float[Tensor, "batch query trunk_in_features"],
    ) -> (
        Float[Tensor, "batch out_features"] | Float[Tensor, "batch query out_features"]
    ):
        r"""Predict reconstructed output channels at query locations."""
        latent_coefficients = self.predict_latent(branch_input, trunk_input)
        return self.decode_latent(latent_coefficients)
