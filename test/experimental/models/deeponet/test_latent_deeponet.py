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

import random

import pytest
import torch
import torch.nn as nn

from physicsnemo.core import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.deeponet import LatentDeepONet
from test.common import validate_checkpoint, validate_forward_accuracy


class LatentLinear(Module):
    """Small serializable PhysicsNeMo module for custom LatentDeepONet tests."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(meta=ModelMetaData())
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize("mode", ["pod", "decoder"])
def test_latent_deeponet_constructor_defaults(mode):
    """Test LatentDeepONet constructor defaults for each reconstruction mode."""
    pod_basis = torch.randn(4, 16) if mode == "pod" else None
    model = LatentDeepONet(
        branch_in_features=32,
        trunk_in_features=3,
        out_features=4,
        latent_dim=16,
        reconstruction_mode=mode,
        pod_basis=pod_basis,
    )

    assert model.activation_fn == "gelu"
    assert model.out_features == 4
    assert model.latent_dim == 16
    assert model.reconstruction_mode == mode
    assert isinstance(model, Module)
    assert hasattr(model, "branch_net")
    assert hasattr(model, "trunk_net")
    assert hasattr(model, "latent_bias")
    assert hasattr(model, "output_bias")
    assert hasattr(model, "meta")
    if mode == "decoder":
        assert hasattr(model, "decoder_net")
        assert model.pod_basis is None
    else:
        assert model.decoder_net is None
        assert model.pod_basis.shape == (4, 16)


def test_latent_deeponet_constructor_custom_args():
    """Test LatentDeepONet custom constructor arguments and attributes."""
    model = LatentDeepONet(
        branch_in_features=16,
        trunk_in_features=2,
        out_features=5,
        latent_dim=24,
        reconstruction_mode="decoder",
        branch_layers=3,
        trunk_layers=4,
        decoder_layers=1,
        layer_size=48,
        decoder_layer_size=32,
        activation_fn="relu",
    )

    assert model.activation_fn == "relu"
    assert model.out_features == 5
    assert model.latent_dim == 24
    assert model.branch_layers == 3
    assert model.trunk_layers == 4
    assert model.decoder_layers == 1
    assert model.layer_size == 48
    assert model.decoder_layer_size == 32


def test_latent_deeponet_forward_pod_scalar_query(device):
    """Test POD-mode forward with rank-2 trunk input."""
    torch.manual_seed(0)
    pod_basis = torch.randn(5, 12, device=device)
    model = LatentDeepONet(
        branch_in_features=20,
        trunk_in_features=3,
        out_features=5,
        latent_dim=12,
        reconstruction_mode="pod",
        pod_basis=pod_basis,
    ).to(device)

    batch_size = 5
    branch_input = torch.randn(batch_size, 20, device=device)
    trunk_input = torch.randn(batch_size, 3, device=device)
    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, 5)
    assert validate_forward_accuracy(
        model,
        (branch_input, trunk_input),
        file_name="experimental/models/deeponet/data/latent_deeponet_pod_scalar_output.pth",
    )


def test_latent_deeponet_forward_pod_multi_query(device):
    """Test POD-mode forward with rank-3 trunk input."""
    torch.manual_seed(0)
    pod_basis = torch.randn(5, 12, device=device)
    model = LatentDeepONet(
        branch_in_features=20,
        trunk_in_features=3,
        out_features=5,
        latent_dim=12,
        reconstruction_mode="pod",
        pod_basis=pod_basis,
    ).to(device)

    batch_size = 4
    num_query = 7
    branch_input = torch.randn(batch_size, 20, device=device)
    trunk_input = torch.randn(batch_size, num_query, 3, device=device)
    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, num_query, 5)
    assert validate_forward_accuracy(
        model,
        (branch_input, trunk_input),
        file_name="experimental/models/deeponet/data/latent_deeponet_pod_multi_query_output.pth",
    )


def test_latent_deeponet_forward_decoder_scalar_query(device):
    """Test decoder-mode forward with rank-2 trunk input."""
    torch.manual_seed(0)
    model = LatentDeepONet(
        branch_in_features=20,
        trunk_in_features=3,
        out_features=5,
        latent_dim=12,
        reconstruction_mode="decoder",
    ).to(device)

    batch_size = 5
    branch_input = torch.randn(batch_size, 20, device=device)
    trunk_input = torch.randn(batch_size, 3, device=device)
    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, 5)
    assert validate_forward_accuracy(
        model,
        (branch_input, trunk_input),
        file_name="experimental/models/deeponet/data/latent_deeponet_decoder_scalar_output.pth",
    )


@pytest.mark.parametrize("mode", ["pod", "decoder"])
def test_latent_deeponet_predict_latent_shapes(mode, device):
    """Test latent prediction shapes for both trunk input ranks."""
    torch.manual_seed(3)
    pod_basis = torch.randn(4, 10, device=device) if mode == "pod" else None
    model = LatentDeepONet(
        branch_in_features=8,
        trunk_in_features=2,
        out_features=4,
        latent_dim=10,
        reconstruction_mode=mode,
        pod_basis=pod_basis,
    ).to(device)

    branch_input = torch.randn(3, 8, device=device)
    trunk_input_rank2 = torch.randn(3, 2, device=device)
    latent_rank2 = model.predict_latent(branch_input, trunk_input_rank2)
    assert latent_rank2.shape == (3, 10)

    trunk_input_rank3 = torch.randn(3, 6, 2, device=device)
    latent_rank3 = model.predict_latent(branch_input, trunk_input_rank3)
    assert latent_rank3.shape == (3, 6, 10)


def test_latent_deeponet_custom_submodule_path(device):
    """Test custom branch/trunk/decoder PhysicsNeMo module injection path."""
    torch.manual_seed(0)
    model = LatentDeepONet(
        branch_in_features=6,
        trunk_in_features=2,
        out_features=4,
        latent_dim=5,
        reconstruction_mode="decoder",
        branch_net=LatentLinear(6, 5),
        trunk_net=LatentLinear(2, 5),
        decoder_net=LatentLinear(5, 4),
    ).to(device)

    branch_input = torch.randn(2, 6, device=device)
    trunk_input = torch.randn(2, 9, 2, device=device)
    output = model(branch_input, trunk_input)

    assert output.shape == (2, 9, 4)


def test_latent_deeponet_rejects_plain_torch_custom_submodules():
    """Test plain PyTorch custom modules are rejected before checkpointing."""
    pod_basis = torch.randn(4, 5)
    with pytest.raises(TypeError, match="branch_net"):
        LatentDeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            out_features=4,
            latent_dim=5,
            reconstruction_mode="pod",
            branch_net=nn.Linear(6, 5),
            trunk_net=LatentLinear(2, 5),
            pod_basis=pod_basis,
        )

    with pytest.raises(TypeError, match="trunk_net"):
        LatentDeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            out_features=4,
            latent_dim=5,
            reconstruction_mode="pod",
            branch_net=LatentLinear(6, 5),
            trunk_net=nn.Linear(2, 5),
            pod_basis=pod_basis,
        )

    with pytest.raises(TypeError, match="decoder_net"):
        LatentDeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            out_features=4,
            latent_dim=5,
            reconstruction_mode="decoder",
            decoder_net=nn.Linear(5, 4),
        )


def test_latent_deeponet_invalid_constructor_checks():
    """Test constructor validation checks."""
    with pytest.raises(ValueError, match="reconstruction_mode"):
        LatentDeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            out_features=4,
            latent_dim=5,
            reconstruction_mode="bad",
        )

    with pytest.raises(ValueError, match="pod_basis is required"):
        LatentDeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            out_features=4,
            latent_dim=5,
            reconstruction_mode="pod",
        )

    with pytest.raises(ValueError, match="pod_basis must be None"):
        LatentDeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            out_features=4,
            latent_dim=5,
            reconstruction_mode="decoder",
            pod_basis=torch.randn(4, 5),
        )

    with pytest.raises(ValueError, match="decoder_net must be None"):
        LatentDeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            out_features=4,
            latent_dim=5,
            reconstruction_mode="pod",
            decoder_net=LatentLinear(5, 4),
            pod_basis=torch.randn(4, 5),
        )

    with pytest.raises(ValueError, match="pod_basis"):
        LatentDeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            out_features=4,
            latent_dim=5,
            reconstruction_mode="pod",
            pod_basis=torch.randn(4, 6),
        )


def test_latent_deeponet_invalid_shape_checks(device):
    """Test shape validation for branch/trunk inputs and latent decoder output."""
    pod_basis = torch.randn(4, 5, device=device)
    model = LatentDeepONet(
        branch_in_features=6,
        trunk_in_features=2,
        out_features=4,
        latent_dim=5,
        reconstruction_mode="pod",
        pod_basis=pod_basis,
    ).to(device)

    branch_input = torch.randn(3, 5, device=device)
    trunk_input = torch.randn(3, 2, device=device)
    with pytest.raises(ValueError, match="branch_input"):
        model(branch_input, trunk_input)

    branch_input = torch.randn(3, 6, device=device)
    trunk_input = torch.randn(3, 4, device=device)
    with pytest.raises(ValueError, match="trunk_input"):
        model(branch_input, trunk_input)

    trunk_input = torch.randn(3, 2, 2, 2, device=device)
    with pytest.raises(ValueError, match="trunk_input"):
        model(branch_input, trunk_input)

    trunk_input = torch.randn(4, 2, device=device)
    with pytest.raises(ValueError, match="batch sizes"):
        model(branch_input, trunk_input)

    with pytest.raises(ValueError, match="latent_coefficients"):
        model.decode_latent(torch.randn(3, 4, device=device))

    bad_decoder = LatentDeepONet(
        branch_in_features=6,
        trunk_in_features=2,
        out_features=4,
        latent_dim=5,
        reconstruction_mode="decoder",
        decoder_net=LatentLinear(5, 3),
    ).to(device)
    with pytest.raises(ValueError, match="decoder output shape"):
        bad_decoder.decode_latent(torch.randn(3, 5, device=device))


@pytest.mark.parametrize("mode", ["pod", "decoder"])
def test_latent_deeponet_checkpoint(mode, device):
    """Test LatentDeepONet checkpoint save/load parity."""
    torch.manual_seed(21)
    kwargs = {
        "branch_in_features": 10,
        "trunk_in_features": 3,
        "out_features": 4,
        "latent_dim": 8,
        "reconstruction_mode": mode,
    }
    if mode == "pod":
        kwargs["pod_basis"] = torch.randn(4, 8, device=device)

    model_1 = LatentDeepONet(**kwargs).to(device)
    model_2 = LatentDeepONet(**kwargs).to(device)

    batch_size = random.randint(1, 4)
    num_query = 5
    branch_input = torch.randn(batch_size, 10, device=device)
    trunk_input = torch.randn(batch_size, num_query, 3, device=device)

    assert validate_checkpoint(model_1, model_2, (branch_input, trunk_input))


def test_latent_deeponet_checkpoint_attributes_pod(device):
    """Test pod-mode checkpoint preserves attributes, basis, and outputs."""
    import tempfile
    from pathlib import Path

    torch.manual_seed(14)
    pod_basis = torch.randn(3, 7, device=device)
    original_model = LatentDeepONet(
        branch_in_features=12,
        trunk_in_features=3,
        out_features=3,
        latent_dim=7,
        reconstruction_mode="pod",
        pod_basis=pod_basis,
        branch_layers=3,
        trunk_layers=2,
        layer_size=40,
    ).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_latent_deeponet_pod.mdlus"
        original_model.save(str(checkpoint_path))
        loaded_model = Module.from_checkpoint(str(checkpoint_path)).to(device)

        assert loaded_model.reconstruction_mode == original_model.reconstruction_mode
        assert loaded_model.out_features == original_model.out_features
        assert loaded_model.latent_dim == original_model.latent_dim
        torch.testing.assert_close(loaded_model.pod_basis, original_model.pod_basis)

        branch_input = torch.randn(3, 12, device=device)
        trunk_input = torch.randn(3, 6, 3, device=device)
        with torch.no_grad():
            original_output = original_model(branch_input, trunk_input)
            loaded_output = loaded_model(branch_input, trunk_input)
        torch.testing.assert_close(original_output, loaded_output)


def test_latent_deeponet_checkpoint_attributes_decoder(device):
    """Test decoder-mode checkpoint preserves attributes and outputs."""
    import tempfile
    from pathlib import Path

    torch.manual_seed(15)
    original_model = LatentDeepONet(
        branch_in_features=12,
        trunk_in_features=3,
        out_features=3,
        latent_dim=7,
        reconstruction_mode="decoder",
        branch_layers=3,
        trunk_layers=2,
        decoder_layers=3,
        layer_size=40,
        decoder_layer_size=24,
    ).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_latent_deeponet_decoder.mdlus"
        original_model.save(str(checkpoint_path))
        loaded_model = Module.from_checkpoint(str(checkpoint_path)).to(device)

        assert loaded_model.reconstruction_mode == original_model.reconstruction_mode
        assert loaded_model.out_features == original_model.out_features
        assert loaded_model.latent_dim == original_model.latent_dim
        assert loaded_model.pod_basis is None

        branch_input = torch.randn(3, 12, device=device)
        trunk_input = torch.randn(3, 6, 3, device=device)
        with torch.no_grad():
            original_output = original_model(branch_input, trunk_input)
            loaded_output = loaded_model(branch_input, trunk_input)
        torch.testing.assert_close(original_output, loaded_output)
