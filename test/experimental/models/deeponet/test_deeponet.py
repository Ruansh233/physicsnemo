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
from physicsnemo.experimental.models.deeponet import DeepONet
from test.common import validate_checkpoint, validate_forward_accuracy


class LatentLinear(Module):
    """Small serializable PhysicsNeMo module for custom DeepONet tests."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(meta=ModelMetaData())
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_deeponet_constructor(config):
    """Test DeepONet constructor defaults and custom arguments."""
    if config == "default":
        model = DeepONet(branch_in_features=32, trunk_in_features=3)
        assert model.activation_fn == "gelu"
        assert model.velocity_dim == 3
        assert model.out_features == 4
        assert model.output_channel_names == ("u", "v", "w", "p")
        assert model.latent_dim == 64
    else:
        model = DeepONet(
            branch_in_features=16,
            trunk_in_features=2,
            latent_dim=24,
            velocity_dim=2,
            branch_layers=3,
            trunk_layers=4,
            layer_size=48,
            activation_fn="relu",
        )
        assert model.activation_fn == "relu"
        assert model.velocity_dim == 2
        assert model.out_features == 3
        assert model.output_channel_names == ("u0", "u1", "p")
        assert model.latent_dim == 24

    assert isinstance(model, Module)
    assert hasattr(model, "branch_net")
    assert hasattr(model, "trunk_net")
    assert hasattr(model, "bias")
    assert hasattr(model, "meta")


def test_deeponet_forward_scalar_query(device):
    """Test DeepONet forward with rank-2 trunk input."""
    torch.manual_seed(0)
    model = DeepONet(branch_in_features=20, trunk_in_features=3).to(device)

    batch_size = 5
    branch_input = torch.randn(batch_size, 20, device=device)
    trunk_input = torch.randn(batch_size, 3, device=device)
    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, 4)
    assert validate_forward_accuracy(
        model,
        (branch_input, trunk_input),
        file_name="experimental/models/deeponet/data/deeponet_scalar_output.pth",
    )


def test_deeponet_forward_multi_query(device):
    """Test DeepONet forward with rank-3 trunk input."""
    torch.manual_seed(0)
    model = DeepONet(branch_in_features=20, trunk_in_features=3).to(device)

    batch_size = 4
    num_query = 7
    branch_input = torch.randn(batch_size, 20, device=device)
    trunk_input = torch.randn(batch_size, num_query, 3, device=device)
    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, num_query, 4)
    assert validate_forward_accuracy(
        model,
        (branch_input, trunk_input),
        file_name="experimental/models/deeponet/data/deeponet_multi_query_output.pth",
    )


def test_deeponet_pressure_is_last_channel(device):
    """Test output channel semantics where pressure is the last channel."""
    torch.manual_seed(7)
    model = DeepONet(
        branch_in_features=8,
        trunk_in_features=3,
        velocity_dim=2,
        latent_dim=12,
        layer_size=16,
    ).to(device)

    branch_input = torch.randn(3, 8, device=device)
    trunk_input = torch.randn(3, 4, 3, device=device)
    output = model(branch_input, trunk_input)
    pressure = output[..., -1]
    velocity = output[..., :-1]

    assert model.output_channel_names[-1] == "p"
    assert velocity.shape[-1] == model.velocity_dim
    assert pressure.shape == (3, 4)


def test_deeponet_custom_submodule_path(device):
    """Test custom branch/trunk PhysicsNeMo module injection path."""
    torch.manual_seed(0)
    model = DeepONet(
        branch_in_features=6,
        trunk_in_features=2,
        latent_dim=5,
        velocity_dim=3,
        branch_net=LatentLinear(6, 20),
        trunk_net=LatentLinear(2, 20),
    ).to(device)

    branch_input = torch.randn(2, 6, device=device)
    trunk_input = torch.randn(2, 9, 2, device=device)
    output = model(branch_input, trunk_input)

    assert output.shape == (2, 9, 4)


def test_deeponet_rejects_plain_torch_custom_submodules():
    """Test plain PyTorch custom modules are rejected before checkpointing."""
    with pytest.raises(TypeError, match="branch_net"):
        DeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            latent_dim=5,
            branch_net=nn.Linear(6, 20),
            trunk_net=LatentLinear(2, 20),
        )

    with pytest.raises(TypeError, match="trunk_net"):
        DeepONet(
            branch_in_features=6,
            trunk_in_features=2,
            latent_dim=5,
            branch_net=LatentLinear(6, 20),
            trunk_net=nn.Linear(2, 20),
        )


def test_deeponet_invalid_shape_checks(device):
    """Test shape validation for branch and trunk inputs."""
    model = DeepONet(branch_in_features=6, trunk_in_features=2).to(device)

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


def test_deeponet_checkpoint(device):
    """Test DeepONet checkpoint save/load parity."""
    model_1 = DeepONet(branch_in_features=10, trunk_in_features=3).to(device)
    model_2 = DeepONet(branch_in_features=10, trunk_in_features=3).to(device)

    batch_size = random.randint(1, 4)
    num_query = 5
    branch_input = torch.randn(batch_size, 10, device=device)
    trunk_input = torch.randn(batch_size, num_query, 3, device=device)

    assert validate_checkpoint(model_1, model_2, (branch_input, trunk_input))


def test_deeponet_checkpoint_attributes(device):
    """Test that loading from checkpoint preserves attributes and outputs."""
    import tempfile
    from pathlib import Path

    original_model = DeepONet(
        branch_in_features=12,
        trunk_in_features=3,
        latent_dim=18,
        velocity_dim=2,
        branch_layers=3,
        trunk_layers=2,
        layer_size=40,
    ).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_deeponet.mdlus"
        original_model.save(str(checkpoint_path))
        loaded_model = Module.from_checkpoint(str(checkpoint_path)).to(device)

        assert loaded_model.velocity_dim == original_model.velocity_dim
        assert loaded_model.out_features == original_model.out_features
        assert loaded_model.latent_dim == original_model.latent_dim
        assert loaded_model.output_channel_names == original_model.output_channel_names

        torch.manual_seed(99)
        branch_input = torch.randn(3, 12, device=device)
        trunk_input = torch.randn(3, 6, 3, device=device)
        with torch.no_grad():
            original_output = original_model(branch_input, trunk_input)
            loaded_output = loaded_model(branch_input, trunk_input)
        torch.testing.assert_close(original_output, loaded_output)


def test_deeponet_custom_submodule_checkpoint(device):
    """Test custom PhysicsNeMo branch/trunk modules save and load."""
    original_model = DeepONet(
        branch_in_features=6,
        trunk_in_features=2,
        latent_dim=5,
        velocity_dim=3,
        branch_net=LatentLinear(6, 20),
        trunk_net=LatentLinear(2, 20),
    ).to(device)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_deeponet_custom.mdlus"
        original_model.save(str(checkpoint_path))
        loaded_model = Module.from_checkpoint(str(checkpoint_path)).to(device)

        torch.manual_seed(12)
        branch_input = torch.randn(2, 6, device=device)
        trunk_input = torch.randn(2, 4, 2, device=device)
        with torch.no_grad():
            original_output = original_model(branch_input, trunk_input)
            loaded_output = loaded_model(branch_input, trunk_input)
        torch.testing.assert_close(original_output, loaded_output)
