# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from examples.cfd.cavity_deeponet.cavity_deeponet.modeling import TensorNormalizer
from examples.cfd.cavity_latent_deeponet.cavity_latent_deeponet.config import (
    CavityLatentDeepONetConfig,
    load_config,
)
from examples.cfd.cavity_latent_deeponet.cavity_latent_deeponet.modeling import (
    build_latent_deeponet,
    compute_field_pod_bases,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)


def test_cavity_latent_config_defaults():
    cfg = CavityLatentDeepONetConfig()
    assert cfg.reconstruction_mode == "pod"
    assert cfg.field_order == ("u", "v", "w", "p")
    assert set(cfg.pod_modes.keys()) == {"u", "v", "w", "p"}
    assert set(cfg.autoencoder_latent_dims.keys()) == {"u", "v", "w", "p"}


def test_load_config_rejects_missing_field_mapping(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        "reconstruction_mode: pod\n"
        "field_order: [u, v, w, p]\n"
        "pod_modes:\n"
        "  u: 4\n"
        "  v: 4\n"
        "  w: 2\n"
        "  p: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pod_modes"):
        load_config(config_file)


def test_build_latent_deeponet_decoder_field_dims():
    cfg = CavityLatentDeepONetConfig(
        reconstruction_mode="autoencoder",
        autoencoder_latent_dims={"u": 6, "v": 5, "w": 4, "p": 3},
        layer_size=16,
        decoder_layer_size=16,
        branch_layers=1,
        trunk_layers=1,
        decoder_layers=1,
    )
    model = build_latent_deeponet(cfg)
    assert model.output_channel_names == ("u", "v", "w", "p")
    assert model.field_latent_dims == {"u": 6, "v": 5, "w": 4, "p": 3}
    assert model.reconstruction_mode == "decoder"
    assert len({id(model.branch_encoder), id(model.trunk_encoder)}) == 2


def test_compute_field_pod_bases_uses_field_specific_ranks():
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0, 2.0],
            [2.0, 0.0, 0.0, 3.0],
            [3.0, 0.0, 0.0, 4.0],
            [2.0, 0.0, 0.0, 4.0],
            [4.0, 0.0, 0.0, 6.0],
            [6.0, 0.0, 0.0, 8.0],
            [3.0, 0.0, 0.0, 6.0],
            [6.0, 0.0, 0.0, 9.0],
            [9.0, 0.0, 0.0, 12.0],
        ],
        dtype=torch.float32,
    )
    bases = compute_field_pod_bases(
        target,
        case_count=3,
        field_order=("u", "v", "w", "p"),
        pod_modes={"u": 2, "v": 1, "w": 1, "p": 3},
    )
    assert bases["u"].shape == (3, 2)
    assert bases["p"].shape == (3, 3)
    torch.testing.assert_close(bases["u"].T @ bases["u"], torch.eye(2), atol=1e-6, rtol=1e-6)


def test_compute_field_pod_bases_rejects_rank_too_large():
    target = torch.randn(6, 4)
    with pytest.raises(ValueError, match="pod_modes\\['u'\\]"):
        compute_field_pod_bases(
            target,
            case_count=2,
            field_order=("u", "v", "w", "p"),
            pod_modes={"u": 4, "v": 1, "w": 1, "p": 1},
        )


def test_compute_field_pod_bases_rejects_missing_or_zero_rank():
    target = torch.randn(6, 4)
    with pytest.raises(ValueError, match="Missing pod_modes"):
        compute_field_pod_bases(
            target,
            case_count=2,
            field_order=("u", "v", "w", "p"),
            pod_modes={"u": 1, "v": 1, "w": 1},
        )

    with pytest.raises(ValueError, match="must be >= 1"):
        compute_field_pod_bases(
            target,
            case_count=2,
            field_order=("u", "v", "w", "p"),
            pod_modes={"u": 0, "v": 1, "w": 1, "p": 1},
        )


def test_build_latent_deeponet_pod_field_dims():
    cfg = CavityLatentDeepONetConfig(
        reconstruction_mode="pod",
        pod_modes={"u": 3, "v": 4, "w": 2, "p": 5},
        layer_size=16,
        branch_layers=1,
        trunk_layers=1,
    )
    pod_bases = {
        "u": torch.randn(7, 3),
        "v": torch.randn(7, 4),
        "w": torch.randn(7, 2),
        "p": torch.randn(7, 5),
    }
    model = build_latent_deeponet(cfg, pod_bases=pod_bases)
    assert model.field_latent_dims == {"u": 3, "v": 4, "w": 2, "p": 5}
    assert model.reconstruction_mode == "pod"
    assert model.num_points == 7
    assert len(model.field_models) == 0


@pytest.mark.parametrize("mode", ["pod", "autoencoder"])
def test_fieldwise_forward_shapes(mode):
    cfg = CavityLatentDeepONetConfig(
        reconstruction_mode=mode,
        pod_modes={"u": 2, "v": 2, "w": 2, "p": 2},
        autoencoder_latent_dims={"u": 3, "v": 3, "w": 2, "p": 2},
        layer_size=8,
        decoder_layer_size=8,
        branch_layers=1,
        trunk_layers=1,
        decoder_layers=1,
    )
    pod_bases = {
        field_name: torch.randn(5, cfg.pod_modes[field_name])
        for field_name in cfg.field_order
    }
    model = build_latent_deeponet(cfg, pod_bases=pod_bases if mode == "pod" else None)
    branch = torch.randn(7, 1)
    trunk_rank2 = torch.randn(7, 3)
    if mode == "pod":
        branch = torch.randn(10, 1)
        trunk_rank2 = torch.randn(10, 3)
    output_rank2 = model(branch, trunk_rank2)
    assert output_rank2.shape == (branch.shape[0], 4)

    branch_cases = torch.randn(2, 1)
    trunk_rank3 = torch.randn(2, 5, 3)
    output_rank3 = model(branch_cases, trunk_rank3)
    assert output_rank3.shape == (2, 5, 4)


def test_pod_forward_requires_full_case_grid():
    cfg = CavityLatentDeepONetConfig(
        reconstruction_mode="pod",
        pod_modes={"u": 2, "v": 2, "w": 2, "p": 2},
        layer_size=8,
        branch_layers=1,
        trunk_layers=1,
    )
    model = build_latent_deeponet(
        cfg,
        pod_bases={field_name: torch.randn(5, 2) for field_name in cfg.field_order},
    )
    with pytest.raises(ValueError, match="multiple of num_points"):
        model(torch.randn(7, 1), torch.randn(7, 3))


@pytest.mark.parametrize("mode", ["pod", "autoencoder"])
def test_forward_rejects_bad_trunk_feature_width(mode):
    cfg = CavityLatentDeepONetConfig(
        reconstruction_mode=mode,
        pod_modes={"u": 2, "v": 2, "w": 2, "p": 2},
        autoencoder_latent_dims={"u": 2, "v": 2, "w": 2, "p": 2},
        layer_size=8,
        decoder_layer_size=8,
        branch_layers=1,
        trunk_layers=1,
        decoder_layers=1,
    )
    pod_bases = {field_name: torch.randn(5, 2) for field_name in cfg.field_order}
    model = build_latent_deeponet(cfg, pod_bases=pod_bases if mode == "pod" else None)

    with pytest.raises(ValueError, match="trunk_input"):
        model(torch.randn(10, 1), torch.randn(10, 99))

    with pytest.raises(ValueError, match="trunk_input"):
        model(torch.randn(2, 1), torch.randn(2, 5, 99))


def test_model_honors_configured_dtype():
    cfg = CavityLatentDeepONetConfig(
        reconstruction_mode="autoencoder",
        autoencoder_latent_dims={"u": 2, "v": 2, "w": 2, "p": 2},
        layer_size=8,
        decoder_layer_size=8,
        branch_layers=1,
        trunk_layers=1,
        decoder_layers=1,
        dtype="float64",
    )
    model = build_latent_deeponet(cfg).to(dtype=torch.float64)
    output = model(torch.randn(3, 1, dtype=torch.float64), torch.randn(3, 3, dtype=torch.float64))
    assert output.dtype == torch.float64


def test_checkpoint_round_trip(tmp_path):
    cfg = CavityLatentDeepONetConfig(
        reconstruction_mode="autoencoder",
        autoencoder_latent_dims={"u": 4, "v": 4, "w": 3, "p": 3},
        layer_size=16,
        decoder_layer_size=16,
        branch_layers=1,
        trunk_layers=1,
        decoder_layers=1,
    )
    model = build_latent_deeponet(cfg)
    branch = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    trunk = torch.tensor(
        [[0.0, 0.1, 0.5], [0.2, 0.3, 0.5], [0.4, 0.1, 0.5]], dtype=torch.float32
    )
    target = torch.tensor(
        [
            [1.0, 0.1, 0.0, 0.2],
            [0.9, 0.2, 0.0, 0.3],
            [0.8, 0.3, 0.0, 0.4],
        ],
        dtype=torch.float32,
    )
    branch_norm = TensorNormalizer.fit(branch)
    trunk_norm = TensorNormalizer.fit(trunk)
    target_norm = TensorNormalizer.fit(target)

    checkpoint_path = tmp_path / "latent_deeponet.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        config=cfg,
        branch_normalizer=branch_norm,
        trunk_normalizer=trunk_norm,
        target_normalizer=target_norm,
    )

    reloaded_model = build_latent_deeponet(cfg)
    loaded_cfg, loaded_branch, loaded_trunk, loaded_target = load_checkpoint(
        checkpoint_path,
        reloaded_model,
        device="cpu",
        dtype=torch.float32,
    )
    assert loaded_cfg.reconstruction_mode == cfg.reconstruction_mode
    assert loaded_cfg.autoencoder_latent_dims == cfg.autoencoder_latent_dims
    torch.testing.assert_close(loaded_branch.mean, branch_norm.mean)
    torch.testing.assert_close(loaded_trunk.std, trunk_norm.std)
    torch.testing.assert_close(loaded_target.mean, target_norm.mean)

    with torch.no_grad():
        original_output = model(branch, trunk)
        restored_output = reloaded_model(branch, trunk)
    torch.testing.assert_close(original_output, restored_output)


def test_pod_checkpoint_can_rebuild_model_from_saved_basis(tmp_path):
    cfg = CavityLatentDeepONetConfig(
        reconstruction_mode="pod",
        pod_modes={"u": 2, "v": 2, "w": 1, "p": 2},
        layer_size=8,
        branch_layers=1,
        trunk_layers=1,
    )
    pod_bases = {
        "u": torch.randn(5, 2),
        "v": torch.randn(5, 2),
        "w": torch.randn(5, 1),
        "p": torch.randn(5, 2),
    }
    model = build_latent_deeponet(cfg, pod_bases=pod_bases)
    branch = torch.tensor([[0.1], [0.1], [0.1], [0.1], [0.1]], dtype=torch.float32)
    trunk = torch.randn(5, 3)
    branch_norm = TensorNormalizer.identity(branch)
    trunk_norm = TensorNormalizer.identity(trunk)
    target_norm = TensorNormalizer.identity(torch.randn(5, 4))

    checkpoint_path = tmp_path / "latent_deeponet_pod.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        config=cfg,
        branch_normalizer=branch_norm,
        trunk_normalizer=trunk_norm,
        target_normalizer=target_norm,
    )

    loaded_model, loaded_cfg, _, _, _ = load_model_from_checkpoint(
        checkpoint_path,
        device="cpu",
        dtype=torch.float32,
    )
    assert loaded_cfg.reconstruction_mode == "pod"
    assert loaded_model.num_points == 5
    with torch.no_grad():
        torch.testing.assert_close(model(branch, trunk), loaded_model(branch, trunk))
