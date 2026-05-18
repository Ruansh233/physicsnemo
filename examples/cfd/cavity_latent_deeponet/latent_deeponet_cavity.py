# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Backward-compatible entrypoint for the cavity field-wise latent DeepONet workflow."""

from __future__ import annotations

from examples.cfd.cavity_latent_deeponet.cavity_latent_deeponet import (
    CavityLatentDeepONetConfig,
    FieldwiseLatentDeepONet,
    build_latent_deeponet,
    compute_field_pod_bases,
    load_checkpoint,
    load_config,
    load_model_from_checkpoint,
    main,
    save_checkpoint,
    train_latent_deeponet,
)

__all__ = [
    "CavityLatentDeepONetConfig",
    "FieldwiseLatentDeepONet",
    "build_latent_deeponet",
    "compute_field_pod_bases",
    "load_checkpoint",
    "load_config",
    "load_model_from_checkpoint",
    "main",
    "save_checkpoint",
    "train_latent_deeponet",
]


if __name__ == "__main__":
    main()
