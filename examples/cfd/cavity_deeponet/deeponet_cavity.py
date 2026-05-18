# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Backward-compatible entrypoint for the cavity DeepONet workflow."""

from __future__ import annotations

from examples.cfd.cavity_deeponet.cavity_deeponet import (
    REQUIRED_OPENFOAM_COMMANDS,
    CavityCaseMetaData,
    CavityDeepONetConfig,
    CavityPhysicalLimits,
    CavitySampleTensors,
    TensorNormalizer,
    build_deeponet,
    build_sample_tensors,
    compute_relative_l2,
    compute_reynolds_number,
    ensure_openfoam_environment,
    extract_case_metadata,
    generate_dataset,
    generate_split_datasets,
    run_case_for_viscosity,
    train_deeponet,
    validate_viscosity_and_reynolds,
    visualize_predictions,
)
from examples.cfd.cavity_deeponet.cavity_deeponet import (
    get_foam_case_cls as _get_foam_case_cls,
)
from examples.cfd.cavity_deeponet.cavity_deeponet import (
    load_config as _load_config,
)
from examples.cfd.cavity_deeponet.cavity_deeponet.pipeline import main

__all__ = [
    "CavityCaseMetaData",
    "CavityDeepONetConfig",
    "CavityPhysicalLimits",
    "CavitySampleTensors",
    "REQUIRED_OPENFOAM_COMMANDS",
    "TensorNormalizer",
    "build_deeponet",
    "build_sample_tensors",
    "compute_relative_l2",
    "compute_reynolds_number",
    "ensure_openfoam_environment",
    "extract_case_metadata",
    "generate_dataset",
    "generate_split_datasets",
    "run_case_for_viscosity",
    "train_deeponet",
    "validate_viscosity_and_reynolds",
    "visualize_predictions",
    "_get_foam_case_cls",
    "_load_config",
    "main",
]

if __name__ == "__main__":
    main()
