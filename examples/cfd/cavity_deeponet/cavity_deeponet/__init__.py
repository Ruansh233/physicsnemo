# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .config import CavityCaseMetaData, CavityDeepONetConfig, CavityPhysicalLimits, load_config
from .modeling import TensorNormalizer, build_deeponet, compute_relative_l2, train_deeponet
from .openfoam_data import (
    CavitySampleTensors,
    REQUIRED_OPENFOAM_COMMANDS,
    build_sample_tensors,
    compute_reynolds_number,
    ensure_openfoam_environment,
    extract_case_metadata,
    generate_dataset,
    generate_split_datasets,
    get_foam_case_cls,
    run_case_for_viscosity,
    validate_viscosity_and_reynolds,
)
from .pipeline import main
from .visualization import visualize_predictions

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
    "get_foam_case_cls",
    "load_config",
    "main",
    "run_case_for_viscosity",
    "train_deeponet",
    "validate_viscosity_and_reynolds",
    "visualize_predictions",
]
