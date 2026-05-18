# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .config import CavityCaseMetaData, CavityPINNConfig, load_config
from .openfoam_data import (
    REQUIRED_OPENFOAM_COMMANDS,
    CavityCaseTensors,
    build_case_tensors,
    compute_reynolds_number,
    ensure_openfoam_environment,
    extract_case_metadata,
    generate_cases_for_viscosities,
    get_foam_case_cls,
    run_case_for_viscosity,
)
from .physics import (
    CavityNavierStokesPDE,
    compute_physics_residuals,
    make_physics_informer,
)
from .splits import build_split_manifest, derive_viscosity_bounds, save_split_manifest
from .visualization import save_split_triptych_visualizations

__all__ = [
    "CavityCaseMetaData",
    "CavityCaseTensors",
    "CavityNavierStokesPDE",
    "CavityPINNConfig",
    "REQUIRED_OPENFOAM_COMMANDS",
    "build_case_tensors",
    "build_split_manifest",
    "compute_physics_residuals",
    "compute_reynolds_number",
    "derive_viscosity_bounds",
    "ensure_openfoam_environment",
    "extract_case_metadata",
    "generate_cases_for_viscosities",
    "get_foam_case_cls",
    "load_config",
    "make_physics_informer",
    "run_case_for_viscosity",
    "save_split_triptych_visualizations",
    "save_split_manifest",
]
