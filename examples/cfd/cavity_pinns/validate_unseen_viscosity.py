# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

from examples.cfd.cavity_pinns.cavity_pinns.config import load_config
from examples.cfd.cavity_pinns.cavity_pinns.workflow import (
    run_unseen_viscosity_workflow,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a trained cavity PINN on disjoint unseen viscosities."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_unseen_viscosity_workflow(config)


if __name__ == "__main__":
    main()
