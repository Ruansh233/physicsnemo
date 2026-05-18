# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _sample_log_uniform_values(
    rng: np.random.Generator,
    *,
    count: int,
    nu_min: float,
    nu_max: float,
) -> list[float]:
    if count <= 0:
        return []
    if nu_min <= 0.0 or nu_max <= 0.0:
        raise ValueError("Viscosity bounds must be positive for log-uniform sampling.")
    if nu_min >= nu_max:
        raise ValueError("Expected nu_min < nu_max.")

    values: list[float] = []
    log_min = math.log(nu_min)
    log_max = math.log(nu_max)
    while len(values) < count:
        candidate = float(math.exp(rng.uniform(log_min, log_max)))
        if any(
            math.isclose(candidate, value, rel_tol=0.0, abs_tol=1.0e-15)
            for value in values
        ):
            continue
        values.append(candidate)
    return values


def build_split_manifest(
    *,
    seed: int,
    train_count: int,
    validate_count: int,
    test_count: int,
    nu_min: float,
    nu_max: float,
) -> dict[str, object]:
    if train_count < 0 or validate_count < 0 or test_count < 0:
        raise ValueError("Split counts must be non-negative.")

    seen_count = train_count + validate_count + test_count
    if seen_count <= 0:
        raise ValueError("Expected at least one case across train/validate/test.")

    heldout_count = validate_count + test_count
    if heldout_count > 0:
        if train_count < 2:
            raise ValueError(
                "Interpolation split requires train_count >= 2 when validate/test are used."
            )
        if heldout_count > seen_count - 2:
            raise ValueError(
                "Not enough interior seen points to allocate validate/test as interpolation."
            )

    rng = np.random.default_rng(seed)
    seen_values = _sample_log_uniform_values(
        rng,
        count=seen_count,
        nu_min=nu_min,
        nu_max=nu_max,
    )
    seen_sorted = sorted(seen_values)

    holdout_indices: set[int] = set()
    if heldout_count > 0:
        interior_indices = np.arange(1, seen_count - 1)
        sampled_indices = rng.choice(
            interior_indices, size=heldout_count, replace=False
        )
        holdout_indices = {int(idx) for idx in np.sort(sampled_indices)}

    train_values: list[float] = []
    heldout_values: list[float] = []
    for idx, value in enumerate(seen_sorted):
        if idx in holdout_indices:
            heldout_values.append(value)
        else:
            train_values.append(value)

    if len(train_values) != train_count:
        raise RuntimeError(
            "Unexpected train split size after interpolation assignment."
        )

    validate_values = heldout_values[:validate_count]
    test_values = heldout_values[validate_count:]

    return {
        "seed": int(seed),
        "nu_min": float(nu_min),
        "nu_max": float(nu_max),
        "train": train_values,
        "validate": validate_values,
        "test": test_values,
    }


def save_split_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
