# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import CavityCaseMetaData


def derive_viscosity_bounds(
    metadata: CavityCaseMetaData,
    reynolds_min: float,
    reynolds_max: float,
) -> tuple[float, float]:
    if reynolds_min <= 0.0 or reynolds_max <= 0.0:
        raise ValueError("Reynolds bounds must be positive.")
    if reynolds_min >= reynolds_max:
        raise ValueError("Expected reynolds_min < reynolds_max.")
    if metadata.lid_velocity <= 0.0 or metadata.length_scale <= 0.0:
        raise ValueError("Expected positive cavity metadata.")

    reference = metadata.lid_velocity * metadata.length_scale
    nu_min = reference / reynolds_max
    nu_max = reference / reynolds_min
    if nu_min <= 0.0 or nu_max <= 0.0 or nu_min >= nu_max:
        raise ValueError("Derived invalid viscosity range from metadata and Reynolds bounds.")
    return nu_min, nu_max


def _sample_log_uniform_values(
    rng: np.random.Generator,
    *,
    count: int,
    nu_min: float,
    nu_max: float,
    blocked: tuple[float, ...] = (),
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
        existing = blocked + tuple(values)
        if any(math.isclose(candidate, value, rel_tol=0.0, abs_tol=1.0e-15) for value in existing):
            continue
        values.append(candidate)
    return values


def build_split_manifest(
    *,
    seed: int,
    train_count: int,
    val_count: int,
    test_count: int,
    unseen_count: int,
    nu_min: float,
    nu_max: float,
) -> dict[str, object]:
    if train_count < 0 or val_count < 0 or test_count < 0 or unseen_count < 0:
        raise ValueError("Split counts must be non-negative.")

    rng = np.random.default_rng(seed)
    seen_count = train_count + val_count + test_count
    if seen_count <= 0:
        raise ValueError("Expected at least one seen case across train/validate/test.")

    heldout_count = val_count + test_count
    if heldout_count > 0:
        if train_count < 2:
            raise ValueError(
                "Interpolation split requires train_count >= 2 when validate/test are used."
            )
        if heldout_count > seen_count - 2:
            raise ValueError(
                "Not enough interior seen points to allocate validate/test as interpolation."
            )

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
        sampled_indices = rng.choice(interior_indices, size=heldout_count, replace=False)
        holdout_indices = {int(idx) for idx in np.sort(sampled_indices)}

    train_values: list[float] = []
    heldout_values: list[float] = []
    for idx, value in enumerate(seen_sorted):
        if idx in holdout_indices:
            heldout_values.append(value)
        else:
            train_values.append(value)

    unseen_values = _sample_log_uniform_values(
        rng,
        count=unseen_count,
        nu_min=nu_min,
        nu_max=nu_max,
        blocked=tuple(seen_sorted),
    )
    if len(train_values) != train_count:
        raise RuntimeError("Unexpected train split size after interpolation assignment.")

    validate_values = heldout_values[:val_count]
    test_values = heldout_values[val_count:]

    return {
        "seed": int(seed),
        "nu_min": float(nu_min),
        "nu_max": float(nu_max),
        "train": train_values,
        "validate": validate_values,
        "test": test_values,
        "unseen": unseen_values,
    }


def save_split_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
