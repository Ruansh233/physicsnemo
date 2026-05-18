# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from examples.cfd.cavity_pinns.cavity_pinns.config import CavityPINNConfig, load_config
from examples.cfd.cavity_pinns.cavity_pinns.workflow import (
    _evaluate_split,
    _load_split_cases,
    _prepare_split_manifest,
    _train_model,
)

OBJECTIVE_TO_METRIC = {
    "validate_supervised_mse": "supervised_mse",
    "validate_relative_l2": "relative_l2",
    "validate_physics_residual_mse": "physics_residual_mse",
}


def _parse_int_list(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one string value.")
    return values


def _sample_trials(
    full_grid: Sequence[tuple[int, int, str, float, int]],
    *,
    max_trials: int,
    seed: int,
) -> list[tuple[int, int, str, float, int]]:
    if max_trials <= 0 or max_trials >= len(full_grid):
        return list(full_grid)
    rng = random.Random(seed)
    trial_indices = sorted(rng.sample(range(len(full_grid)), k=max_trials))
    return [full_grid[idx] for idx in trial_indices]


def _write_best_config(path: Path, config: CavityPINNConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(config)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Architecture search utility for cavity PINN. "
            "Runs multiple trials and ranks them by validation metrics."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Base YAML config path."
    )
    parser.add_argument(
        "--layers", default="6,8,10,12", help="Comma-separated hidden-layer counts."
    )
    parser.add_argument(
        "--widths", default="256,320,384,512", help="Comma-separated hidden sizes."
    )
    parser.add_argument(
        "--activations",
        default="gelu,silu,tanh",
        help="Comma-separated activation names (relu, gelu, silu, tanh).",
    )
    parser.add_argument(
        "--learning-rates",
        default="0.002,0.001",
        help="Comma-separated learning rates.",
    )
    parser.add_argument(
        "--steps",
        default="1000,1500",
        help="Comma-separated train_steps values.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Max number of sampled trials. 0 means run full grid.",
    )
    parser.add_argument(
        "--trial-seed",
        type=int,
        default=123,
        help="Seed used only for random trial subsampling when --max-trials > 0.",
    )
    parser.add_argument(
        "--objective",
        choices=tuple(OBJECTIVE_TO_METRIC.keys()),
        default="validate_supervised_mse",
        help="Validation objective to minimize.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("examples/cfd/cavity_pinns/outputs/architecture_search"),
        help="Directory for search summary files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top trials to print.",
    )
    parser.add_argument(
        "--evaluate-test-all",
        action="store_true",
        help="Also evaluate test split for every trial (slower).",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    if base_config.spatial_dim != 2:
        raise NotImplementedError(
            "Architecture tuning currently supports spatial_dim=2 only."
        )

    layers = _parse_int_list(args.layers)
    widths = _parse_int_list(args.widths)
    activations = _parse_str_list(args.activations)
    learning_rates = _parse_float_list(args.learning_rates)
    steps_values = _parse_int_list(args.steps)
    full_grid = list(
        itertools.product(layers, widths, activations, learning_rates, steps_values)
    )
    trial_grid = _sample_trials(
        full_grid, max_trials=args.max_trials, seed=args.trial_seed
    )
    if not trial_grid:
        raise ValueError("No architecture trials to run.")

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing dataset once and reusing it across {len(trial_grid)} trials.")
    manifest = _prepare_split_manifest(base_config)
    train_cases = _load_split_cases(base_config, manifest["train"], "train")
    validate_cases = _load_split_cases(base_config, manifest["validate"], "validate")
    test_cases = _load_split_cases(base_config, manifest["test"], "test")

    objective_metric = OBJECTIVE_TO_METRIC[args.objective]
    trial_results: list[dict[str, float | int | str]] = []
    for trial_idx, (layer_count, width, activation, lr, steps) in enumerate(trial_grid):
        trial_config = replace(
            base_config,
            model_layers=layer_count,
            model_layer_size=width,
            activation_fn=activation,
            learning_rate=lr,
            train_steps=steps,
            save_visualizations=False,
        )
        print(
            f"[trial {trial_idx + 1:03d}/{len(trial_grid):03d}] "
            f"layers={layer_count}, width={width}, activation={activation}, lr={lr:.3e}, steps={steps}"
        )

        model, physics_informer, normalizer = _train_model(
            config=trial_config,
            train_cases=train_cases,
        )
        train_metrics = _evaluate_split(
            model=model,
            physics_informer=physics_informer,
            normalizer=normalizer,
            cases=train_cases,
            config=trial_config,
            split_name=f"train_trial_{trial_idx:03d}",
        )
        validate_metrics = _evaluate_split(
            model=model,
            physics_informer=physics_informer,
            normalizer=normalizer,
            cases=validate_cases,
            config=trial_config,
            split_name=f"validate_trial_{trial_idx:03d}",
        )

        trial_record: dict[str, float | int | str] = {
            "trial_index": trial_idx,
            "model_layers": layer_count,
            "model_layer_size": width,
            "activation_fn": activation,
            "learning_rate": lr,
            "train_steps": steps,
            "train_supervised_mse": train_metrics["supervised_mse"],
            "train_relative_l2": train_metrics["relative_l2"],
            "validate_supervised_mse": validate_metrics["supervised_mse"],
            "validate_relative_l2": validate_metrics["relative_l2"],
            "validate_physics_residual_mse": validate_metrics["physics_residual_mse"],
            "objective": validate_metrics[objective_metric],
        }

        if args.evaluate_test_all:
            test_metrics = _evaluate_split(
                model=model,
                physics_informer=physics_informer,
                normalizer=normalizer,
                cases=test_cases,
                config=trial_config,
                split_name=f"test_trial_{trial_idx:03d}",
            )
            trial_record["test_supervised_mse"] = test_metrics["supervised_mse"]
            trial_record["test_relative_l2"] = test_metrics["relative_l2"]
            trial_record["test_physics_residual_mse"] = test_metrics[
                "physics_residual_mse"
            ]

        trial_results.append(trial_record)
        print(f"trial_objective={trial_record['objective']:.6e}")

    ranked_results = sorted(trial_results, key=lambda row: float(row["objective"]))
    top_k = max(1, min(args.top_k, len(ranked_results)))
    print(f"Top {top_k} trials by {args.objective}:")
    for rank, row in enumerate(ranked_results[:top_k], start=1):
        print(
            f"rank={rank} trial={row['trial_index']} objective={float(row['objective']):.6e} "
            f"layers={row['model_layers']} width={row['model_layer_size']} "
            f"activation={row['activation_fn']} lr={float(row['learning_rate']):.3e} "
            f"steps={row['train_steps']}"
        )

    best = ranked_results[0]
    best_config = replace(
        base_config,
        model_layers=int(best["model_layers"]),
        model_layer_size=int(best["model_layer_size"]),
        activation_fn=str(best["activation_fn"]),
        learning_rate=float(best["learning_rate"]),
        train_steps=int(best["train_steps"]),
    )
    best_config_path = results_dir / "best_architecture_config.yaml"
    _write_best_config(best_config_path, best_config)

    summary = {
        "objective": args.objective,
        "trial_count": len(ranked_results),
        "search_space": {
            "layers": layers,
            "widths": widths,
            "activations": activations,
            "learning_rates": learning_rates,
            "steps": steps_values,
            "max_trials": args.max_trials,
            "trial_seed": args.trial_seed,
        },
        "manifest": manifest,
        "best_trial": best,
        "best_config_path": str(best_config_path),
        "ranked_trials": ranked_results,
    }
    summary_path = results_dir / "architecture_search_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved search summary to: {summary_path}")
    print(f"Saved best config to: {best_config_path}")
    print(
        "Run full training with the best config:\n"
        f"uv run python examples/cfd/cavity_pinns/train.py --config {best_config_path}"
    )


if __name__ == "__main__":
    main()
