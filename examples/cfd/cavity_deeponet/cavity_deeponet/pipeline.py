# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .modeling import (
    TensorNormalizer,
    build_deeponet,
    compute_relative_l2,
    save_checkpoint,
    train_deeponet,
)
from .openfoam_data import generate_split_datasets
from .visualization import visualize_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Cavity DeepONet with foamlib-driven OpenFOAM.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.train_steps <= 0 and cfg.save_visualizations:
        raise ValueError(
            "train_steps must be > 0 when save_visualizations=True. "
            "Otherwise visualized predictions are from an untrained model."
        )

    model = build_deeponet(cfg).to(cfg.device)
    split_tensors, manifest = generate_split_datasets(cfg)
    train_branch, train_trunk, train_target = split_tensors["train"]
    train_branch = train_branch.to(cfg.device)
    train_trunk = train_trunk.to(cfg.device)
    train_target = train_target.to(cfg.device)

    branch_normalizer = (
        TensorNormalizer.fit(train_branch)
        if cfg.normalize_inputs
        else TensorNormalizer.identity(train_branch)
    )
    trunk_normalizer = (
        TensorNormalizer.fit(train_trunk)
        if cfg.normalize_inputs
        else TensorNormalizer.identity(train_trunk)
    )
    target_normalizer = (
        TensorNormalizer.fit(train_target)
        if cfg.normalize_targets
        else TensorNormalizer.identity(train_target)
    )
    model_branch_input = branch_normalizer.transform(train_branch)
    model_trunk_input = trunk_normalizer.transform(train_trunk)
    model_target = target_normalizer.transform(train_target)

    final_loss = train_deeponet(
        model,
        branch_input=model_branch_input,
        trunk_input=model_trunk_input,
        target=model_target,
        lr=cfg.learning_rate,
        steps=cfg.train_steps,
        weight_decay=cfg.weight_decay,
        lbfgs_steps=cfg.lbfgs_steps,
        lbfgs_lr=cfg.lbfgs_lr,
    )
    if cfg.save_trained_model:
        checkpoint_path = Path(cfg.model_checkpoint_path)
        save_checkpoint(
            checkpoint_path,
            model,
            branch_normalizer=branch_normalizer,
            trunk_normalizer=trunk_normalizer,
            target_normalizer=target_normalizer,
        )
        print(f"Saved trained model checkpoint to: {checkpoint_path}")

    model.eval()
    split_metrics: dict[str, tuple[float, tuple[float, ...]]] = {}
    for split_name, tensors in split_tensors.items():
        branch_input, trunk_input, target = tensors
        branch_input = branch_input.to(cfg.device)
        trunk_input = trunk_input.to(cfg.device)
        target = target.to(cfg.device)
        with torch.no_grad():
            normalized_prediction = model(
                branch_normalizer.transform(branch_input),
                trunk_normalizer.transform(trunk_input),
            )
            prediction = target_normalizer.inverse(normalized_prediction)
        split_metrics[split_name] = compute_relative_l2(prediction, target)
        print(f"{split_name} branch tensor shape: {tuple(branch_input.shape)}")
        print(f"{split_name} trunk tensor shape:  {tuple(trunk_input.shape)}")
        print(f"{split_name} target tensor shape: {tuple(target.shape)}")

    train_rel_l2, train_channel_rel_l2 = split_metrics["train"]
    validate_rel_l2, validate_channel_rel_l2 = split_metrics["validate"]
    test_rel_l2, test_channel_rel_l2 = split_metrics["test"]

    print(
        "Split counts: "
        f"train={len(manifest['train'])}, "
        f"validate={len(manifest['validate'])}, "
        f"test={len(manifest['test'])}"
    )
    print(f"Output channel names: {model.output_channel_names}")
    if cfg.train_steps > 0 or cfg.lbfgs_steps > 0:
        print(
            "Final normalized training loss after "
            f"{cfg.train_steps} AdamW steps and {cfg.lbfgs_steps} LBFGS steps: {final_loss:.6e}"
        )
        print(f"Train physical aggregate rel-L2: {train_rel_l2:.6e}")
        print(f"Validate physical aggregate rel-L2: {validate_rel_l2:.6e}")
        print(f"Test physical aggregate rel-L2: {test_rel_l2:.6e}")
        for channel_name, train_error, validate_error, test_error in zip(
            model.output_channel_names,
            train_channel_rel_l2,
            validate_channel_rel_l2,
            test_channel_rel_l2,
        ):
            print(
                f"Physical rel-L2[{channel_name}]: "
                f"train={train_error:.6e}, "
                f"validate={validate_error:.6e}, "
                f"test={test_error:.6e}"
            )
    if cfg.save_visualizations:
        vis_branch, vis_trunk, vis_target = split_tensors["test"]
        visualize_predictions(
            model=model,
            branch_input=vis_branch.to(cfg.device),
            trunk_input=vis_trunk.to(cfg.device),
            target=vis_target.to(cfg.device),
            output_dir=Path(cfg.visualization_dir),
            max_cases=cfg.visualization_max_cases,
            branch_normalizer=branch_normalizer,
            trunk_normalizer=trunk_normalizer,
            target_normalizer=target_normalizer,
        )
        print(f"Saved visualization figures to: {cfg.visualization_dir}")
