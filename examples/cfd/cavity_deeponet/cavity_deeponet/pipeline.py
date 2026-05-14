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
    train_deeponet,
)
from .openfoam_data import generate_dataset
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
    branch_input, trunk_input, target = generate_dataset(cfg)
    branch_input = branch_input.to(cfg.device)
    trunk_input = trunk_input.to(cfg.device)
    target = target.to(cfg.device)

    branch_normalizer = (
        TensorNormalizer.fit(branch_input)
        if cfg.normalize_inputs
        else TensorNormalizer.identity(branch_input)
    )
    trunk_normalizer = (
        TensorNormalizer.fit(trunk_input)
        if cfg.normalize_inputs
        else TensorNormalizer.identity(trunk_input)
    )
    target_normalizer = (
        TensorNormalizer.fit(target)
        if cfg.normalize_targets
        else TensorNormalizer.identity(target)
    )
    model_branch_input = branch_normalizer.transform(branch_input)
    model_trunk_input = trunk_normalizer.transform(trunk_input)
    model_target = target_normalizer.transform(target)

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

    model.eval()
    with torch.no_grad():
        normalized_prediction = model(model_branch_input, model_trunk_input)
        prediction = target_normalizer.inverse(normalized_prediction)
    relative_l2, channel_relative_l2 = compute_relative_l2(prediction, target)

    print(f"Branch tensor shape: {tuple(branch_input.shape)}")
    print(f"Trunk tensor shape:  {tuple(trunk_input.shape)}")
    print(f"Target tensor shape: {tuple(target.shape)}")
    print(f"Output channel names: {model.output_channel_names}")
    if cfg.train_steps > 0 or cfg.lbfgs_steps > 0:
        print(
            "Final normalized training loss after "
            f"{cfg.train_steps} AdamW steps and {cfg.lbfgs_steps} LBFGS steps: {final_loss:.6e}"
        )
        print(f"Physical aggregate rel-L2: {relative_l2:.6e}")
        for channel_name, channel_error in zip(model.output_channel_names, channel_relative_l2):
            print(f"Physical rel-L2[{channel_name}]: {channel_error:.6e}")
    if cfg.save_visualizations:
        visualize_predictions(
            model=model,
            branch_input=branch_input,
            trunk_input=trunk_input,
            target=target,
            output_dir=Path(cfg.visualization_dir),
            max_cases=cfg.visualization_max_cases,
            branch_normalizer=branch_normalizer,
            trunk_normalizer=trunk_normalizer,
            target_normalizer=target_normalizer,
        )
        print(f"Saved visualization figures to: {cfg.visualization_dir}")
