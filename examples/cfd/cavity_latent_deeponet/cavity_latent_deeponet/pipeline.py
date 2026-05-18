# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from examples.cfd.cavity_deeponet.cavity_deeponet.modeling import (
    TensorNormalizer,
    compute_relative_l2,
)
from examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data import (
    generate_split_datasets,
)
from examples.cfd.cavity_deeponet.cavity_deeponet.visualization import (
    visualize_predictions,
)

from .config import load_config
from .modeling import (
    build_latent_deeponet,
    compute_field_pod_bases,
    save_checkpoint,
    train_latent_deeponet,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cavity Field-wise Latent DeepONet with foamlib-driven OpenFOAM."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.train_steps <= 0 and cfg.save_visualizations:
        raise ValueError(
            "train_steps must be > 0 when save_visualizations=True. "
            "Otherwise visualized predictions are from an untrained model."
        )

    dtype = getattr(torch, cfg.dtype)
    split_tensors, manifest = generate_split_datasets(cfg)
    train_branch, train_trunk, train_target = split_tensors["train"]
    train_branch = train_branch.to(device=cfg.device, dtype=dtype)
    train_trunk = train_trunk.to(device=cfg.device, dtype=dtype)
    train_target = train_target.to(device=cfg.device, dtype=dtype)

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
    pod_bases = None
    if cfg.reconstruction_mode.lower() == "pod":
        pod_bases = compute_field_pod_bases(
            model_target,
            case_count=cfg.train_case_count,
            field_order=cfg.field_order,
            pod_modes=cfg.pod_modes,
        )
    model = build_latent_deeponet(cfg, pod_bases=pod_bases).to(
        device=cfg.device, dtype=dtype
    )

    final_loss = train_latent_deeponet(
        model,
        branch_input=model_branch_input,
        trunk_input=model_trunk_input,
        target=model_target,
        lr=cfg.learning_rate,
        steps=cfg.train_steps,
        weight_decay=cfg.weight_decay,
        log_steps=cfg.log_steps,
        enable_realtime_loss_plot=cfg.enable_realtime_loss_plot,
        lbfgs_steps=cfg.lbfgs_steps,
        lbfgs_lr=cfg.lbfgs_lr,
    )

    if cfg.save_trained_model:
        checkpoint_path = Path(cfg.model_checkpoint_path)
        save_checkpoint(
            checkpoint_path,
            model,
            config=cfg,
            branch_normalizer=branch_normalizer,
            trunk_normalizer=trunk_normalizer,
            target_normalizer=target_normalizer,
        )
        print(f"Saved trained model checkpoint to: {checkpoint_path}")

    model.eval()
    split_metrics: dict[str, tuple[float, tuple[float, ...]]] = {}
    for split_name, tensors in split_tensors.items():
        branch_input, trunk_input, target = tensors
        branch_input = branch_input.to(device=cfg.device, dtype=dtype)
        trunk_input = trunk_input.to(device=cfg.device, dtype=dtype)
        target = target.to(device=cfg.device, dtype=dtype)
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
        for split_name, (aggregate, channel_values) in split_metrics.items():
            print(f"{split_name} physical aggregate rel-L2: {aggregate:.6e}")
            for channel_name, channel_error in zip(
                model.output_channel_names, channel_values
            ):
                print(
                    f"Physical rel-L2[{channel_name}] on {split_name}: {channel_error:.6e}"
                )

    if cfg.save_visualizations:
        vis_branch, vis_trunk, vis_target = split_tensors["test"]
        visualize_predictions(
            model=model,
            branch_input=vis_branch.to(device=cfg.device, dtype=dtype),
            trunk_input=vis_trunk.to(device=cfg.device, dtype=dtype),
            target=vis_target.to(device=cfg.device, dtype=dtype),
            output_dir=Path(cfg.visualization_dir),
            max_cases=cfg.visualization_max_cases,
            branch_normalizer=branch_normalizer,
            trunk_normalizer=trunk_normalizer,
            target_normalizer=target_normalizer,
        )
        print(f"Saved visualization figures to: {cfg.visualization_dir}")
