# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from omegaconf import OmegaConf

from examples.cfd.cavity_deeponet.cavity_deeponet.config import CavityDeepONetConfig
from examples.cfd.cavity_deeponet.cavity_deeponet.modeling import (
    TensorNormalizer,
    build_deeponet,
    compute_relative_l2,
    train_deeponet,
)
from examples.cfd.cavity_deeponet.cavity_deeponet.openfoam_data import generate_dataset
from examples.cfd.cavity_deeponet.cavity_deeponet.visualization import (
    visualize_predictions,
)


def _load_workflow_config(
    config_path: Path | None,
) -> tuple[CavityDeepONetConfig, tuple[float, ...], tuple[float, ...]]:
    base_cfg = OmegaConf.structured(CavityDeepONetConfig())
    user_cfg = (
        OmegaConf.load(config_path) if config_path is not None else OmegaConf.create({})
    )

    train_values = tuple(float(v) for v in user_cfg.pop("train_viscosity_values", []))
    test_values = tuple(float(v) for v in user_cfg.pop("test_viscosity_values", []))
    if not train_values:
        train_values = tuple(float(v) for v in base_cfg.viscosity_values)
    if not test_values:
        raise ValueError(
            "Please provide non-empty `test_viscosity_values` in the workflow config."
        )
    overlap = set(train_values).intersection(test_values)
    if overlap:
        raise ValueError(
            "Unseen-viscosity test requires disjoint train/test viscosity sets. "
            f"Found overlap: {sorted(overlap)}"
        )

    merged_cfg = OmegaConf.merge(base_cfg, user_cfg)
    cfg = CavityDeepONetConfig(**OmegaConf.to_container(merged_cfg, resolve=True))
    return cfg, train_values, test_values


def _fit_normalizers(
    cfg: CavityDeepONetConfig,
    branch: torch.Tensor,
    trunk: torch.Tensor,
    target: torch.Tensor,
):
    branch_normalizer = (
        TensorNormalizer.fit(branch)
        if cfg.normalize_inputs
        else TensorNormalizer.identity(branch)
    )
    trunk_normalizer = (
        TensorNormalizer.fit(trunk)
        if cfg.normalize_inputs
        else TensorNormalizer.identity(trunk)
    )
    target_normalizer = (
        TensorNormalizer.fit(target)
        if cfg.normalize_targets
        else TensorNormalizer.identity(target)
    )
    return branch_normalizer, trunk_normalizer, target_normalizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepONet cavity unseen-viscosity workflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to unseen-viscosity YAML config.",
    )
    args = parser.parse_args()

    cfg, train_viscosities, test_viscosities = _load_workflow_config(args.config)

    train_cfg = replace(
        cfg,
        viscosity_values=train_viscosities,
        run_root=str(Path(cfg.run_root).parent / (Path(cfg.run_root).name + "_train")),
    )
    test_cfg = replace(
        cfg,
        viscosity_values=test_viscosities,
        run_root=str(
            Path(cfg.run_root).parent / (Path(cfg.run_root).name + "_unseen_test")
        ),
    )

    model = build_deeponet(cfg).to(cfg.device)
    train_branch, train_trunk, train_target = generate_dataset(train_cfg)
    test_branch, test_trunk, test_target = generate_dataset(test_cfg)

    train_branch = train_branch.to(cfg.device)
    train_trunk = train_trunk.to(cfg.device)
    train_target = train_target.to(cfg.device)
    test_branch = test_branch.to(cfg.device)
    test_trunk = test_trunk.to(cfg.device)
    test_target = test_target.to(cfg.device)

    branch_norm, trunk_norm, target_norm = _fit_normalizers(
        cfg, train_branch, train_trunk, train_target
    )

    train_branch_model = branch_norm.transform(train_branch)
    train_trunk_model = trunk_norm.transform(train_trunk)
    train_target_model = target_norm.transform(train_target)
    test_branch_model = branch_norm.transform(test_branch)
    test_trunk_model = trunk_norm.transform(test_trunk)

    final_loss = train_deeponet(
        model,
        branch_input=train_branch_model,
        trunk_input=train_trunk_model,
        target=train_target_model,
        lr=cfg.learning_rate,
        steps=cfg.train_steps,
        weight_decay=cfg.weight_decay,
        lbfgs_steps=cfg.lbfgs_steps,
        lbfgs_lr=cfg.lbfgs_lr,
    )

    model.eval()
    with torch.no_grad():
        train_pred = target_norm.inverse(model(train_branch_model, train_trunk_model))
        test_pred = target_norm.inverse(model(test_branch_model, test_trunk_model))

    train_rel_l2, train_channels = compute_relative_l2(train_pred, train_target)
    test_rel_l2, test_channels = compute_relative_l2(test_pred, test_target)

    print(f"Train viscosities: {train_viscosities}")
    print(f"Unseen test viscosities: {test_viscosities}")
    print(f"Final normalized training loss: {final_loss:.6e}")
    print(f"Train physical aggregate rel-L2: {train_rel_l2:.6e}")
    print(f"Unseen-test physical aggregate rel-L2: {test_rel_l2:.6e}")
    for channel_name, tr, te in zip(
        model.output_channel_names, train_channels, test_channels
    ):
        print(f"Channel {channel_name}: train={tr:.6e}, unseen_test={te:.6e}")

    if cfg.save_visualizations:
        vis_dir = Path(cfg.visualization_dir) / "unseen_viscosity"
        visualize_predictions(
            model=model,
            branch_input=test_branch,
            trunk_input=test_trunk,
            target=test_target,
            output_dir=vis_dir,
            max_cases=cfg.visualization_max_cases,
            branch_normalizer=branch_norm,
            trunk_normalizer=trunk_norm,
            target_normalizer=target_norm,
        )
        print(f"Saved unseen-viscosity visualizations to: {vis_dir}")


if __name__ == "__main__":
    main()
