# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from physicsnemo.experimental.models.deeponet import DeepONet

from .modeling import TensorNormalizer


def visualize_predictions(
    model: DeepONet,
    branch_input: Tensor,
    trunk_input: Tensor,
    target: Tensor,
    output_dir: Path,
    max_cases: int = 3,
    branch_normalizer: TensorNormalizer | None = None,
    trunk_normalizer: TensorNormalizer | None = None,
    target_normalizer: TensorNormalizer | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install with "
            "`pip install matplotlib` or add it to your example environment."
        ) from exc

    if max_cases <= 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    branch_model_input = (
        branch_normalizer.transform(branch_input)
        if branch_normalizer is not None
        else branch_input
    )
    trunk_model_input = (
        trunk_normalizer.transform(trunk_input) if trunk_normalizer is not None else trunk_input
    )
    model.eval()
    with torch.no_grad():
        pred = model(branch_model_input, trunk_model_input)
        if target_normalizer is not None:
            pred = target_normalizer.inverse(pred)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    trunk_np = trunk_input.detach().cpu().numpy()
    branch_np = branch_input.detach().cpu().numpy().reshape(-1)

    unique_nu = np.unique(branch_np)
    channel_names = model.output_channel_names
    for case_idx, nu in enumerate(unique_nu[:max_cases]):
        mask = np.isclose(branch_np, nu)
        x = trunk_np[mask, 0]
        y = trunk_np[mask, 1]
        tri = mtri.Triangulation(x, y)

        for channel_idx, channel_name in enumerate(channel_names):
            truth_field = target_np[mask, channel_idx]
            pred_field = pred_np[mask, channel_idx]
            diff_field = np.abs(pred_field - truth_field)

            fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
            vmin = float(np.min(truth_field))
            vmax = float(np.max(truth_field))

            true_plot = ax[0].tricontourf(tri, truth_field, levels=50, vmin=vmin, vmax=vmax)
            pred_plot = ax[1].tricontourf(tri, pred_field, levels=50, vmin=vmin, vmax=vmax)
            diff_plot = ax[2].tricontourf(tri, diff_field, levels=50)
            fig.colorbar(true_plot, ax=ax[0])
            fig.colorbar(pred_plot, ax=ax[1])
            fig.colorbar(diff_plot, ax=ax[2])

            ax[0].set_title("True")
            ax[1].set_title("Pred")
            ax[2].set_title("Difference")
            for axis in ax:
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.set_aspect("equal", adjustable="box")

            fig.suptitle(f"nu={nu:.6g}, field={channel_name}")
            out_file = output_dir / f"case_{case_idx:03d}_nu_{nu:.6g}_{channel_name}.png"
            fig.savefig(out_file, dpi=200)
            plt.close(fig)
