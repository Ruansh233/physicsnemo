# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .config import CavityPINNConfig
from .modeling import predict_fields
from .openfoam_data import CavityCaseTensors


def _channel_names(config: CavityPINNConfig) -> tuple[str, ...]:
    return (*config.velocity_names, config.pressure_name)


def save_split_triptych_visualizations(
    *,
    model: nn.Module,
    cases: list[CavityCaseTensors],
    config: CavityPINNConfig,
    split_name: str,
    predict_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
) -> None:
    if not config.save_visualizations:
        return
    if config.spatial_dim != 2:
        raise NotImplementedError(
            "Visualization currently supports spatial_dim=2 only."
        )
    if config.visualization_max_cases <= 0 or not cases:
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required when save_visualizations=true. "
            "Install with `pip install -r examples/cfd/cavity_pinns/requirements.txt`."
        ) from exc

    output_dir = Path(config.visualization_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = _channel_names(config)
    field_predictor = predict_fn or predict_fields

    model.eval()
    with torch.no_grad():
        for case_idx, case in enumerate(cases[: config.visualization_max_cases]):
            coords = case.coordinates.to(config.device)
            visc = case.viscosity.to(config.device)
            target = case.target.cpu().numpy()
            prediction = field_predictor(model, coords, visc).cpu().numpy()
            coord_np = case.coordinates.cpu().numpy()
            tri = mtri.Triangulation(coord_np[:, 0], coord_np[:, 1])
            nu = float(case.viscosity[0, 0].item())

            for channel_idx, channel_name in enumerate(channels):
                truth_field = target[:, channel_idx]
                pred_field = prediction[:, channel_idx]
                error_field = np.abs(pred_field - truth_field)
                shared_vmin = float(np.min(np.concatenate((truth_field, pred_field))))
                shared_vmax = float(np.max(np.concatenate((truth_field, pred_field))))

                fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
                true_plot = ax[0].tricontourf(
                    tri,
                    truth_field,
                    levels=50,
                    vmin=shared_vmin,
                    vmax=shared_vmax,
                )
                pred_plot = ax[1].tricontourf(
                    tri,
                    pred_field,
                    levels=50,
                    vmin=shared_vmin,
                    vmax=shared_vmax,
                )
                err_plot = ax[2].tricontourf(tri, error_field, levels=50)
                fig.colorbar(true_plot, ax=ax[0])
                fig.colorbar(pred_plot, ax=ax[1])
                fig.colorbar(err_plot, ax=ax[2])

                ax[0].set_title("True")
                ax[1].set_title("Predicted")
                ax[2].set_title("Error")
                for axis in ax:
                    axis.set_xlabel(config.coordinate_names[0])
                    axis.set_ylabel(config.coordinate_names[1])
                    axis.set_aspect("equal", adjustable="box")

                fig.suptitle(
                    f"{split_name} | case={case_idx:03d} | nu={nu:.6g} | {channel_name}"
                )
                fig.savefig(
                    output_dir / f"case_{case_idx:03d}_nu_{nu:.6g}_{channel_name}.png",
                    dpi=200,
                )
                plt.close(fig)
