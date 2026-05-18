# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sympy import Function, Number, Symbol
from torch import Tensor

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer


class CavityNavierStokesPDE(PDE):
    """Incompressible steady Navier-Stokes PDE for PINN residuals."""

    def __init__(self, spatial_dim: int = 2, nu: str = "nu"):
        if spatial_dim != 2:
            raise NotImplementedError("Current cavity PINN PDE residuals are implemented for 2D only.")
        self.dim = spatial_dim
        x, y = Symbol("x"), Symbol("y")
        u = Function("u")(x, y)
        v = Function("v")(x, y)
        p = Function("p")(x, y)
        nu_field = Function(nu)(x, y) if isinstance(nu, str) else Number(nu)
        self.equations = {
            "continuity": u.diff(x) + v.diff(y),
            "momentum_x": (
                u * u.diff(x)
                + v * u.diff(y)
                + p.diff(x)
                - nu_field * (u.diff(x, 2) + u.diff(y, 2))
            ),
            "momentum_y": (
                u * v.diff(x)
                + v * v.diff(y)
                + p.diff(y)
                - nu_field * (v.diff(x, 2) + v.diff(y, 2))
            ),
        }


def make_physics_informer(*, spatial_dim: int, device: str) -> PhysicsInformer:
    pde = CavityNavierStokesPDE(spatial_dim=spatial_dim, nu="nu")
    return PhysicsInformer(
        required_outputs=["continuity", "momentum_x", "momentum_y"],
        equations=pde,
        grad_method="autodiff",
        device=device,
    )


def compute_physics_residuals(
    *,
    physics_informer: PhysicsInformer,
    coordinates: Tensor,
    prediction: Tensor,
    viscosity: Tensor,
    spatial_dim: int,
) -> dict[str, Tensor]:
    if spatial_dim != 2:
        raise NotImplementedError("Current cavity PINN residual assembly supports spatial_dim=2 only.")
    if prediction.shape[-1] != 3:
        raise ValueError(f"Expected prediction with 3 channels (u,v,p), got {prediction.shape[-1]}.")
    return physics_informer.forward(
        {
            "coordinates": coordinates,
            "u": prediction[:, 0:1],
            "v": prediction[:, 1:2],
            "p": prediction[:, 2:3],
            "nu": viscosity,
        }
    )
