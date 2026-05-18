# Cavity Hybrid PINN Example

This example trains a hybrid PINN surrogate for the steady lid-driven cavity.
It uses OpenFOAM snapshots as supervised targets and augments training with:

- boundary-condition loss (moving lid + no-slip walls)
- incompressible Navier-Stokes residual loss through `PhysicsInformer`

The training loop normalizes the model inputs and field outputs internally, then
converts predictions back to physical units for metrics, boundary conditions,
visualizations, and Navier-Stokes residuals. Boundary-condition samples are
placed on the physical cavity walls rather than at the nearest cell centers.

The current implementation targets the 2D cavity case and keeps interfaces
dimension-aware to simplify future extension to 3D.

## Prerequisites

1. Activate your Python environment (recommended workflow in this repo is `uv`).
2. Install example-local dependency:

```bash
uv pip install -r examples/cfd/cavity_pinns/requirements.txt
```

3. Activate OpenFOAM in your shell before running:

```bash
of2312
```

The runner checks `blockMesh`, `icoFoam`, and `postProcess` availability and
fails fast if OpenFOAM is not active.

## Run Standard Train/Validate/Test Workflow

```bash
uv run python examples/cfd/cavity_pinns/train.py --config examples/cfd/cavity_pinns/cavity_pinns.yaml
```

This run:

- samples viscosity values log-uniformly such that derived Reynolds numbers are in `[10, 100]`
- splits whole viscosity cases into `train`, `validate`, and `test`
- enforces interpolation by assigning `validate` and `test` viscosities strictly inside the sampled training viscosity range
- persists the split manifest under `run_root`

## Run Unseen-Viscosity Validation Workflow

```bash
uv run python examples/cfd/cavity_pinns/validate_unseen_viscosity.py --config examples/cfd/cavity_pinns/cavity_pinns_unseen_viscosity.yaml
```

This run trains on sampled in-range training viscosities and evaluates on
disjoint sampled unseen viscosities.

## Tune Architecture For Better Accuracy

Use the architecture search helper to sweep network depth/width/activation and
basic training knobs while reusing the same sampled dataset split:

```bash
uv run python examples/cfd/cavity_pinns/tune_architecture.py \
  --config examples/cfd/cavity_pinns/cavity_pinns.yaml \
  --layers 4,6,8 \
  --widths 128,256,320 \
  --activations tanh,silu,gelu \
  --learning-rates 0.001,0.0005 \
  --steps 5000,10000 \
  --objective validate_relative_l2
```

Outputs are written under `examples/cfd/cavity_pinns/outputs/architecture_search/`:

- `architecture_search_results.json` with ranked trial metrics
- `best_architecture_config.yaml` for full retraining

## Configuration Notes

Main configuration fields:

- OpenFOAM/case settings: `case_path`, `run_root`, `run_openfoam`
- split setup: `viscosity_seed`, `train_case_count`, `validate_case_count`, `test_case_count`, `unseen_case_count`
- physics constraints: `reynolds_min`, `reynolds_max`
- model/training: `model_layers`, `model_layer_size`, `activation_fn`, `train_steps`
- hybrid loss setup: `supervised_loss_weight`, `boundary_loss_weight`,
  `boundary_loss_ramp_steps`, `physics_loss_weight`, `physics_loss_ramp_steps`,
  `boundary_points_per_wall`
- visualization: `save_visualizations`, `visualization_max_cases`, `visualization_dir`

The default configuration uses a `tanh` MLP and ramps the boundary and physics
losses in after the supervised signal has started fitting. This avoids forcing
the optimizer to satisfy noisy, differently scaled objectives at full strength
from the first step.

When `save_visualizations: true`, each evaluated split writes 3-panel figures per
field (`True`, `Predicted`, `Error`) under `visualization_dir/<split_name>/`.
