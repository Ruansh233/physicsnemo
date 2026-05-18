# Cavity DeepONet Example

This example trains and evaluates a DeepONet surrogate for the OpenFOAM cavity case in:

- `examples/cfd/cavity_deeponet/cavity`

The model predicts four channels per query point:

- velocity vector field: `(u, v, w)`
- pressure scalar field: `p`

with output order `(u, v, w, p)`.

## Prerequisites

1. Activate your Python environment (recommended workflow in this repo is `uv`).
2. Install example-local dependency:

```bash
uv pip install -r examples/cfd/cavity_deeponet/requirements.txt
```

3. Activate OpenFOAM in your shell before running the example. For example, if you have OpenFOAM installed at `/path/to/OpenFOAM`, run:

```bash
source /path/to/OpenFOAM/etc/bashrc
```

The runner checks `blockMesh`, `icoFoam`, and `postProcess` availability and fails fast if OpenFOAM is not active.

## Run The Example

Use the thin wrapper script:

```bash
uv run python examples/cfd/cavity_deeponet/deeponet_cavity.py --config examples/cfd/cavity_deeponet/deeponet_cavity.yaml
```

You can also run without `--config` to use built-in defaults from `CavityDeepONetConfig`.

## How `deeponet_cavity.yaml` Is Used

`examples/cfd/cavity_deeponet/deeponet_cavity.yaml` is loaded by `load_config()` in:

- `examples/cfd/cavity_deeponet/cavity_deeponet/config.py`

Behavior:

1. Build default structured config from `CavityDeepONetConfig`.
2. If `--config` is passed, load the YAML with `OmegaConf.load(...)`.
3. Merge user YAML over defaults via `OmegaConf.merge(...)`.
4. Materialize the final dataclass and pass it through the pipeline.

The merged config drives:

- split generation (`viscosity_seed`, `train_case_count`, `validate_case_count`, `test_case_count`, `manifest_filename`)
- OpenFOAM data generation (`case_path`, `run_root`, `run_openfoam`)
- physical validity bounds (`nu_min`, `nu_max`, `reynolds_min`, `reynolds_max`)
- DeepONet architecture (`latent_dim`, `branch_layers`, `trunk_layers`, `layer_size`, `activation_fn`)
- optimization (`learning_rate`, `weight_decay`, `train_steps`, `lbfgs_steps`, `lbfgs_lr`)
- normalization and output settings (`normalize_inputs`, `normalize_targets`, `save_visualizations`, `visualization_dir`, `visualization_max_cases`)
- model persistence and reuse (`model_checkpoint_path`, `save_trained_model`, `use_saved_model_for_unseen`)

## Model Persistence And Unseen Test

- Main workflow (`deeponet_cavity.py`) trains on train/validate/test splits and saves a checkpoint to `model_checkpoint_path` when `save_trained_model: true`.
- Unseen workflow (`deeponet_cavity_unseen_viscosity.py`) supports two modes:
  - `use_saved_model_for_unseen: true`: load saved model + normalizers from `model_checkpoint_path` and evaluate only unseen-test viscosities.
  - `use_saved_model_for_unseen: false`: train on `train_viscosity_values`, optionally save checkpoint, then evaluate `test_viscosity_values`.

## Notes On Generated Run Directories

Generated foamlib run folders (for example `.foamlib_runs`, `.foamlib_runs_train`, and `.foamlib_runs_unseen_test`) are local execution artifacts and are ignored by `.gitignore`:

- `examples/cfd/cavity_deeponet/.foamlib_runs*/`

If present locally and not needed, you can remove them safely.
