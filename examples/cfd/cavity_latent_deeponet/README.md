# Cavity Field-wise Latent DeepONet

This example extends the cavity DeepONet workflow with a shared-backbone latent
model. The branch and trunk encoders are shared across fields, while each output
field (`u`, `v`, `w`, `p`) has its own latent head and reconstruction path.

In POD mode, the training snapshots are reshaped by case and field, then a
separate spatial POD basis is computed for each field. In autoencoder/decoder
mode, each field gets its own configurable latent dimension and learned decoder
head.

## Key Config Knobs

- `reconstruction_mode`: `pod` or `autoencoder` (`autoencoder` maps to decoder mode)
- `pod_modes`: POD rank per field when `reconstruction_mode: pod`
- `autoencoder_latent_dims`: latent size per field when `reconstruction_mode: autoencoder`

## Run

Activate OpenFOAM before launching the workflow. In this workspace, OpenFOAM is
installed under `/home/shenhui_ruan/OpenFOAM/OpenFOAM-v2312`, so source its
`etc/bashrc` first:

```bash
source /home/shenhui_ruan/OpenFOAM/OpenFOAM-v2312/etc/bashrc
```

The example uses the cavity OpenFOAM case from
`examples/cfd/cavity_deeponet/cavity` and requires `blockMesh`, `icoFoam`, and
`postProcess` to be available in the active shell.

```bash
uv run python examples/cfd/cavity_latent_deeponet/latent_deeponet_cavity.py \
  --config examples/cfd/cavity_latent_deeponet/latent_deeponet_cavity.yaml
```

## Requirements

```bash
uv pip install -r examples/cfd/cavity_latent_deeponet/requirements.txt
```
