# PyTorch ROCm 7.11 integration notes

This directory is the TheRock-side integration boundary for the custom ROCm 7.11
PyTorch wheel family.

The source of truth for PyTorch wheel packaging and consumer venv creation is now
the PyTorch fork:

```text
ChristophBellmann/rocm-7.11-pytorch-gfx103x
branch: christoph/gfx1031-buildfixes
path: tools/rocm_release/
```

## Policy

TheRock owns:
- ROCm stack build and validation
- in-tree vs. promoted `/opt/rocm` validation profiles
- integration wrappers where needed

The PyTorch fork owns:
- `torch` wheel promotion helpers
- `torchcodec` and `torchaudio` companion wheel helpers
- NumPy ABI probe helpers
- deterministic ROCm/PyTorch project venv creation

Do not reimplement project venv logic in this repo. Use the canonical helper:

```bash
cd /path/to/rocm-7.11-pytorch-gfx103x
bash tools/rocm_release/create_rocm_venv.sh \
  --venv .venv \
  --rocm-prefix /opt/rocm \
  -y
```

For application packages, use the generated constraint-aware wrapper:

```bash
.venv/bin/pip-rocm install openai-whisper
```

## NumPy ABI policy

New custom PyTorch wheel families should be rebuilt and probed with:

```text
NUMPY_SPEC='numpy>=2,<3'
```

Use `numpy<2` only for explicit legacy reproduction of old NumPy-1 ABI wheels.
TensorFlow keeps its own isolated runtime pins and is not part of this PyTorch
policy.

## Promoted wheel aliases

The promoted PyTorch wheel family lives under:

```text
/opt/rocm/wheels/pytorch_rocm711/
```

Stable aliases:

```text
/opt/rocm/wheels/pytorch_rocm711/torch-current.whl
/opt/rocm/wheels/pytorch_rocm711/torchcodec-current.whl
/opt/rocm/wheels/pytorch_rocm711/torchaudio-current.whl
```

## Build/promote flow

After building `torch` in the PyTorch fork and after NumPy ABI probing succeeds:

```bash
cd /path/to/rocm-7.11-pytorch-gfx103x
sudo bash tools/rocm_release/install_pytorch_rocm_wheel_to_opt.sh ./dist/torch-*.whl
sudo bash tools/rocm_release/install_torchcodec_rocm_wheel_to_opt.sh
sudo bash tools/rocm_release/install_torchaudio_rocm_wheel_to_opt.sh
```

Then validate from this TheRock repo:

```bash
python3 validation/validate.py --profile pytorch_rocm711_promoted --yes --power --log
python3 validation/validate.py --profile whisper --yes --power --log
```

## Deprecated local venv helper

`validation/scripts/pytorch_rocm/install_pytorch_rocm_wheel_to_venv.sh` is kept
only as a pointer to the canonical helper and exits intentionally.
