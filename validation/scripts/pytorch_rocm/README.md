# ROCm 7.11 release helpers

This directory contains the TheRock-side helpers for the custom ROCm 7.11 PyTorch wheel family.

Scope:
- build the custom `torch` wheel from this repo
- build matching companion wheels for `torchcodec` and `torchaudio`
- promote all three wheels to `/opt/rocm/wheels/pytorch_rocm711/`
- install the promoted wheel family into project venvs with the required ROCm runtime environment

TheRock validation should consume the promoted wheel family from `/opt/rocm` and validate runtime behavior.
It should not own the packaging scripts.

## Layout

- `build_torchcodec_rocm_wheel.sh`
- `build_torchaudio_rocm_wheel.sh`
- `install_pytorch_rocm_wheel_to_opt.sh`
- `install_torchcodec_rocm_wheel_to_opt.sh`
- `install_torchaudio_rocm_wheel_to_opt.sh`
- `install_pytorch_rocm_wheel_to_venv.sh`

Default local release root:
- `validation/workspace/cache/git/`
- `validation/workspace/cache/wheels/pytorch_rocm711/`
- `validation/workspace/cache/venvs/`
- `validation/workspace/cache/install-backups/`

Canonical override:
- `RELEASE_ROOT=/path/to/release-state`

Compatibility note:
- `WORKSPACE_DIR` is still accepted as a legacy alias for `RELEASE_ROOT`.

## Typical flow

Build `torch` with the existing repo-native flow so that a wheel lands in `./dist/`.
Then:

```bash
./validation/scripts/pytorch_rocm/install_pytorch_rocm_wheel_to_opt.sh validation/workspace/cache/wheels/pytorch_rocm711/torch-*.whl
./validation/scripts/pytorch_rocm/build_torchcodec_rocm_wheel.sh --rocm-prefix /opt/rocm
./validation/scripts/pytorch_rocm/install_torchcodec_rocm_wheel_to_opt.sh
./validation/scripts/pytorch_rocm/build_torchaudio_rocm_wheel.sh --rocm-prefix /opt/rocm
./validation/scripts/pytorch_rocm/install_torchaudio_rocm_wheel_to_opt.sh
```

`install_pytorch_rocm_wheel_to_opt.sh` rewrites embedded RPATH/RUNPATH entries
inside the `torch` wheel before promotion so the installed wheel no longer
points back to an in-tree ROCm build directory such as
`build-stage2/dist/rocm/lib`.

That keeps the flow aligned with the other custom wheel families:
- build and verify against the repo-local output first
- promote a system-safe wheel into `/opt/rocm`
- validate the promoted install separately against `/opt/rocm`

Consumer projects can then install the promoted family with:

```bash
./validation/scripts/pytorch_rocm/install_pytorch_rocm_wheel_to_venv.sh --venv .venv --rocm-prefix /opt/rocm
```

Stable system aliases:
- `/opt/rocm/wheels/pytorch_rocm711/torch-current.whl`
- `/opt/rocm/wheels/pytorch_rocm711/torchcodec-current.whl`
- `/opt/rocm/wheels/pytorch_rocm711/torchaudio-current.whl`

## Notes

- These helpers assume a system ROCm install under `/opt/rocm` for the promoted/consumer path.
- If `/opt/rocm` is not available yet, they can fall back to `<repo>/<build-dir>/dist/rocm` for local build verification.
- The consuming venv should currently pin `numpy<2` until the custom wheel family is rebuilt for NumPy 2.x ABI compatibility.
