# Custom ROCm Artifacts

This file separates:

- system-installable custom artifacts under `/opt/rocm`
- git-only ROCm/gfx1031 fixes that live in source repos and are not separate system packages

## 1. System-installable custom wheels

These are the currently promoted custom Python artifacts:

### ONNX Runtime

- repo family: `rocm-7.11-onnxruntime-gfx103x`
- system dir: `/opt/rocm/wheels/onnxruntime_rocm711/`
- active wheel:
  - `onnxruntime_rocm-1.22.2-cp312-cp312-linux_x86_64.whl`
- stable alias:
  - `onnxruntime-current.whl`

### PyTorch family

- repo family: `rocm-7.11-pytorch-gfx103x`
- system dir: `/opt/rocm/wheels/pytorch_rocm711/`
- active wheels:
  - `torch-2.11.0a0+git7c67cbd-cp312-cp312-linux_x86_64.whl`
  - `torchaudio-2.11.0+34c52a6-cp312-cp312-linux_x86_64.whl`
  - `torchcodec-0.11.0a0-cp312-cp312-linux_x86_64.whl`
- stable aliases:
  - `torch-current.whl`
  - `torchaudio-current.whl`
  - `torchcodec-current.whl`

### TensorFlow

- repo family: `rocm-7.11-tensorflow-gfx103x`
- system dir: `/opt/rocm/wheels/tensorflow_rocm_custom/`
- active wheel:
  - `tensorflow-2.20.0.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl`
- stable alias:
  - `tensorflow-current.whl`

## 2. Source-of-truth build and promotion locations

The delivery rule is:

1. build repo-local against custom ROCm
2. validate repo-local against in-tree ROCm
3. promote to `/opt/rocm/wheels/...`
4. validate again against `/opt/rocm`

### PyTorch family

- source repo:
  - `validation/workspace/cache/git/pytorch_rocm711`
- release helpers:
  - `validation/workspace/cache/git/pytorch_rocm711/tools/rocm_release/`

### ONNX Runtime

- source repo:
  - `validation/workspace/cache/git/onnxruntime_rocm711`
- release helpers:
  - `validation/workspace/cache/git/onnxruntime_rocm711/tools/rocm_release/`

### TensorFlow

- source repo:
  - `validation/workspace/builds/tensorflow_rocm/tensorflow`
- release helpers:
  - `validation/workspace/builds/tensorflow_rocm/tensorflow/tools/rocm_release/`

Note:
- TensorFlow currently uses `validation/workspace/builds/...` as its active source/build location.
- That is functional, but less symmetric than the `cache/git/...` pattern used by PyTorch and ORT.

## 3. Git-only ROCm/gfx1031 fixes

These are real fixes, but they are not separate installable system artifacts.

### TheRock build policy fixes

Files:
- `config_gfx1031.yaml`
- `CMakeLists.txt`

Examples:
- `hipBLASLt` disabled by default for gfx1031
- `hipSPARSELt` disabled by default for gfx1031

These are stability/build-policy decisions for the custom ROCm stack, not standalone packages.

### PyTorch fork fixes

Repo:
- `validation/workspace/cache/git/pytorch_rocm711`

Examples from current branch history:
- `rocm: make hipblaslt optional for gfx1031 builds`
- `miopen: query workspace size by solution_id for immediate conv path`
- `miopen: size find() workspace from max immediate-solution requirement`

These are source/build/runtime fixes inside the fork.

### ONNX Runtime fork fixes

Repo:
- `validation/workspace/cache/git/onnxruntime_rocm711`

Examples from current branch history:
- `Fix ROCm provider static TLS loading and remove preload dependency`
- `Relax flatbuffers generated header version checks for toolchain compatibility`

These are provider/toolchain fixes inside the fork.

### TensorFlow fork fixes

Repo:
- `validation/workspace/builds/tensorflow_rocm/tensorflow`

Examples from current branch history:
- `ROCm gfx1031 build fixes: hipBLASLt gating, flatbuffers v25 tolerance, device id support`
- `gfx1031: make rocm wheel build self-contained`

These are source/build fixes inside the fork.

## 4. Packaging-only fixes

These are also not separate installable artifacts. They are part of the promotion logic.

### PyTorch wheel promotion

- helper rewrites wheel `RPATH/RUNPATH` so promoted wheels load ROCm from `/opt/rocm`
- helper validates important runtime constraints before install

### ONNX Runtime wheel promotion

- helper rewrites ORT provider/runtime `RPATH`s so promoted wheels resolve ROCm from `/opt/rocm`

### TensorFlow wheel promotion

- promote helper standardizes wheel placement and stable aliasing under `/opt/rocm/wheels/tensorflow_rocm_custom/`

## 5. What is not a separate system artifact

The following are intentionally not separate installable packages:

- gfx1031 build policy in TheRock
- source patches in the framework forks
- validation profiles and workload logic in `validation/`
- promotion helpers under each fork's `tools/rocm_release/`

They matter operationally, but they are not another artifact family beyond the promoted wheels.
