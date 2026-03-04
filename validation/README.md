# Validation (ROCm usability + workload validation)

This directory contains the **repo-local validation suite** for `TheRock_gfx1031`.
It validates in-tree ROCm artifacts under `<builddir>/dist/rocm` and avoids depending on `/opt/rocm` for runtime checks.

Goals:
1. **ROCm usability proof**: `rocminfo`, HIP compile+run, and small library smokes/benches.
2. **Representative workloads (optional)**: llama.cpp, Ollama, Whisper, Open Interpreter, MFEM, PETSc, PyTorch, TensorFlow wheel build.

Bundled sample inputs:
- `validation/src/assets/samples/audio/Take2_Audio1-1.wav`
- `validation/src/assets/samples/prompts/tiny_prompt.txt`

## Quick start

Default run (`run.profile` from `validation/config/defaults.yaml`, currently `all`):
```bash
python3 validation/scripts/validate.py
```

Lightweight ROCm-only checks (no downloads):
```bash
python3 validation/scripts/validate.py --profile quick --no-downloads
```

Non-interactive default run:
```bash
python3 validation/scripts/validate.py --yes
```

Write per-step logs:
```bash
python3 validation/scripts/validate.py --log
```

## Daily commands

### Main runner

- Default profile (`all`, comprehensive):
  ```bash
  python3 validation/scripts/validate.py
  ```
- Explicit `all` profile:
  ```bash
  python3 validation/scripts/validate.py --profile all
  ```
- Compat profile (`full`, mostly defaults, reduced scope vs `all`):
  ```bash
  python3 validation/scripts/validate.py --profile full
  ```
- Quick profile:
  ```bash
  python3 validation/scripts/validate.py --profile quick
  ```
- Quick + hard no-downloads:
  ```bash
  python3 validation/scripts/validate.py --profile quick --no-downloads
  ```

### Build-dir selection

Default build-dir priority:
- `build-stage2`
- `build`
- `build-stage1`

Commands:
```bash
python3 validation/scripts/validate.py --build-dirs build-stage2
python3 validation/scripts/validate.py --all-build-dirs
```

### Output and monitoring options

```bash
python3 validation/scripts/validate.py --power
python3 validation/scripts/validate.py --no-power
python3 validation/scripts/validate.py --summary-multiline
python3 validation/scripts/validate.py --log
```

### Utilities

```bash
python3 validation/scripts/doctor.py
python3 validation/scripts/cache_gc.py
python3 validation/scripts/cache_gc.py --all
python3 validation/scripts/report_open.py
python3 validation/scripts/report_open.py --open
```

## Profiles and what they mean

Focused profiles:
- llama.cpp: `llama_cpp`, `llama_cpp_infer`, `llama_cpp_smoke`
- Ollama: `ollama`, `ollama_smoke`
- Whisper: `whisper`
- MFEM: `mfem`
- PETSc: `petsc`
- ONNX Runtime ROCm wheel build: `onnxruntime`
- ONNX Runtime in-tree ROCm wheel + inference test: `onnxruntime_in_tree`
- PyTorch GPU compute: `pytorch`
- PyTorch in-tree ROCm enforcement: `pytorch_in_tree`
- PyTorch ROCm 7.11 source build: `pytorch_rocm711_source`
- TensorFlow ROCm wheel build: `tensorflow`

Show CLI help:
```bash
python3 validation/scripts/validate.py --help
```

## Workload-focused one-shot commands

```bash
python3 validation/scripts/llama_cpp_validate.py
python3 validation/scripts/llama_cpp_validate.py --smoke

python3 validation/scripts/ollama_validate.py
python3 validation/scripts/ollama_doctor.py --yes

python3 validation/scripts/whisper_validate.py
python3 validation/scripts/mfem_validate.py
python3 validation/scripts/onnxruntime_validate.py
python3 validation/scripts/tensorflow_validate.py
```

Additional targeted runs:
```bash
python3 validation/scripts/validate.py --profile ollama --yes --power --log
python3 validation/scripts/validate.py --profile petsc --yes --power --log
python3 validation/scripts/validate.py --profile onnxruntime --yes --log
python3 validation/scripts/validate.py --profile onnxruntime_in_tree --yes --power --log
python3 validation/scripts/validate.py --profile pytorch --yes --power
python3 validation/scripts/validate.py --profile tensorflow --yes --log
```

## Custom builds against this ROCm stack

This section documents custom framework builds/wheels that are intended to run against
the custom ROCm stack produced by this repository.

### PyTorch (ROCm 7.11-aligned, source build)

- Primary path in this repo:
  - `validation/config/profiles/pytorch_rocm711_source.yaml`
  - `external-builds/pytorch/build_prod_wheels.py`
- Typical wheel output:
  - `validation/workspace/cache/wheels/pytorch_rocm711/` (or configured wheel dir)
- Typical promote target:
  - `/opt/rocm/wheels/pytorch_rocm711/`

### ONNX Runtime (ROCm)

- Common workflow in this setup:
  - build ONNX Runtime wheel from a dedicated ORT fork/branch
  - then validate against the same ROCm stack used here
- Build helpers in this repo:
  - `validation/scripts/onnxruntime_rocm/build_onnxruntime_rocm_wheel.sh`
  - `validation/scripts/onnxruntime_rocm/start_onnxruntime_rocm_build_systemd.sh`
  - `validation/scripts/onnxruntime_rocm/monitor_onnxruntime_rocm_build.sh`
  - `validation/scripts/onnxruntime_rocm/install_onnxruntime_rocm_wheel_to_opt.sh`
- Artifacts:
  - build workspace: `validation/workspace/builds/onnxruntime_rocm/`
  - wheels: `validation/workspace/cache/wheels/onnxruntime_rocm711/`
- Known fork reference (for reproducibility):
  - Repo: `https://github.com/ChristophBellmann/rocm-7.11-onnxruntime-gfx103x`
  - Branch: `christoph/gfx1031-tls-fix`
  - Commit: `22f739e`
  - Release tag: `v1.22.2-rocm711-gfx1031-tlsfix1`
  - Release asset:
    - `https://github.com/ChristophBellmann/rocm-7.11-onnxruntime-gfx103x/releases/download/v1.22.2-rocm711-gfx1031-tlsfix1/onnxruntime_rocm-1.22.2-cp312-cp312-linux_x86_64.whl`
- Typical promote target:
  - `/opt/rocm/wheels/onnxruntime_rocm711/`
- In-tree validation profile:
  - `validation/config/profiles/onnxruntime_in_tree.yaml`
  - runs:
    - ROCm sanity
    - ONNX Runtime wheel build
    - ONNX Runtime inference test with `ROCMExecutionProvider`
  - inference step verifies:
    - provider availability in session
    - provider events from ONNX Runtime profiling (`rocm_events > 0`)
    - throughput metrics (`avg_ms`, `iters_per_s`)
    - optional power metrics when `--power` is enabled

### TensorFlow ROCm wheel

- Build helpers in this repo:
  - `validation/scripts/tensorflow_rocm/build_tensorflow_rocm_wheel.sh`
  - `validation/scripts/tensorflow_rocm/start_tensorflow_rocm_build_systemd.sh`
  - `validation/scripts/tensorflow_rocm/monitor_tensorflow_rocm_build.sh`
  - `validation/scripts/tensorflow_rocm/install_tensorflow_rocm_wheel_to_opt.sh`
- Artifacts:
  - build workspace: `validation/workspace/builds/tensorflow_rocm/`
  - wheels: `validation/workspace/cache/wheels/tensorflow_rocm_custom/`
  - ccache (validation TensorFlow only): `validation/workspace/cache/ccache/`
- Typical promote target:
  - `/opt/rocm/wheels/tensorflow_rocm_custom/`
- Default source config:
  - `workloads.tensorflow.repo_url`: `https://github.com/ChristophBellmann/rocm-7.11-tensorflow-gfx103x.git`
  - `workloads.tensorflow.ref`: `r2.20-rocm-enhanced`

Cache note: Validation TensorFlow uses `validation/workspace/cache/ccache/` (separate from repo `.ccache/`).

### Quick post-build verification

```bash
python -c "import torch; print(torch.__version__, torch.version.rocm, torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

ONNX Runtime inference validation with power:
```bash
python3 validation/scripts/validate.py --profile onnxruntime_in_tree --yes --power --log
```

## Workloads: behavior and metrics

### Ollama

Enabled by default in `all` with model `llama3.2:3b-instruct-q4_0` (~1.9GB).
Measured metrics:
- `tok/s`
- `ttft` (ms)
- `avg_tok` (ms/token)
- optional power/energy when `--power` is enabled

### llama.cpp (docker)

- Uses AMD ROCm llama.cpp container path.
- Inference mode needs a GGUF model URL (`workloads.llama_cpp.model_url`).
- Reports:
  - `pp_tok/s` (prompt processing)
  - `tg_tok/s` (token generation)
  - optional power/energy

Modes:
- strict GPU inference (default for wrapper):
  ```bash
  python3 validation/scripts/llama_cpp_validate.py
  ```
- smoke-only (no model download/inference):
  ```bash
  python3 validation/scripts/llama_cpp_validate.py --smoke
  ```

### Whisper

Uses bundled short WAV by default; can repeat audio to a target duration (`audio_target_s`) for sustained load.

### MFEM

- Uses pinned MFEM ref from config.
- Forces `-DHIP_PLATFORM=amd` to avoid empty HIP platform edge cases.
- Builds `ex1` explicitly (many MFEM examples are not in default target).

### PETSc

Builds PETSc with HIP and runs a solver check. This can be long.

### PyTorch

`all` defaults to ROCm 7.11-aligned source build to avoid channel mismatch and hidden CPU fallback.
If you need a faster wheel-based check, run `--profile pytorch`.

### TensorFlow ROCm wheel

Heavy source build workflow. See section **Custom builds against this ROCm stack** for scripts,
artifact paths, and default source configuration.

### ONNX Runtime inference (ROCm)

`onnxruntime_in_tree` includes a real inference step after wheel build.
Reported metrics include:
- `avg_ms`
- `iters_per_s`
- `rocm_events` (from ONNX Runtime profiling; must be > 0)
- optional power metrics (`E`, `avgW`, `dW`, `maxW`, `gpu%`, `mem%`) with `--power`

## How the suite works

- **In-tree ROCm activation**:
  each step sets `ROCM_PATH`, `PATH`, `LD_LIBRARY_PATH` to target `<builddir>/dist/rocm`.
- **Sustained-load tests**:
  core checks run long enough to make GPU usage visible, not only short bursts.
- **Optional power sampling**:
  with `--power`, reads AMDGPU sysfs power and integrates approximate energy (`Ws`).
- **Repo-local Python runtime**:
  auto-venv under `validation/workspace/envs/py/`.
- **Download gating**:
  download/build workloads are guarded by one startup prompt (`--yes` to skip prompt).
- **Workload strictness**:
  workload CPU fallback is treated as `FAIL`.
- **gfx1031 compatibility hint**:
  for docker images without `gfx1031` code objects, suite can use `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- **Artifacts location**:
  runtime artifacts remain under `validation/workspace/`.

## Build stage differences

Why results differ between build dirs:
- `build-stage2`: usually complete runtime/tooling, primary target.
- `build-stage1`: bootstrap stage; some runtime checks may legitimately `SKIP`.

## Configuration reference

Base config:
- `validation/config/defaults.yaml`

Profiles:
- `validation/config/profiles/all.yaml` (default profile via `run.profile`)
- `validation/config/profiles/full.yaml` (compat profile)
- `validation/config/profiles/quick.yaml`
- `validation/config/profiles/airgapped.yaml`
- `validation/config/profiles/ollama.yaml`
- `validation/config/profiles/ollama_smoke.yaml`
- `validation/config/profiles/llama_cpp.yaml`
- `validation/config/profiles/llama_cpp_infer.yaml`
- `validation/config/profiles/llama_cpp_smoke.yaml`
- `validation/config/profiles/whisper.yaml`
- `validation/config/profiles/mfem.yaml`
- `validation/config/profiles/pytorch.yaml`
- `validation/config/profiles/pytorch_in_tree.yaml`
- `validation/config/profiles/pytorch_rocm711_source.yaml`
- `validation/config/profiles/onnxruntime.yaml`
- `validation/config/profiles/onnxruntime_in_tree.yaml`
- `validation/config/profiles/petsc.yaml`
- `validation/config/profiles/tensorflow.yaml`

Layout/env hints:
- `validation/config/layout/gfx_targets.yaml`
- `validation/config/layout/install_layouts.yaml`
- `validation/config/layout/env_exports.yaml`

## Directory layout

```text
validation/
├─ README.md
├─ AI_WORKFLOW_VALIDATION.md
├─ ANWEISUNG_STRUKTUR.md
├─ pyproject.toml
├─ requirements-lock.txt
├─ .gitignore
├─ .env.example
├─ config/
├─ scripts/
├─ src/
├─ tests/
└─ workspace/
```

## Legal / third-party

Third-party references used by optional workload checks:
- `validation/src/assets/notices/THIRD_PARTY_NOTICES.md`
