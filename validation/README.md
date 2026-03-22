# Validation (ROCm usability + workload validation)

This directory contains the **repo-local validation suite** for `TheRock_gfx1031`.
Its default mode validates in-tree ROCm artifacts under `<builddir>/dist/rocm`.
Separate `*_promoted` profiles validate explicitly promoted system artifacts under `/opt/rocm`.

Goals:
1. **ROCm usability proof**: `rocminfo`, HIP compile+run, and small library smokes/benches.
2. **Representative workloads (optional)**: llama.cpp, Ollama, Whisper, Open Interpreter, MFEM, PETSc, PyTorch, TensorFlow wheel build.

Bundled sample inputs:
- `validation/src/assets/samples/audio/Take2_Audio1-1.wav`
- `validation/src/assets/samples/prompts/tiny_prompt.txt`

## Quick start

Default run (`run.profile` from `validation/config/defaults.yaml`, currently `all`):
```bash
python3 validation/validate.py
```

Lightweight ROCm-only checks (no downloads):
```bash
python3 validation/validate.py --profile quick --no-downloads
```

Non-interactive default run:
```bash
python3 validation/validate.py --yes
```

Write per-step logs:
```bash
python3 validation/validate.py --log
```

## Daily commands

### Main runner

Primary interface: use `validation/validate.py --profile ...`.
The public CLI surface lives directly under `validation/`.
`validation/scripts/` is internal launcher/bootstrap code, not a second user-facing API.

- Default profile (`all`, comprehensive):
  ```bash
  python3 validation/validate.py
  ```
- Explicit `all` profile:
  ```bash
  python3 validation/validate.py --profile all
  ```
- Compat profile (`full`, mostly defaults, reduced scope vs `all`):
  ```bash
  python3 validation/validate.py --profile full
  ```
- Quick profile:
  ```bash
  python3 validation/validate.py --profile quick
  ```
- Quick + hard no-downloads:
  ```bash
  python3 validation/validate.py --profile quick --no-downloads
  ```

### Build-dir selection

Default build-dir priority:
- `build-stage2`
- `build`
- `build-stage1`

Commands:
```bash
python3 validation/validate.py --build-dirs build-stage2
python3 validation/validate.py --all-build-dirs
```

### Output and monitoring options

```bash
python3 validation/validate.py --power
python3 validation/validate.py --no-power
python3 validation/validate.py --summary-multiline
python3 validation/validate.py --log
```

### Utilities

```bash
python3 validation/doctor.py
python3 validation/cache_gc.py
python3 validation/cache_gc.py --all
python3 validation/report_open.py
python3 validation/report_open.py --open
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
- ONNX Runtime in-tree real Piper TTS graph: `onnxruntime_in_tree_tts`
- ONNX Runtime in-tree real Piper TTS CPU-vs-ROCm benchmark: `onnxruntime_in_tree_tts_benchmark`
- ONNX Runtime in-tree ROCm wheel + MIGraphX EP inference test: `onnxruntime_migraphx_build`
- ONNX Runtime promoted ROCm wheel from `/opt/rocm`: `onnxruntime_rocm711_promoted`
- ONNX Runtime promoted real Piper TTS graph from `/opt/rocm`: `onnxruntime_rocm711_promoted_tts`
- ONNX Runtime promoted real Piper TTS CPU-vs-ROCm benchmark: `onnxruntime_rocm711_promoted_tts_benchmark`
- PyTorch GPU compute: `pytorch`
- PyTorch in-tree ROCm enforcement: `pytorch_in_tree`
- PyTorch ROCm 7.11 source build: `pytorch_rocm711_source`
- PyTorch ROCm 7.11 promoted wheel family from `/opt/rocm`: `pytorch_rocm711_promoted`
- TensorFlow ROCm wheel build: `tensorflow`
- TensorFlow in-tree functional-only check: `tensorflow_in_tree_functional_only`
- TensorFlow promoted wheel from `/opt/rocm`: `tensorflow_rocm_custom_promoted`

Show CLI help:
```bash
python3 validation/validate.py --help
```

## Targeted profile runs

Use the root CLI directly. `validation/scripts/` is internal implementation,
not the public command surface.

```bash
python3 validation/validate.py --profile llama_cpp_infer --yes --power --log
python3 validation/validate.py --profile llama_cpp_smoke --yes --power --log

python3 validation/validate.py --profile ollama --yes --power --log
python3 validation/validate.py --profile ollama_smoke --yes --power --log

python3 validation/validate.py --profile whisper --yes --power --log
python3 validation/validate.py --profile mfem --yes --power --log
python3 validation/validate.py --profile onnxruntime --yes --log
python3 validation/validate.py --profile tensorflow --yes --log
```

Additional targeted runs:
```bash
python3 validation/validate.py --profile ollama --yes --power --log
python3 validation/validate.py --profile petsc --yes --power --log
python3 validation/validate.py --profile onnxruntime --yes --log
python3 validation/validate.py --profile onnxruntime_in_tree --yes --power --log
python3 validation/validate.py --profile onnxruntime_in_tree_tts --yes --power --log
python3 validation/validate.py --profile onnxruntime_in_tree_tts_benchmark --yes --log
python3 validation/validate.py --profile onnxruntime_migraphx_build --yes --log
python3 validation/validate.py --profile onnxruntime_rocm711_promoted_tts --yes --power --log
python3 validation/validate.py --profile onnxruntime_rocm711_promoted_tts_benchmark --yes --log
python3 validation/validate.py --profile pytorch --yes --power
python3 validation/validate.py --profile tensorflow --yes --log
```

Optional secondary consumer-side repro for the real Piper failure
(outside this repo, if a consumer checkout exists):
```bash
cd <consumer-repo>
source ./scripts/env/activate_rocm_torch_env.sh
.venv/bin/python scripts/eval/repro_piper_onnx_provider.py \
  --model training_local/datasets/external/piper_voices/en_US-lessac-low.onnx \
  --provider rocm \
  --seed 0
```

Preferred structural repro stays in `validation/`:
```bash
python3 validation/validate.py --profile onnxruntime_in_tree_tts --yes --power --log
```

Historical narrow in-tree diagnostic override (kept only as an earlier narrowing step):
```bash
ORT_ROCM_FORCE_CPU_OP_NODES='Mul@/dp/flows.' \
python3 validation/validate.py --profile onnxruntime_in_tree_tts --yes --power --log
```

## Custom builds against this ROCm stack

This section documents custom framework builds/wheels that are intended to run against
the custom ROCm stack produced by this repository.

### PyTorch (ROCm 7.11-aligned, source build)

- Primary path in this repo:
  - `validation/config/profiles/pytorch_rocm711_source.yaml`
  - `external-builds/pytorch/build_prod_wheels.py`
- Boundary note:
  - `external-builds/pytorch/` is the in-tree/upstream-oriented source-build helper area.
  - It is not the source of truth for the promoted custom gfx1031 wheel family.
- Known fork reference (for reproducibility):
  - Repo: `https://github.com/ChristophBellmann/rocm-7.11-pytorch-gfx103x`
  - Branch: `christoph/gfx1031-buildfixes`
- Typical wheel output:
  - `validation/workspace/cache/wheels/pytorch_rocm711/` (or configured wheel dir)
- Preferred validation-side source checkout for the source-build helper:
  - `validation/workspace/cache/git/pytorch_rocm711/`
- Typical promote target:
  - `/opt/rocm/wheels/pytorch_rocm711/`
  - stable alias:
    - `/opt/rocm/wheels/pytorch_rocm711/torch-current.whl`
    - `/opt/rocm/wheels/pytorch_rocm711/torchcodec-current.whl` (optional companion wheel)
    - `/opt/rocm/wheels/pytorch_rocm711/torchaudio-current.whl` (optional companion wheel)
- Packaging helpers now live in the PyTorch fork and are the source of truth:
  - Repo: `https://github.com/ChristophBellmann/rocm-7.11-pytorch-gfx103x`
  - Directory: `tools/rocm_release/`
  - Typical local commands there:
    - `./tools/rocm_release/install_pytorch_rocm_wheel_to_opt.sh`
    - `./tools/rocm_release/build_torchcodec_rocm_wheel.sh`
    - `./tools/rocm_release/install_torchcodec_rocm_wheel_to_opt.sh`
    - `./tools/rocm_release/build_torchaudio_rocm_wheel.sh`
    - `./tools/rocm_release/install_torchaudio_rocm_wheel_to_opt.sh`
    - `./tools/rocm_release/install_pytorch_rocm_wheel_to_venv.sh`
- Typical cached artifact location in the PyTorch fork:
  - `.rocm_release/wheels/pytorch_rocm711/`
- Cache note:
  - Use `validation/workspace/cache/...` directly.
  - No parallel cache root under `validation/` is supported.
- Promote helper there:
  - `tools/rocm_release/install_pytorch_rocm_wheel_to_opt.sh`
  - validates:
    - `libtorch_hip.so` does not depend on `libhipblaslt`
  - reports:
    - whether `libtorch_cpu.so` declares `libomp` directly or expects the ROCm `libomp` runtime via the runtime environment
  - compatibility note:
    - consuming venvs should currently pin `numpy<2` because the wheel is built against the NumPy 1.x ABI
  - backs up the destination directory under:
    - `.rocm_release/install-backups/<timestamp>/pytorch_wheels/`
  - and preserves an existing target wheel as:
    - `/opt/rocm/wheels/pytorch_rocm711/<wheel>.bak_<timestamp>`
- Project venv consumer helper:
  - `tools/rocm_release/install_pytorch_rocm_wheel_to_venv.sh`
  - default input wheel:
    - `/opt/rocm/wheels/pytorch_rocm711/torch-current.whl`
  - automatically installs:
    - `/opt/rocm/wheels/pytorch_rocm711/torchcodec-current.whl`
    - `/opt/rocm/wheels/pytorch_rocm711/torchaudio-current.whl`
    - when that companion wheel is present
  - writes venv-local runtime wrappers:
    - `<venv>/bin/activate_rocm_pytorch.sh`
    - `<venv>/bin/python-rocm`
- Validation profile for the promoted wheel family:
  - `validation/config/profiles/pytorch_rocm711_promoted.yaml`
  - installs:
    - `/opt/rocm/wheels/pytorch_rocm711/torch-current.whl`
    - `/opt/rocm/wheels/pytorch_rocm711/torchcodec-current.whl`
    - `/opt/rocm/wheels/pytorch_rocm711/torchaudio-current.whl`
  - removes conflicting in-tree ROCm Python SDK packages from the validation venv first
  - validates:
    - imports of `torch`, `torchcodec`, and `torchaudio`
    - GPU execution with sustained conv workloads
    - runtime source prefix `/opt/rocm`
    - performance and power metrics in the normal validation report

### ONNX Runtime (ROCm)

- Common workflow in this setup:
  - build ONNX Runtime wheel from a dedicated ORT fork/branch
  - then validate against the same ROCm stack used here
- Build helpers in this repo:
  - `validation/scripts/onnxruntime_rocm/build_onnxruntime_rocm_wheel.sh` (TheRock integration wrapper)
  - `validation/scripts/onnxruntime_rocm/systemd_onnxruntime_rocm_build.sh` (`start`/`monitor`)
  - `validation/scripts/onnxruntime_rocm/start_onnxruntime_rocm_build_systemd.sh` (compat wrapper)
  - `validation/scripts/onnxruntime_rocm/monitor_onnxruntime_rocm_build.sh` (compat wrapper)
  - `validation/scripts/onnxruntime_rocm/install_onnxruntime_rocm_wheel_to_opt.sh` (TheRock integration wrapper)
  - `validation/scripts/onnxruntime_rocm/verify_onnxruntime_rocm_provider_sync.sh` (local provider/source freshness check)
- Artifacts:
  - build workspace: `validation/workspace/builds/onnxruntime_rocm/`
  - wheels: `validation/workspace/cache/wheels/onnxruntime_rocm711/`
- TheRock wrapper defaults:
  - `ROCM_PATH=<repo>/build-stage2/dist/rocm`
  - enables `ccache` launchers when `ccache` is available and `ORT_USE_CCACHE!=0`
- Known fork reference (for reproducibility):
  - Repo: `https://github.com/ChristophBellmann/rocm-7.11-onnxruntime-gfx103x`
  - Branch: `christoph/gfx1031-buildfixes`
  - Commit: `d48fd80dd`
  - Release tag: `v1.22.2-rocm711-gfx1031-tlsfix1`
  - Release asset:
    - `https://github.com/ChristophBellmann/rocm-7.11-onnxruntime-gfx103x/releases/download/v1.22.2-rocm711-gfx1031-tlsfix1/onnxruntime_rocm-1.22.2-cp312-cp312-linux_x86_64.whl`
- Typical promote target:
  - `/opt/rocm/wheels/onnxruntime_rocm711/`
  - stable alias:
    - `/opt/rocm/wheels/onnxruntime_rocm711/onnxruntime-current.whl`
- Packaging helpers now live in the ORT fork and are the source of truth:
  - Repo: `https://github.com/ChristophBellmann/rocm-7.11-onnxruntime-gfx103x`
  - Directory: `tools/rocm_release/`
  - Typical local commands there:
    - `./tools/rocm_release/build_onnxruntime_rocm_wheel.sh`
    - `./tools/rocm_release/install_onnxruntime_rocm_wheel_to_opt.sh`
  - Typical cached artifact location in the ORT fork:
    - `.rocm_release/wheels/onnxruntime_rocm711/`
  - launcher-based rebuild acceleration is supported there via:
    - `CMAKE_C_COMPILER_LAUNCHER=ccache`
    - `CMAKE_CXX_COMPILER_LAUNCHER=ccache`
- After local ORT provider source edits, verify that the active provider really
  matches the rebuilt source before trusting debug results:
  - `bash validation/scripts/onnxruntime_rocm/verify_onnxruntime_rocm_provider_sync.sh --source onnxruntime/core/providers/rocm/<file>.cc`
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
- Real TTS graph profiles:
  - `validation/config/profiles/onnxruntime_in_tree_tts.yaml`
  - `validation/config/profiles/onnxruntime_rocm711_promoted_tts.yaml`
  - expect a staged Piper ONNX model under:
    - `validation/workspace/cache/models/onnxruntime_tts/`
  - the staged model and sidecar must be real files inside that cache dir:
    - copy them there
    - do not symlink to a consumer checkout such as `Mogli-Lab`
  - default behavior:
    - pick the newest `*.onnx` from that cache directory
    - use `<model>.onnx.json` as the sidecar config
    - build deterministic valid phoneme-id tensors directly from the Piper config
    - run a CPU reference first, then the same graph on `ROCMExecutionProvider`
  - report/verify:
    - case count and average case latency
    - `ROCMExecutionProvider` profile events
    - runtime source prefix (`build-stage2/dist/rocm` or `/opt/rocm`)
    - MIOpen workspace warning count from stderr
  - purpose:
    - expose real Piper/ORT/MIOpen regressions that do not show up in the small `mnist.onnx` validation
- MIGraphX EP build+validation profile:
  - `validation/config/profiles/onnxruntime_migraphx_build.yaml`
  - runs:
    - ROCm sanity
    - ONNX Runtime wheel build with `--use_migraphx`
    - ONNX Runtime inference test with `MIGraphXExecutionProvider`
- Promoted system validation profile:
  - `validation/config/profiles/onnxruntime_rocm711_promoted.yaml`
  - installs from:
    - `/opt/rocm/wheels/onnxruntime_rocm711/`
    - stable alias: `/opt/rocm/wheels/onnxruntime_rocm711/onnxruntime-current.whl`
  - validates:
    - `ROCMExecutionProvider` inference
    - runtime source prefix `/opt/rocm`
    - throughput metrics and optional power metrics

### TensorFlow ROCm wheel

- Build helpers in this repo:
  - `validation/scripts/tensorflow_rocm/build_tensorflow_rocm_wheel.sh`
  - `validation/scripts/tensorflow_rocm/start_tensorflow_rocm_build_systemd.sh`
  - `validation/scripts/tensorflow_rocm/monitor_tensorflow_rocm_build.sh`
  - `validation/scripts/tensorflow_rocm/install_tensorflow_rocm_wheel_to_opt.sh`
- Separation of concerns:
  - `validation/scripts/tensorflow_rocm/build_tensorflow_rocm_wheel.sh` is now only the TheRock integration wrapper.
  - The actual TensorFlow wheel packaging helpers live in the TensorFlow fork itself under:
    - `tools/rocm_release/build_tensorflow_rocm_wheel.sh`
    - `tools/rocm_release/install_tensorflow_rocm_wheel_to_opt.sh`
- Artifacts:
  - build workspace: `validation/workspace/builds/tensorflow_rocm/`
  - wheels: `validation/workspace/cache/wheels/tensorflow_rocm_custom/`
  - ccache (validation TensorFlow only): `validation/workspace/cache/ccache/`
  - functional runtime venv: `validation/workspace/envs/tensorflow_rocm/`
  - preferred fork checkout: `validation/workspace/cache/git/tensorflow_rocm711/`
  - legacy wrapper fallback only: `validation/workspace/builds/tensorflow_rocm/tensorflow/`
- Typical promote target:
  - `/opt/rocm/wheels/tensorflow_rocm_custom/`
  - stable alias:
    - `/opt/rocm/wheels/tensorflow_rocm_custom/tensorflow-current.whl`
- Promote helper behavior:
  - `validation/scripts/tensorflow_rocm/install_tensorflow_rocm_wheel_to_opt.sh`
  - delegates to the TensorFlow fork release helper
  - backs up the current destination directory under:
    - `.rocm_release/install-backups/<timestamp>/tensorflow_wheels/`
  - and also preserves an existing target wheel as:
    - `/opt/rocm/wheels/tensorflow_rocm_custom/<wheel>.bak_<timestamp>`
- Default source config:
  - `workloads.tensorflow.repo_url`: `https://github.com/ChristophBellmann/rocm-7.11-tensorflow-gfx103x.git`
  - `workloads.tensorflow.ref`: `christoph/gfx1031-buildfixes`
- Wheel runtime fix:
  - the build helper postprocesses the produced TensorFlow wheel and renames TensorFlow-bundled LLVM dynamic symbols across the wheel DSOs. This prevents ROCm COMGR / HIP from binding against TensorFlow's private LLVM copy at runtime when validating against the in-tree ROCm build.
- Functional validation isolation:
  - the post-build TensorFlow matmul step installs the wheel into `validation/workspace/envs/tensorflow_rocm/`, not the shared validation venv.
  - this prevents TensorFlow-specific dependency pins from mutating unrelated validation workloads.
- Validation profile `tensorflow` also runs a post-build TensorFlow GPU matmul benchmark and reports `tflops_est` plus the computed operation (`C=A*B` dense matmul).
- Validation profile `tensorflow_in_tree_functional_only` reuses an already built local wheel and validates the in-tree ROCm runtime path.
- Validation profile `tensorflow_rocm_custom_promoted` installs from `/opt/rocm/wheels/tensorflow_rocm_custom/` and validates the system ROCm runtime path (`/opt/rocm`).
- Important boundary:
  - `validation/validate.py --profile tensorflow` remains an **in-tree ROCm** workflow.
  - a system-installed `/opt/rocm` stack is smoke-tested separately after wheel promotion, not by changing the default validation contract.

System smoke example after promotion:

```bash
python3 -m venv validation/workspace/envs/tensorflow_rocm_system

export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HSA_PATH=/opt/rocm
export PATH="$PWD/validation/workspace/envs/tensorflow_rocm_system/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/lib/host-math/lib:/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/llvm/lib:${LD_LIBRARY_PATH:-}"
export TF_ROCM_DISABLE_HIPBLASLT=1
export TF_ROCM_USE_HIPBLASLT=0
export TF_ROCM_DISABLE_HIPBLASLT_INIT=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install --force-reinstall 'numpy<2' 'protobuf<7' \
  /opt/rocm/wheels/tensorflow_rocm_custom/tensorflow-2.20.0.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl
```

Validated system smoke on **2026-03-08**:
- runtime libs loaded from `/opt/rocm/lib`
- op: `C = A * B` dense matmul
- shape: `4096 x 4096 x 4096`
- dtype: `fp16`
- throughput: `21.61 TFLOPS`

Cache note: Validation TensorFlow uses `validation/workspace/cache/ccache/` (separate from repo `.ccache/`).

### Quick post-build verification

```bash
python -c "import torch; print(torch.__version__, torch.version.rocm, torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

ONNX Runtime inference validation with power:
```bash
python3 validation/validate.py --profile onnxruntime_in_tree --yes --power --log
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
- strict GPU inference:
  ```bash
  python3 validation/validate.py --profile llama_cpp_infer --yes --power --log
  ```
- smoke-only (no model download/inference):
  ```bash
  python3 validation/validate.py --profile llama_cpp_smoke --yes --power --log
  ```

### Whisper

Uses bundled short WAV by default; can repeat audio to a target duration (`audio_target_s`) for sustained load.

### MFEM

- Uses pinned MFEM ref from config.
- Source checkout lives in `validation/workspace/cache/git/mfem/`.
- Build artifacts live in `validation/workspace/builds/mfem_build/`.
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

`onnxruntime_in_tree_tts` is the heavier diagnostic path for Piper-style TTS models.
It is intentionally separate from the small MNIST benchmark because it exercises
the real TTS graph shape/scales path that exposed real ROCm-only correctness
regressions that the small MNIST benchmark did not catch. The TTS profile is
only green if:
- the CPU reference passes
- the ROCm run passes
- `ROCMExecutionProvider` events are present
- the final ROCm output matches the CPU reference within the configured
  `rtol`/`atol`
- on shape mismatches, validation now writes focused node-level JSON diagnostics
  under the current run directory:
  - `validation/workspace/runs/<run_id>/artifacts/onnxruntime_tts_shape_diag_<case>_current.json`
  - `validation/workspace/runs/<run_id>/artifacts/onnxruntime_tts_shape_diag_<case>_nofast.json`
  - exact-chain artifacts under
    `validation/workspace/runs/<run_id>/artifacts/onnxruntime_tts_shape_diag_<case>_<mode>_exact_chain/`
    including `dp_shape_cast.report.json`

Current expected state for `onnxruntime_in_tree_tts` on a healthy stack:
- CPU reference passes
- ROCm path passes
- `ROCMExecutionProvider` events are present
- `compare_ok=<all>/<all>`
- `miopen_warn=0`
- current March 2026 real-model diagnosis also requires a fixed seed because the
  Piper graph contains `RandomNormalLike`:
  - `ort.set_seed(0)`
  - `numpy.random.seed(0)`
  - for deterministic Piper validation, freeze `RandomNormalLike` nodes to
    dynamic zero tensors so the run follows real ROCm bugs instead of
    provider-local RNG drift
- current accepted green path for full real Piper TTS on gfx1031:
  - ORT ROCm provider fixes from the ONNX Runtime fork
  - deterministic `RandomNormalLike` freezing in `validation`
- CPU fallback overrides such as:
  - `ORT_ROCM_FORCE_CPU_OP_NODES='Mul@/dp/flows.'`
  are diagnostic only and must not be treated as the stack fix

For performance triage there are now dedicated Piper benchmark profiles:
- `onnxruntime_in_tree_tts_benchmark`
- `onnxruntime_rocm711_promoted_tts_benchmark`

These benchmark profiles keep the deterministic `RandomNormalLike` freeze, then
measure real-case Piper timings in isolated child processes for:
- session creation
- one cold inference
- one second inference after a single warm-up run

The per-case JSON artifact is written to:
- `validation/workspace/runs/<run_id>/artifacts/onnxruntime_tts_benchmark.json`

Current March 21 2026 in-tree benchmark snapshot on gfx1031:
- CPU create: about `743 ms`
- CPU cold inference: about `45 ms`
- CPU second-run inference: about `28 ms`
- ROCm create: about `1089 ms`
- ROCm cold inference: about `11895 ms`
- ROCm second-run inference: about `3562 ms`
- benchmark artifact: `validation/workspace/runs/2026-03-21_171622/artifacts/onnxruntime_tts_benchmark.json`

That benchmark is intentionally diagnostic, not a pass/fail correctness gate:
- cold CPU/ROCm measurement must succeed
- second-run ROCm failures are recorded in the artifact, because on a 12 GiB
  card they are part of the performance/stability picture for this real model

Current March 2026 diagnosis snapshot:
- the primary investigation site is `validation/`, not the consumer repo
- a dedicated ORT runtime venv is used for TTS validation:
  - `validation/workspace/envs/onnxruntime_rocm`
- the suite now treats semantic CPU-vs-ROCm mismatches as `FAIL`, not only hard
  exceptions
- there are at least two distinct ROCm-only bug families in the real Piper
  graph:
  - a confirmed fast-reduction correctness bug in the encoder normalization path
    (`ReduceMean`/`ReduceSum` on ROCm can produce the wrong value for the same
    tensor; `ORT_ROCM_DISABLE_FAST_REDUCTION=1` fixes that path diagnostically)
  - a second independent non-reduction bug remains in repeated local
    `dp/flows.7` ramp subgraphs even with fast reduction disabled
- current March 2026 shape-path diagnosis for the local staged Lessac model:
  - the first shape-driving divergence can already appear in the top-level
    `/dp/Split -> Exp -> Mul -> Ceil -> Cast -> CumSum -> Reshape_1` chain
  - on the current stack this can inflate the duration length from CPU `41` to
    ROCm `2992+` for the same real input case
  - the isolated `dp_shape_cast` mini-repro stays green on ROCm when it is fed
    CPU-captured boundary tensors, so that exact chain is not the primary bug
  - the first currently proven upstream boundary mismatch feeding that chain is
    `/dp/flows.0/Mul_1_output_0`
  - the smaller `/dp/flows.0/Sub -> /dp/flows.0/Mul -> /dp/flows.0/Mul_1`
    mini-repro is also green with CPU-captured boundary tensors
  - the first currently proven non-constant upstream mismatch for that smaller
    flow path is `/dp/flows.3/Mul_35_output_0`
  - a standalone deterministic profile now exists for that branch work:
    - `onnxruntime_in_tree_tts_flow_probe`
    - it runs the internal probe helper as its own validation step
    - it freezes Piper `RandomNormalLike` nodes to dynamic zero tensors by
      default
  - current deterministic branch probing above that point still localizes one
    step further to `/dp/flows.3/Split_output_0`, fed from
    `/dp/flows.4/Slice_output_0`
  - the isolated deterministic `dp_flow3_split_path` mini-repro is green on
    ROCm when it is fed CPU-captured boundary tensors; the first upstream
    boundary mismatch for that path is `/dp/flows.5/Mul_35_output_0`
  - forcing `miopen_conv_use_max_workspace=0` versus `=1` in the current
    staged-model debug repro does not change the first mismatching tensor, the
    `Cast` explosion (`2992`), or the final blown-up output length
  - the current workspace-warning family still reports `provided ... size:
    33554432` in both runs; the ORT ROCm `ConvTranspose` algo-search path also
    still hard-codes the 32 MiB search buffer
  - after the workspace fix, the last large full-model mismatch was traced to
    the second `/RandomNormalLike` in the `/flow` branch; once both Piper
    RNG nodes are frozen dynamically, the remaining CPU-vs-ROCm drift drops to
    low `1e-5` noise and the real TTS validation is green again
  - `ORT_ROCM_DISABLE_FAST_REDUCTION=1` does not resolve that shape-path family
    in the current staged-model repro
- the earlier `flow7` Mul suspicion is now downgraded:
  - `/dp/flows.7/Mul_10`
  - `/dp/flows.7/Mul_16`
  still fail in the full graph, but an exact isolated `flow7` gate mini-repro
  is green on ROCm when it is fed CPU-captured boundary tensors
- that means the faulty `Mul_10`/`Mul_16` values are downstream symptoms, not a
  proven bare Mul kernel bug
- current deterministic upstream path above those gates:
  - the first stable full-graph boundary mismatch for the local logits path is
    `/dp/flows.7/Transpose_output_0`
  - targeted full-graph probes above that point show earlier deterministic
    divergences already at:
    - `/dp/flows.7/convs/Mul_15_output_0`
    - `/dp/flows.7/pre/Conv_output_0`
    - `/dp/Mul_output_0`
  - the deterministic `/dp/RandomNormalLike -> /Gather_2 -> /dp/Mul_1 ->
    /dp/flows.8/Slice -> /dp/flows.7/Split` path is green, so that input path
    is no longer treated as the primary deterministic fault site
- exact node forcing remains debug-only and is not an accepted solution
- standalone minimal extracted subgraphs can run correctly on ROCm while the
  full graph still fails; the current best diagnosis is therefore still
  topology-/partitioning-specific ROCm EP behavior in the real Piper graph

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
- `validation/config/profiles/onnxruntime_migraphx.yaml`
- `validation/config/profiles/onnxruntime_migraphx_build.yaml`
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
