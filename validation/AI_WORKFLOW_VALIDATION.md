# AI Workflow (Validation Suite)
_This document defines the operational workflow for maintaining `validation/` inside `TheRock_gfx1031`._

`validation/` is the post-build validation layer for this repo. It proves that the
custom ROCm stack is usable both:

- in-tree against `<builddir>/dist/rocm`
- and, where explicitly intended, after promotion to `/opt/rocm`

## Situation

- Desired outcome:
  - keep `validation/validate.py` and `validation/src/` as the single public validation surface
  - keep validation reproducible, explicit, and cache-contained under `validation/workspace/`
  - ensure promoted custom wheels are validated again against `/opt/rocm`
- Default assumption:
  - validation targets the in-tree ROCm build
- Explicit exception:
  - profiles ending in `*_promoted` intentionally validate the promoted system state under `/opt/rocm`
- Constraints:
  - no surprise multi-GB downloads
  - no system package installs from validation code
  - no new public user wrappers under `validation/scripts/`
  - keep workload artifacts inside `validation/workspace/`

## Operating Model

- Public entrypoints live only at the root of `validation/`:
  - `validation/validate.py`
  - `validation/doctor.py`
  - `validation/cache_gc.py`
  - `validation/report_open.py`
- `validation/scripts/` is internal bootstrap/launcher code only.
- Validation logic lives under `validation/src/`.
- The normal delivery path for custom framework artifacts is:
  1. build repo-local against custom ROCm
  2. validate repo-local against in-tree ROCm
  3. promote to `/opt/rocm/wheels/...`
  4. validate again with the matching `*_promoted` profile
- Build/promote source of truth for framework wheels lives in the framework forks, not in `validation/` itself:
  - PyTorch family
  - ONNX Runtime
  - TensorFlow

## Execution Rules

- Start with a small smoke run before broad changes:
  - `python3 validation/validate.py --profile quick --log`
- When adding or changing a workload step:
  - keep cache/build output under `validation/workspace/`
  - use `SKIP` for missing prerequisites, `FAIL` for real malfunctions
  - make runtime expectations realistic for gfx1031 / RX 6700 XT
  - for ORT real-model diagnostics (for example Piper TTS), stage external models under `validation/workspace/cache/models/`
  - if a real-model diagnostic exposes a stack bug, keep it as an explicit profile instead of weakening the main smoke path
- When validating promoted artifacts:
  - use the explicit `*_promoted` profiles only
  - confirm the runtime really resolves ROCm from `/opt/rocm`, not from the in-tree build
- `doctor` remains a no-download sanity path.
- If a workflow-specific convenience wrapper is tempting, prefer extending the main profile-driven CLI instead.

## Verification

- Minimum verification for validation changes:
  - help/CLI still works from `validation/*.py`
  - profile references still resolve
  - logs and run artifacts remain under `validation/workspace/runs/`
- For third-party workload changes:
  - verify the workload respects `validation/workspace/cache/` and `validation/workspace/builds/`
  - run at least one targeted validation command with `--log`
  - when the goal is to expose a real downstream runtime bug, prefer a CPU-reference + ROCm pair so the failure is attributed to the ROCm stack rather than to the model fixture
  - for Piper/ORT real-model triage, keep the repro deterministic:
    - explicit staged model path
    - explicit real `ids` fixture
    - `ort.set_seed(0)` and `numpy.random.seed(0)`
  - if the current best diagnosis depends on `ORT_ROCM_FORCE_CPU_OPS`, `ORT_ROCM_FORCE_CPU_NODES`, or `ORT_ROCM_FORCE_CPU_OP_NODES`, record the exact override and whether it is only a diagnostic workaround or already the accepted stack fix
- For promoted-system changes:
  - verify the promoted profile succeeds
  - verify runtime linkage/prefix points to `/opt/rocm` where applicable

## Knowledge Capture

- Update `validation/README.md` when commands, profiles, prompts, or cache/build locations change.
- Update `CUSTOM_ROCM_ARTIFACTS.md` when the boundary between system artifacts and git-only fixes changes.
- Keep structure rules in `validation/ANWEISUNG_STRUKTUR.md` only.
- Do not duplicate operational command detail across `AGENTS.md`, `AI_WORKFLOW_THEROCK_GFX1031.md`, and `validation/README.md`.

## Change Recording

- Keep validation commits focused and behavior-oriented.
- Prefer messages like:
  - `validation: add promoted profile for onnxruntime`
  - `validation: simplify public cli surface`
  - `validation: fix promoted torch wheel consumption`
- If a change removes wrappers, paths, or public commands, update docs in the same change.

## Resulting State

A correct end state for `validation/` means:

- one public CLI surface under `validation/`
- one implementation tree under `validation/src/`
- all runtime artifacts contained in `validation/workspace/`
- in-tree validation remains the default
- promoted validation is explicit, not implicit
- framework-specific build/promote logic stays in the framework forks

## Handover

- First command for basic validation:
  - `python3 validation/validate.py --profile quick`
- First command for promoted artifact validation:
  - `python3 validation/validate.py --profile <name>_promoted --yes --power --log`
- If something fails:
  - rerun with `--log`
  - include `validation/workspace/runs/<run_id>/logs/*.log`
  - state whether the failure is in-tree or promoted-system
  - for Piper/ORT issues, include whether the CPU reference passed, how many `ROCMExecutionProvider` events were seen, and how many MIOpen workspace warnings were emitted
  - for ORT wheel packaging issues, include whether the wheel-staged `onnxruntime/capi/libonnxruntime_providers_rocm.so` matches `Release/libonnxruntime_providers_rocm.so`
  - for ORT/MIOpen workspace issues, include whether `miopen_conv_use_max_workspace` still kept the search workspace at or above `AlgoSearchWorkspaceSize` (32 MiB floor in the current gfx1031 workflow)
  - for ORT/Piper correctness issues, explicitly separate:
    - diagnostic CPU fallback overrides
    - GPU-only kernel findings
  - if the diagnosis used any of these debug hooks, record them explicitly:
    - `ORT_ROCM_FORCE_CPU_OP_NODES`
    - `ORT_ROCM_FORCE_CPU_OP_EXACT_NODES`
    - `ORT_ROCM_DISABLE_FAST_REDUCTION`
  - current March 2026 Piper/TTS stack status:
    - the primary investigation site is `validation/`, not the consumer repo
    - the real Piper failure now reproduces directly in `validation/` with a
      staged model, a real-case fixture, and a fixed seed
    - there is still no accepted final GPU-only fix
    - the current real-graph diagnosis has at least two distinct ROCm-only
      fault families:
      - the original `/Reshape_1` crash path, whose first proven divergence is
        already at `/dp/flows.0/Mul_1_output_0` and `/dp/flows.2/Slice_output_0`,
        with `/dp/Split_output_0` already non-finite on ROCm
      - a separate deterministic frozen `/dp/flows.5` path with corrupted
        `Expand`, `GreaterOrEqual`, `ReduceSum`, `GatherND`, `GatherElements`,
        and `ScatterND` behavior on ROCm
    - exact node-local CPU forcing such as:
      - `ORT_ROCM_FORCE_CPU_OP_EXACT_NODES='Expand@/dp/flows.5/Expand_15'`
      - `ORT_ROCM_FORCE_CPU_OP_EXACT_NODES='Expand@/dp/flows.5/Expand_25'`
      can make the local extracted subgraphs go green, but these are diagnostic
      overrides only
    - `ORT_ROCM_FORCE_CPU_OPS='Expand'` does not make the full real Piper TTS
      validation pass, so `Expand` corruption is real but not the only active
      failure in the graph
    - standalone minimal `Expand` and `Reshape` ONNX models built from the exact
      raw tensor values of the failing Piper subgraphs run correctly on ROCm;
      the current best diagnosis is therefore topology-/partitioning-specific
      ROCm EP behavior, not an isolated scalar-kernel bug
    - isolated ROCm reduction on problematic Piper tensors can still be wrong,
      but the reduction error is now known to be downstream of earlier ROCm-only
      graph corruption in the full TTS path
