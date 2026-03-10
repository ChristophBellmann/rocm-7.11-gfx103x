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
