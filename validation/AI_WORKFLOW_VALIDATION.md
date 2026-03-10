# AI Workflow (Validation Suite)
_This document is a concrete workflow for maintaining and extending the validation suite under `validation/`._

It follows the structure of the repository-agnostic template in `AI_WORKFLOW.md`,
but is scoped specifically to **validation** (in-tree ROCm, plus optional third‑party app checks).

---

## Situation

- **Desired outcome:** keep `validation/validate.py` + `validation/src/` a reliable “usability proof” of the in-tree build, and (optionally) a practical integration test bed.
- **Constraints:**
  - Must run against `<builddir>/dist/rocm` (no `/opt/rocm` assumptions).
  - Downloads/builds must be **explicitly confirmed** (Y/n) and bounded; avoid surprise multi‑GB downloads.
  - Prefer user-space install locations under `validation/workspace/`.
  - Keep test numbering stable once published (so notes can reference IDs).
- **Autonomy boundary:** do not install system packages, enable systemd services, modify `/usr`, or download huge artifacts without prompting and documenting size/impact.

---

## Orientation

- Check repo status and `validation/` tree.
- Confirm where ROCm is expected (`build-stage2/dist/rocm` etc.).
- Identify what is already cached under `validation/workspace/cache/`.
- If a check fails, re-run with `--log` and inspect `validation/workspace/runs/<run_id>/logs/`.

---

## Intent

- Make the smallest changes that:
  - Keep core in-tree ROCm checks reliable (HIP compile+run, rocminfo, benches).
  - Add third-party checks in a way that is:
    - opt-in via Y/n prompt
    - cached
    - reproducible (fixed inputs, stable versions where feasible)
- Do **not** refactor unrelated build scripts or change build behavior outside validation work.

---

## Execution

- Add/modify steps in `validation/src/steps/`:
  - Use `SKIP` for missing prerequisites; use `FAIL` for real malfunctions.
  - Make expected runtime realistic and hardware-aware (gfx1031 / RX 6700 XT).
- Keep downloads/builds in user-space:
  - `validation/workspace/cache/` for downloads/clones
  - `validation/workspace/builds/` for build directories
- Ensure every third-party step can be disabled via `--profile quick` or `--no-downloads`.

---

## Validation

- Always run a small smoke subset first:
  - `python3 validation/validate.py --profile quick --log`
- If adding a third-party check:
  - Confirm it respects `validation/workspace/` for caches/build outputs.
  - Run with `python3 validation/validate.py --yes --log`.
- Clearly label what was actually verified vs. planned.

---

## Knowledge Capture

- Update `validation/README.md` when:
  - new checks are added
  - defaults or prompts change
  - cache/build locations change
- Record new large dependencies (and approximate sizes) in `validation/README.md`.
- Prefer keeping configuration centralized in the validation code; avoid duplicating logic in README.

---

## Change Recording

- Keep commits focused:
  - “validation: add MFEM HIP check”
  - “validation: fix Ollama download URL”
- Ensure generated files and caches stay gitignored (`validation/workspace/`, `validation/.venv/`).

---

## Resulting State

- Core ROCm usability checks run successfully from a clean clone after building Stage‑2.
- Third-party checks:
  - prompt before downloads/builds
  - can be disabled
  - leave artifacts only under `validation/workspace/`

---

## Handover

- First command to run:
  - `python3 validation/validate.py --profile quick`
- If third-party checks are desired:
  - `python3 validation/validate.py --yes`
- If a check fails: re-run with `--log` and include the `validation/workspace/runs/<run_id>/logs/*.log` + return code.
