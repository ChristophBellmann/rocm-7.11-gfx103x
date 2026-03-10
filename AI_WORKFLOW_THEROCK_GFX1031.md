# AI Workflow (TheRock_gfx1031)

This document defines the repo-specific workflow for AI-assisted work in
`TheRock_gfx1031`.

It is intentionally concrete. The goal is not to invent process during the
session, but to follow the already established build, validation, promotion, and
documentation model of this repo.

## 1. Repository Role

`TheRock_gfx1031` is the central custom ROCm repo for gfx1031.

Its responsibilities are:

- build and maintain the custom ROCm stack
- validate that stack in-tree via `validation/`
- validate promoted system state where needed
- document the boundary between system artifacts and git-only fixes

It is **not** the source-of-truth repo for framework wheel packaging logic.
That lives in the framework forks:

- `rocm-7.11-pytorch-gfx103x`
- `rocm-7.11-onnxruntime-gfx103x`
- `rocm-7.11-tensorflow-gfx103x`

## 2. Operating Model

The expected lifecycle is:

1. build custom ROCm in-tree
2. validate the in-tree result via `validation/`
3. build framework-specific custom wheels in the matching fork
4. validate those wheels against in-tree ROCm
5. promote them to `/opt/rocm/wheels/...`
6. validate the promoted system state again
7. let consumer repos use only the promoted artifacts

Stable system-consumption points are the `*-current.whl` aliases under
`/opt/rocm/wheels/...`.

## 3. First Inspection Pass

Before changing anything:

- check `git status`, current branch, and last relevant commits
- identify whether the request concerns:
  - core TheRock build/config
  - validation
  - framework packaging/promotion
  - documentation only
- confirm whether the work belongs in this repo or in one of the framework forks
- inspect the current promoted artifact model in `CUSTOM_ROCM_ARTIFACTS.md`
- inspect the current user-facing flow in `README.md`

## 4. Where Changes Belong

- Core ROCm build/config/test/monitor changes belong here.
- Validation profiles, validation runner logic, and validation docs belong here.
- Framework-specific wheel build/promote helpers do **not** belong here; they belong in the framework forks.
- Consumer-specific behavior does **not** belong here; it belongs in the consumer repo.

If a change mixes these layers, split it before implementing.

## 5. Build and Validation Rules

- Prefer the existing repo entrypoints and workflow:
  - `build_gfx1031.sh`
  - `install_to_opt.sh`
  - `validation/validate.py`
- Keep in-tree and promoted validation clearly separated.
- Do not silently mix `/opt/rocm` and `<build>/dist/rocm`.
- When promoted validation is the goal, make the prefix expectation explicit and verify it.
- Keep workload/cache/build artifacts inside the repo-controlled locations.

## 6. Documentation Synchronization

Whenever behavior changes, update the owning documents in the same change:

- `README.md` for user-facing flow
- `CUSTOM_ROCM_ARTIFACTS.md` for artifact/fix boundaries
- `validation/README.md` for validation commands and profiles
- `validation/AI_WORKFLOW_VALIDATION.md` for validation operating rules
- `validation/ANWEISUNG_STRUKTUR.md` for structure rules
- `BUILD_EXPERIENCE_NOTES.md` for concrete lessons learned and recovery notes

Do not let `AGENTS.md` turn into an operational manual.

## 7. Git and CI Rules

- Keep commits focused and architecture-aware.
- If workflow files are changed in personal framework forks:
  - automatic Actions should remain disabled unless there is a specific reason not to
  - HTTPS pushes may fail without `workflow` scope
  - manual SSH push is an acceptable fallback
- Prefer clean branch state before ending a session.
- If a push was done via explicit URL, refresh remote-tracking refs before concluding the repo is still `ahead`.

## 8. End State

A correct end state for work in this repo means:

- the change is in the correct repository layer
- the build/validate/promote separation is preserved
- docs match the actual behavior
- system artifacts and git-only fixes remain clearly separated
- `git status` is clean and synced where the work was completed
