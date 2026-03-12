`external-builds/` contains in-tree helper tooling that comes from or aligns with
the upstream TheRock repository.

It is not the source of truth for the custom gfx1031 wheel release flow used in
this repository.

Use it like this:
- Keep `external-builds/` for upstream/in-tree build helpers and historical CI
  integration.
- Use `validation/workspace/cache/git/<framework-fork>/tools/rocm_release/` for
  custom wheel build/promote flows that target this custom ROCm stack.
- Use `validation/validate.py` to validate those wheels against in-tree ROCm and
  promoted `/opt/rocm` installs.

Practical boundary:
- `external-builds/pytorch/` can still be used for heavy in-tree source-build
  experiments or compatibility checks.
- The promoted custom PyTorch family for this repo is owned by the fork
  `rocm-7.11-pytorch-gfx103x`, not by `external-builds/pytorch/`.

Do not introduce new custom wheel promote logic under `external-builds/`.
