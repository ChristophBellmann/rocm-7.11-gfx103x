# Repository scope & cleanup guidance
This repository is a focussed fork of [ROCm/TheRock](https://github.com/ROCm/TheRock) that is tuned for the **RX 6700 XT (gfx1031 / RDNA2)**. The goal is to keep everything that is required to build, install, and validate the gfx1031 stack, while removing or sidelining upstream content that is no longer needed.

## Purpose of this fork
- Provide a native `gfx1031` target so the ROCm/LLVM math libraries and HIP runtimes compile without HSA overrides.
- Supply low‑memory-friendly build scripts (`build_low_memory.sh`, `build_enable_math_clients.sh`) and install helpers (`install_systemwide.sh`, `install_to_opt_rocm.sh`) so `/opt/rocm` can be populated with the tested binaries.
- Document and automate the AI/LLM workflows (Ollama, LM Studio, llama.cpp, etc.) that rely on the built stack.
- Track known working configurations (`BUILD_SUCCESS_NATIVE_GFX1031.md`, `BUILD_SUCCESS_SYSTEM.md`) so you know when a build/install combo is clean.

## What to keep
- **`cmake/therock_amdgpu_targets.cmake`** and any patches that add the `gfx1031` target or improve target validation.
- **`build/` and `build_low_memory.sh` plus helpers** – the artifacts, release notes, and scripts under `build/`, `build_tools/`, `math-libs/`, etc., are the actual outputs that install into `/opt/rocm`.
- **`docs/`, `scripts/`, and `COMMANDS_CHEATSHEET.md` / `QUICK_COMMANDS.md`** – these are the primary guides used by the maintainer to rebuild/install and to operate AI tooling.
- **`install_*` scripts and `BUILD_SUCCESS*.md`** – these describe the reproducible installation flow and serve as reference.
- **`tests/`, `third-party/`, `rocm-libraries/` etc.** – they are either upstream sources or part of the build output; keep them unless you understand a clean way to refactor further.

## What can be pruned or archived
- The generic Ubuntu/Windows installation instructions from the upstream `README.md` are already covered by the official repo. Move any redundant sections into `docs/LEGACY_UPSTREAM.md` or replace them with pointers to `README_CUSTOM.md`.
- Any files copied verbatim from upstream that do not contain gfx1031-specific logic can safely be replaced by a short note referring to the upstream source.
- Keep this fork focused on what is actively used: building gfx1031, installing `/opt/rocm`, and wiring up AI tools.

## Archived upstream docs
The following directories were moved to `docs/upstream/` for reference only; they are not part of the streamlined workflow. Treat them as read-only archives for historical/contextual information:

- `docs/upstream/design` (ROCm/TheRock design notes)
- `docs/upstream/development` (general development manual)
- `docs/upstream/packaging` (packaging how-tos)
- `docs/upstream/rfcs` (design RFCs and discussions)
- `docs/upstream/environment_setup_guide.md` (environment setup instructions)

When you need the upstream guidance, refer to `docs/upstream/<name>/README.md` or open the files directly, but avoid editing them unless you are also syncing with the official upstream repo.

## Cleanup workflow suggestion
1. Use `git status` / `git diff` against `../rocm_repo` to spot files that diverge only in comments or repeated instructions.
2. If a document is mostly upstream content, replace it with a short bridge document (like `README.md` → this repo overview) that clarifies where to look for the real instructions.
3. Keep the custom scripts and docs in their current locations; they are the “trusted” interface for the gfx1031 stack.
4. Before deleting anything, note the file path here and verify that no build/install instructions reference it.

With these guardrails, you can safely “clean up” the repo while preserving everything the gfx1031 workflow needs.
