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
  - for ORT real-model diagnostics, stage model files as local copies under `validation/workspace/cache/models/`:
    do not symlink into consumer repos
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
  - if a workload step force-reinstalls packages into a shared runtime venv (for example the ONNX Runtime ROCm Piper steps), do not treat parallel runs as authoritative reference runs
  - when the goal is to expose a real downstream runtime bug, prefer a CPU-reference + ROCm pair so the failure is attributed to the ROCm stack rather than to the model fixture
  - when the goal is performance triage for a real-model workload, keep the benchmark as an explicit profile/step under the same public `validation/validate.py` surface:
    - do not add a new public wrapper under `validation/scripts/`
    - keep in-tree and promoted performance runs separate via explicit profiles
    - keep CPU and ROCm measurements on the same staged model / same fixtures / same seed
    - if the benchmark exposes output-shape or semantic drift, treat that first as a correctness/stability issue, not as a pure speed result
    - if repeated-run stability itself is under question, add an explicit repeated-run probe profile instead of overloading the main correctness profile
  - for Piper/ORT real-model triage, keep the repro deterministic:
    - explicit staged model path
    - explicit real `ids` fixture
    - `ort.set_seed(0)` and `numpy.random.seed(0)`
  - if the current best diagnosis depends on `ORT_ROCM_FORCE_CPU_OPS`, `ORT_ROCM_FORCE_CPU_NODES`, or `ORT_ROCM_FORCE_CPU_OP_NODES`, record the exact override and whether it is only a diagnostic workaround or already the accepted stack fix
- For promoted-system changes:
  - verify the promoted profile succeeds
  - verify runtime linkage/prefix points to `/opt/rocm` where applicable
  - if a benchmark profile exists for the same workload, run that separately after the functional promoted profile, not instead of it

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
  - for Piper/ORT shape mismatches, also include the per-run shape-diagnostic artifacts under:
    - `validation/workspace/runs/<run_id>/artifacts/onnxruntime_tts_shape_diag_<case>_current.json`
    - `validation/workspace/runs/<run_id>/artifacts/onnxruntime_tts_shape_diag_<case>_nofast.json`
    - `validation/workspace/runs/<run_id>/artifacts/onnxruntime_tts_shape_diag_<case>_<mode>_exact_chain/dp_shape_cast.report.json`
  - for ORT wheel packaging issues, include whether the wheel-staged `onnxruntime/capi/libonnxruntime_providers_rocm.so` matches `Release/libonnxruntime_providers_rocm.so`
  - for ORT/MIOpen workspace issues, include whether `miopen_conv_use_max_workspace` still kept the search workspace at or above `AlgoSearchWorkspaceSize` (32 MiB floor in the current gfx1031 workflow)
  - for ORT/Piper correctness issues, explicitly separate:
    - diagnostic CPU fallback overrides
    - GPU-only kernel findings
  - for ORT/Piper performance issues, explicitly separate:
    - session creation cost
    - first-run cost
    - repeated-run cost
    - whether repeated runs remain semantically stable on ROCm
  - if the same ORT/Piper benchmark behavior reproduces both in-tree and under the matching `*_promoted` profile, treat it as a stack/runtime behavior issue first, not as a promote-path packaging issue
  - if `ORT_DISABLE_ALL` or `ORT_ROCM_DISABLE_FAST_REDUCTION=1` only shifts the iteration at which repeated-run drift appears, record that as a narrowing result, not as a fix
  - for repeated-run ORT/Piper diagnosis, prefer extending the existing internal helper under `validation/src/steps/workloads/onnxruntime/piper_tts_debug.py` over introducing a new public wrapper
  - when repeated-run traces of earlier and later tensors disagree, record the bracket explicitly as the current fault window instead of attributing the bug to the first late tensor that visibly explodes
  - if the diagnosis used any of these debug hooks, record them explicitly:
    - `ORT_ROCM_FORCE_CPU_OP_NODES`
    - `ORT_ROCM_FORCE_CPU_OP_EXACT_NODES`
    - `ORT_ROCM_DISABLE_FAST_REDUCTION`
  - current March 2026 Piper/TTS stack status:
    - the primary investigation site is `validation/`, not the consumer repo
    - the real Piper failure now reproduces directly in `validation/` with a
      staged model, a real-case fixture, and a fixed seed
    - for deterministic Piper work, freeze `RandomNormalLike` nodes to dynamic
      zero tensors so the probe follows real ROCm bugs instead of
      provider-local RNG drift
    - the TTS validation is only green if the final ROCm output matches the CPU
      reference within the configured tolerance; exception-free ROCm execution
      alone is not sufficient
    - the isolated `dp_shape_cast` mini-repro is currently green on ROCm when
      fed CPU-captured boundary tensors; the active shape-path bug is upstream
      of that exact `/dp/Split -> ... -> /Cast` chain
    - the first currently proven upstream boundary mismatch feeding the shape
      chain is `/dp/flows.0/Mul_1_output_0`
    - the smaller `/dp/flows.0/Sub -> /dp/flows.0/Mul -> /dp/flows.0/Mul_1`
      mini-repro is also green with CPU-captured boundary tensors; the first
      currently proven non-constant upstream mismatch for that path is
      `/dp/flows.3/Mul_35_output_0`
    - a standalone deterministic profile now exists:
      - `onnxruntime_in_tree_tts_flow_probe`
      - it runs the internal probe helper as its own validation step
      - it freezes Piper `RandomNormalLike` nodes to dynamic zero tensors by
        default
    - current deterministic branch probing above that point still localizes one
      step further to `/dp/flows.3/Split_output_0`, fed from
      `/dp/flows.4/Slice_output_0`
    - the isolated deterministic `dp_flow3_split_path` mini-repro is green on
      ROCm when it is fed CPU-captured boundary tensors; its first upstream
      boundary mismatch is `/dp/flows.5/Mul_35_output_0`
    - forcing `miopen_conv_use_max_workspace=0` versus `=1` in the current
      staged-model debug repro does not change the first mismatching tensor,
      the `Cast` explosion (`2992`), or the final blown-up output length
    - the current workspace-warning family still reports `provided ... size:
      33554432` in both runs; the ORT ROCm `ConvTranspose` algo-search path
      also still hard-codes the 32 MiB search buffer
    - the March 2026 green TTS validation path now depends on two fixes:
      - the ORT ROCm workspace-search fix in the provider
      - deterministic freezing of Piper `RandomNormalLike` nodes in validation
    - the last large post-workspace mismatch was traced to the second
      `/RandomNormalLike` node in the `/flow` branch, not to a remaining
      decoder/conv kernel correctness bug
    - one confirmed bug family is ROCm fast reduction in the encoder
      normalization path; `ORT_ROCM_DISABLE_FAST_REDUCTION=1` is diagnostic only
      and must not be treated as the solution
    - a second independent non-reduction bug remains in repeated local
      `dp/flows.7` ramp paths even with fast reduction disabled
    - the earlier `flow7` Mul suspicion is now downgraded:
      - `/dp/flows.7/Mul_10`
      - `/dp/flows.7/Mul_16`
      still fail in the full graph, but an exact isolated `flow7` gate
      mini-repro is green on ROCm when it is fed CPU-captured boundary tensors
    - that means the faulty `Mul_10`/`Mul_16` values are downstream symptoms,
      not a proven bare Mul kernel bug
    - current deterministic upstream path above those gates:
      - the first stable full-graph boundary mismatch for the local logits path
        is `/dp/flows.7/Transpose_output_0`
      - targeted full-graph probes above that point show earlier deterministic
        divergences already at:
        - `/dp/flows.7/convs/Mul_15_output_0`
        - `/dp/flows.7/pre/Conv_output_0`
        - `/dp/Mul_output_0`
      - the deterministic `/dp/RandomNormalLike -> /Gather_2 -> /dp/Mul_1 ->
        /dp/flows.8/Slice -> /dp/flows.7/Split` path is green, so that input
        path is no longer treated as the primary deterministic fault site
    - exact isolated mini-repros from the real Piper tensors are now required
      when a node looks suspicious:
      - if the isolated mini-repro is green and the full graph is red, record
        that as topology-/lifetime-/execution-order-specific ROCm EP behavior
    - exact node-local CPU forcing can still be used to narrow the fault, but
      must be recorded as diagnostic only and must not be committed as the fix
    - for the March 2026 repeated-run ROCm drift, the lightest currently
      reliable narrowing artifact is:
      - `validation/workspace/debug/piper_steady_repeat_maskline_trace.json`
    - current repeated-run bracket on `mogli`, ROCm, `ORT_ENABLE_ALL`,
      `miopen_conv_use_max_workspace=1`:
      - `/dp/flows.7/Softmax_1_output_0` still stays in the expected family
      - `/dp/flows.7/Pad_2_output_0` is the first currently visible tensor that
        flips to the alternate family on the next repeated run
      - `/dp/flows.7/ScatterND_4_output_0`,
        `/dp/flows.7/ScatterND_7_output_0`, and
        `/dp/flows.7/Cast_16_output_0` follow immediately after
    - heavier bridge traces around `/dp/flows.7/Add_20_output_0` can perturb
      VRAM state enough to trigger separate allocator or MIOpen failures; do
      not treat those heavier traces as the primary narrowing evidence unless
      they reproduce serially without the extra memory fault
    - if two repeated-run traces over the same neighborhood disagree depending
      on which outputs are kept alive in the debug model, record that as
      stronger evidence for topology-/lifetime-/execution-order-sensitive ROCm
      EP behavior:
      - current example:
        - `validation/workspace/debug/piper_steady_repeat_maskline_trace.json`
          shows `Pad_2` drifting early while `Softmax_1` still looks stable
        - `validation/workspace/debug/piper_steady_repeat_cumsum1line_trace.json`
          keeps `GatherND_3 -> ... -> Pad_2` stable through iter 2 and then
          jumps straight to the later `/Reshape_1` failure
    - for ORT/Piper performance triage, if the benchmark and ORT profile agree
      that almost all node time sits in `ROCMExecutionProvider` `Conv` while
      `Memcpy*` time remains tiny, record that explicitly as a ROCm conv-path
      performance issue rather than summarizing it as transfer overhead or CPU
      fallback
