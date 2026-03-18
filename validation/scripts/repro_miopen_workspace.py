#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validation_root() -> Path:
    return repo_root() / "validation"


def staged_model_root() -> Path:
    return validation_root() / "workspace" / "cache" / "models" / "onnxruntime_tts"


def default_model_path() -> Path:
    models = sorted(
        (p for p in staged_model_root().glob("*.onnx") if p.is_file() and not p.is_symlink()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not models:
        raise FileNotFoundError(f"no staged local ONNX model in {staged_model_root()}")
    return models[0]


def default_case_file(model_path: Path) -> Path:
    return validation_root() / "fixtures" / "onnxruntime_tts" / f"{model_path.stem}.real_cases.json"


def load_case(case_file: Path, case_label: str) -> dict[str, Any]:
    payload = json.loads(case_file.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    if not cases:
        raise ValueError(f"no cases in {case_file}")
    if case_label:
        for case in cases:
            if str(case.get("label", "")) == case_label:
                return case
        raise ValueError(f"case label not found in {case_file}: {case_label}")
    return cases[0]


def build_feed(case: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    ids = [int(v) for v in case["ids"]]
    return {
        "input": np.asarray([ids], dtype=np.int64),
        "input_lengths": np.asarray([len(ids)], dtype=np.int64),
        "scales": np.asarray(case["scales"], dtype=np.float32),
    }


def rocm_lib_hint() -> str:
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as f:
            libs = sorted(
                {
                    line.strip().split()[5]
                    for line in f
                    if len(line.strip().split()) >= 6
                    and line.strip().split()[5].startswith("/")
                    and ".so" in line.strip().split()[5]
                }
            )
    except Exception:
        return ""
    for needle in ("libamdhip64.so", "libMIOpen.so", "librocblas.so"):
        for path in libs:
            base = os.path.basename(path)
            if base == needle or base.startswith(needle + "."):
                return path
    return ""


def warning_pattern() -> re.Pattern[str]:
    return re.compile(
        r"MIOpen\(HIP\): Warning \[IsEnoughWorkspace\].*?Solver <([^>]+)>, workspace required: (\d+), provided ptr: (\S+) size: (\d+)"
    )


def relevant_env_snapshot() -> dict[str, str]:
    keys = [
        "LD_LIBRARY_PATH",
        "MIOPEN_FIND_MODE",
        "MIOPEN_FIND_ENFORCE",
        "MIOPEN_SEARCH_CUTOFF",
        "MIOPEN_DEBUG_DISABLE_FIND_DB",
        "ORT_ROCM_DISABLE_FAST_REDUCTION",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


def run_child(model_path: Path, case_file: Path, case_label: str, provider_options: dict[str, str], profile_prefix: str) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    case = load_case(case_file, case_label)
    feed = build_feed(case)

    def run_session(providers: list[Any], enable_profiling: bool) -> dict[str, Any]:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if enable_profiling:
            so.enable_profiling = True
            so.profile_file_prefix = profile_prefix
        sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        t0 = time.perf_counter()
        outputs = sess.run(None, feed)
        t1 = time.perf_counter()
        profile_path = sess.end_profiling() if enable_profiling else ""
        arr = np.asarray(outputs[0])
        out = {
            "providers": sess.get_providers(),
            "provider_options": sess.get_provider_options() if hasattr(sess, "get_provider_options") else {},
            "elapsed_ms": (t1 - t0) * 1000.0,
            "output_shape": list(arr.shape),
            "output_dtype": str(arr.dtype),
            "output_sample": arr.reshape(-1)[:16].tolist(),
            "profile_path": profile_path,
        }
        return out

    cpu = run_session(["CPUExecutionProvider"], enable_profiling=False)
    rocm_providers: list[Any]
    if provider_options:
        rocm_providers = [("ROCMExecutionProvider", provider_options), "CPUExecutionProvider"]
    else:
        rocm_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
    rocm = run_session(rocm_providers, enable_profiling=True)

    result: dict[str, Any] = {
        "model": str(model_path),
        "case_file": str(case_file),
        "case_label": str(case.get("label", "")),
        "scales": list(case["scales"]),
        "env": relevant_env_snapshot(),
        "requested_rocm_provider_options": provider_options,
        "cpu": cpu,
        "rocm": rocm,
        "hip_lib": rocm_lib_hint(),
    }

    cpu_shape = tuple(cpu["output_shape"])
    rocm_shape = tuple(rocm["output_shape"])
    result["compare"] = {
        "shape_equal": cpu_shape == rocm_shape,
        "cpu_shape": list(cpu_shape),
        "rocm_shape": list(rocm_shape),
    }
    if cpu_shape == rocm_shape:
        cpu_arr = np.asarray(cpu["output_sample"], dtype=np.float32)
        rocm_arr = np.asarray(rocm["output_sample"], dtype=np.float32)
        if cpu_arr.shape == rocm_arr.shape:
            diff = np.abs(cpu_arr - rocm_arr)
            result["compare"]["sample_max_abs_diff"] = float(np.max(diff)) if diff.size else 0.0
            result["compare"]["sample_mean_abs_diff"] = float(np.mean(diff)) if diff.size else 0.0
    return result


def profile_op_counts(profile_path: str) -> dict[str, int]:
    if not profile_path:
        return {}
    path = Path(profile_path)
    if not path.is_file():
        return {}
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counts: Counter[str] = Counter()
    for ev in events:
        args = ev.get("args") or {}
        if args.get("provider") != "ROCMExecutionProvider":
            continue
        op_name = str(args.get("op_name") or "").strip()
        if op_name:
            counts[op_name] += 1
    return dict(sorted(counts.items()))


def parse_warning_entries(text: str) -> list[dict[str, Any]]:
    entries = []
    for match in warning_pattern().finditer(text):
        solver, required, ptr, size = match.groups()
        entries.append(
            {
                "solver": solver,
                "required": int(required),
                "ptr": ptr,
                "provided_size": int(size),
            }
        )
    return entries


def matrix_cases(include_search_enforce: bool) -> list[dict[str, Any]]:
    cases = [
        {"id": "A_baseline", "label": "Baseline", "provider_options": {}, "env": {}},
        {"id": "B_big_gpu_mem_limit", "label": "Huge gpu_mem_limit", "provider_options": {"gpu_mem_limit": str(1 << 40)}, "env": {}},
        {"id": "C_same_as_requested", "label": "Arena same-as-requested", "provider_options": {"arena_extend_strategy": "kSameAsRequested"}, "env": {}},
        {"id": "D_find_normal", "label": "MIOPEN_FIND_MODE=NORMAL", "provider_options": {}, "env": {"MIOPEN_FIND_MODE": "NORMAL"}},
        {"id": "E_find_fast", "label": "MIOPEN_FIND_MODE=FAST", "provider_options": {}, "env": {"MIOPEN_FIND_MODE": "FAST"}},
        {"id": "F_find_hybrid", "label": "MIOPEN_FIND_MODE=HYBRID", "provider_options": {}, "env": {"MIOPEN_FIND_MODE": "HYBRID"}},
        {"id": "H_max_workspace_off", "label": "miopen_conv_use_max_workspace=0", "provider_options": {"miopen_conv_use_max_workspace": "0"}, "env": {}},
        {"id": "I_max_workspace_on", "label": "miopen_conv_use_max_workspace=1", "provider_options": {"miopen_conv_use_max_workspace": "1"}, "env": {}},
    ]
    if include_search_enforce:
        cases.insert(
            6,
            {
                "id": "G_find_fast_search",
                "label": "FAST + FIND_ENFORCE=SEARCH",
                "provider_options": {},
                "env": {"MIOPEN_FIND_MODE": "FAST", "MIOPEN_FIND_ENFORCE": "SEARCH"},
            },
        )
    return cases


def run_matrix(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for case in matrix_cases(bool(args.include_search_enforce)):
        child_env = os.environ.copy()
        child_env.update(case["env"])
        profile_dir = out_dir / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--model",
            str(args.model),
            "--case-file",
            str(args.case_file),
            "--case-label",
            args.case_label,
            "--profile-prefix",
            str(profile_dir / case["id"]),
            "--provider-options-json",
            json.dumps(case["provider_options"], sort_keys=True),
        ]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root()),
                env=child_env,
                text=True,
                capture_output=True,
                timeout=int(args.child_timeout_s),
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(
                exc.cmd,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )
            timed_out = True
        t1 = time.perf_counter()
        combined = proc.stdout + "\n" + proc.stderr
        child_json = None
        for line in proc.stdout.splitlines():
            if line.startswith("REPRO_JSON="):
                child_json = json.loads(line.split("=", 1)[1])
                break
        warning_entries = parse_warning_entries(combined)
        warnings_summary = {
            "count": len(warning_entries),
            "max_required": max((e["required"] for e in warning_entries), default=0),
            "provided_sizes": sorted({e["provided_size"] for e in warning_entries}),
            "solvers": sorted({e["solver"] for e in warning_entries}),
            "entries": warning_entries[:40],
        }
        entry = {
            "id": case["id"],
            "label": case["label"],
            "provider_options": case["provider_options"],
            "env_overrides": case["env"],
            "rc": proc.returncode,
            "timed_out": timed_out,
            "elapsed_ms": (t1 - t0) * 1000.0,
            "warning": warnings_summary,
            "stdout_tail": proc.stdout.splitlines()[-20:],
            "stderr_tail": proc.stderr.splitlines()[-40:],
        }
        if child_json is not None:
            rocm_profile = ((child_json.get("rocm") or {}).get("profile_path") or "").strip()
            child_json["rocm_profile_op_counts"] = profile_op_counts(rocm_profile)
            entry["result"] = child_json
        results.append(entry)

    baseline = next((r for r in results if r["id"] == "A_baseline"), None)
    if baseline is None:
        print("missing baseline result", file=sys.stderr)
        return 2
    if baseline["rc"] != 0:
        print("baseline child failed", file=sys.stderr)
        return 2
    if baseline["warning"]["count"] == 0:
        print("baseline did not reproduce any IsEnoughWorkspace warning", file=sys.stderr)
        return 2

    summary = analyze_results(results)
    payload = {
        "model": str(args.model),
        "case_file": str(args.case_file),
        "case_label": args.case_label,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "analysis": summary,
    }
    (out_dir / "miopen_workspace_matrix.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "miopen_workspace_report.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(r for r in results if r["id"] == "A_baseline")

    def warning_signature(entry: dict[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...]]:
        return (
            tuple(entry["warning"].get("provided_sizes") or []),
            tuple(entry["warning"].get("solvers") or []),
        )

    ort_cases = [r for r in results if r["id"] in {"B_big_gpu_mem_limit", "C_same_as_requested"}]
    miopen_cases = [r for r in results if r["id"] in {"D_find_normal", "E_find_fast", "F_find_hybrid", "G_find_fast_search"}]

    same_provided_size_ort = all(r["warning"].get("provided_sizes") == baseline["warning"].get("provided_sizes") for r in ort_cases)
    same_warning_presence_ort = all((r["warning"].get("count", 0) > 0) == (baseline["warning"].get("count", 0) > 0) for r in ort_cases)
    same_provided_size_miopen = all(
        (r["warning"].get("provided_sizes") == baseline["warning"].get("provided_sizes"))
        for r in miopen_cases
        if r["warning"].get("count", 0) > 0
    )
    miopen_fast = next((r for r in miopen_cases if r["id"] == "E_find_fast"), None)
    miopen_only_cause = False
    miopen_evidence = "warnings persist with identical provided size whenever the search path is active"
    if miopen_fast is not None:
        fast_shape = (((miopen_fast.get("result") or {}).get("rocm") or {}).get("output_shape"))
        base_shape = (((baseline.get("result") or {}).get("rocm") or {}).get("output_shape"))
        if miopen_fast["warning"].get("count", 0) == 0 and fast_shape != base_shape:
            miopen_only_cause = False
            miopen_evidence = "FAST suppresses the warning path, but the model output still changes and remains incorrect"
        elif not same_provided_size_miopen:
            miopen_only_cause = True
            miopen_evidence = "warning signature changed under MIOpen find modes"

    analysis = {
        "baseline_warning_count": baseline["warning"]["count"],
        "baseline_provided_sizes": baseline["warning"]["provided_sizes"],
        "baseline_max_required": baseline["warning"]["max_required"],
        "baseline_solvers": baseline["warning"]["solvers"],
        "ort_memory_limit_dependency": {
            "depends": not (same_provided_size_ort and same_warning_presence_ort),
            "evidence": "no dependency found" if same_provided_size_ort and same_warning_presence_ort else "warning signature changed under ORT memory options",
        },
        "miopen_find_mode_dependency": {
            "depends_only_on_find_mode": miopen_only_cause,
            "evidence": miopen_evidence,
        },
        "model_or_operator_path_dependency": {
            "supported": True,
            "evidence": "warnings are emitted in ORT ROCm Conv/ConvTranspose algo-search code paths that use the model's concrete shapes",
            "source_points": [
                "validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv.h:173",
                "validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv.cc:287",
                "validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv_transpose.cc:131",
            ],
        },
    }
    return analysis


def render_report(payload: dict[str, Any]) -> str:
    results = payload["results"]
    analysis = payload["analysis"]
    lines = [
        "# MIOpen Workspace Investigation",
        "",
        f"- Model: `{payload['model']}`",
        f"- Case file: `{payload['case_file']}`",
        f"- Case label: `{payload['case_label']}`",
        "",
        "## Fundstellen im Repo",
        "",
        "- `validation/src/steps/plan.py`: erstellt ORT-Sessions mit `providers=['ROCMExecutionProvider', 'CPUExecutionProvider']` ohne explizite GPU-Memory-Limits.",
        "- `validation/src/steps/workloads/onnxruntime/piper_tts_debug.py`: erlaubt ROCm-Provider-Optionen als `key=value` und nutzt denselben EP-Pfad.",
        "- `validation/workspace/cache/git/onnxruntime_rocm711/include/onnxruntime/core/session/onnxruntime_c_api.h`: `OrtROCMProviderOptions` defaultet `gpu_mem_limit=SIZE_MAX`.",
        "- `validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/rocm_execution_provider_info.h`: `miopen_conv_use_max_workspace` defaultet auf `true`.",
        "- `validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv.h`: `AlgoSearchWorkspaceSize = 32 * 1024 * 1024`.",
        "- `validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv.cc`: Conv-Search startet bei 32 MiB und wächst nur, wenn `GetMaxWorkspaceSize()` größer meldet.",
        "- `validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv_transpose.cc`: ConvTranspose-Search benutzt direkt `AlgoSearchWorkspaceSize`.",
        "",
        "## Testmatrix",
        "",
        "| Case | Warning? | Provided size(s) | Max required | Solvers | ROCm shape | Sample diff |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in results:
        result = entry.get("result") or {}
        compare = result.get("compare") or {}
        sample_diff = compare.get("sample_max_abs_diff")
        lines.append(
            f"| {entry['id']} | {'yes' if entry['warning']['count'] else 'no'} | "
            f"`{','.join(str(v) for v in entry['warning']['provided_sizes']) or '-'}` | "
            f"`{entry['warning']['max_required']}` | "
            f"`{','.join(entry['warning']['solvers']) or '-'}` | "
            f"`{(result.get('rocm') or {}).get('output_shape', '-')}` | "
            f"`{sample_diff if sample_diff is not None else '-'}` |"
        )
    lines.extend(
        [
            "",
            "## Schlussfolgerung",
            "",
            f"- ORT-Memory-Limit: {'ja' if analysis['ort_memory_limit_dependency']['depends'] else 'nein'}; {analysis['ort_memory_limit_dependency']['evidence']}.",
            f"- Nur MIOpen-Find-Modus: {'ja' if analysis['miopen_find_mode_dependency']['depends_only_on_find_mode'] else 'nein'}; {analysis['miopen_find_mode_dependency']['evidence']}.",
            "- Modell-/Operator-Pfad: ja; die Warnung entsteht in den ORT-ROCm-Conv-/ConvTranspose-Search-Pfaden für die konkreten Piper-Formen.",
            "",
            "## Empfohlener Fix",
            "",
            "- Noch kein automatischer Code-Fix in `validation`: erst nach belastbarer Evidenz im ORT-ROCm-Code patchen.",
            "- Wahrscheinlich relevante Codeänderung im ORT-Fork: den ConvTranspose-Search nicht auf die feste 32-MiB-`AlgoSearchWorkspaceSize` begrenzen oder diesen Pfad konfigurierbar machen.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reproduce and classify MIOpen workspace warnings for ORT ROCm.")
    ap.add_argument("--child", action="store_true", help="Internal child mode for one isolated run")
    ap.add_argument("--model", default=str(default_model_path()))
    ap.add_argument("--case-file", default="")
    ap.add_argument("--case-label", default="mogli")
    ap.add_argument("--out-dir", default=str(validation_root() / "workspace" / "debug" / "miopen_workspace_matrix"))
    ap.add_argument("--provider-options-json", default="{}")
    ap.add_argument("--profile-prefix", default="")
    ap.add_argument("--include-search-enforce", action="store_true", help="Include optional FAST + FIND_ENFORCE=SEARCH case")
    ap.add_argument("--child-timeout-s", type=int, default=120, help="Timeout per isolated child run in matrix mode")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    case_file = Path(args.case_file).expanduser().resolve() if args.case_file else default_case_file(model_path)
    if args.child:
        provider_options = json.loads(args.provider_options_json or "{}")
        profile_prefix = args.profile_prefix or tempfile.mktemp(prefix="miopen-workspace-profile-")
        result = run_child(model_path, case_file, args.case_label, provider_options, profile_prefix)
        print("REPRO_JSON=" + json.dumps(result, sort_keys=True))
        return 0
    return run_matrix(argparse.Namespace(**{**vars(args), "model": model_path, "case_file": case_file}))


if __name__ == "__main__":
    raise SystemExit(main())
