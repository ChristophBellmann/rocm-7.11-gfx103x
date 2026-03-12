from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto
from onnx import numpy_helper
from onnx import utils
import onnxruntime as ort


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def validation_root() -> Path:
    return repo_root() / "validation"


def default_model_path() -> Path:
    model_dir = validation_root() / "workspace" / "cache" / "models" / "onnxruntime_tts"
    models = sorted(model_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        raise FileNotFoundError(f"no staged Piper ONNX model in {model_dir}")
    return models[0]


def default_case_file(model_path: Path) -> Path:
    return validation_root() / "fixtures" / "onnxruntime_tts" / f"{model_path.stem}.real_cases.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Internal debug helper for real Piper ONNX TTS on ORT CPU vs ROCm.")
    ap.add_argument("--model", default=str(default_model_path()), help="Path to staged Piper ONNX model")
    ap.add_argument("--case-file", default="", help="Path to real-case JSON fixture")
    ap.add_argument("--case-label", default="", help="Only run the matching case label; default is the first case")
    ap.add_argument("--seed", type=int, default=0, help="Seed for ORT and NumPy")
    ap.add_argument(
        "--disable-fast-reduction",
        action="store_true",
        help="Set ORT_ROCM_DISABLE_FAST_REDUCTION=1 for the ROCm run",
    )
    ap.add_argument(
        "--output",
        action="append",
        dest="outputs",
        default=[],
        help="Explicit intermediate tensor output to compare; can be passed multiple times",
    )
    ap.add_argument(
        "--node-prefix",
        action="append",
        dest="node_prefixes",
        default=[],
        help="Collect all outputs of nodes whose name starts with this prefix; can be passed multiple times",
    )
    ap.add_argument(
        "--node-name",
        action="append",
        dest="node_names",
        default=[],
        help="Collect all outputs of an exact node name; can be passed multiple times",
    )
    ap.add_argument("--max-report", type=int, default=30, help="How many mismatches to include in the JSON report")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="If >0, split requested outputs into deterministic chunks and run one session per chunk",
    )
    ap.add_argument(
        "--graph-mode",
        choices=["full", "extract"],
        default="full",
        help="Run against the full model with appended outputs (preferred) or an extracted submodel",
    )
    ap.add_argument(
        "--opt-level",
        choices=["disable", "basic", "extended", "all"],
        default="disable",
        help="ORT graph optimization level for the debug session",
    )
    ap.add_argument(
        "--freeze-dp-random-zeros",
        action="store_true",
        help="Replace /dp/RandomNormalLike with a zero-valued Constant for deterministic Piper debug runs",
    )
    ap.add_argument(
        "--extract-per-output",
        action="store_true",
        help="For extract mode, build one extracted submodel per requested output instead of a shared submodel",
    )
    ap.add_argument("--out-json", default="", help="Optional path for the JSON result")
    return ap.parse_args()


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


def build_feed(case: dict[str, Any]) -> dict[str, np.ndarray]:
    ids = [int(v) for v in case["ids"]]
    return {
        "input": np.asarray([ids], dtype=np.int64),
        "input_lengths": np.asarray([len(ids)], dtype=np.int64),
        "scales": np.asarray(case["scales"], dtype=np.float32),
    }


def collect_outputs(model_path: Path, node_prefixes: list[str], node_names: list[str], explicit_outputs: list[str]) -> list[str]:
    model = onnx.load(str(model_path))
    outputs: list[str] = []
    seen: set[str] = set()
    for node in model.graph.node:
        node_name = node.name or ""
        include = False
        if node_name in node_names:
            include = True
        if any(node_name.startswith(prefix) for prefix in node_prefixes):
            include = True
        if include:
            for output_name in node.output:
                if output_name and output_name not in seen:
                    seen.add(output_name)
                    outputs.append(output_name)
    for output_name in explicit_outputs:
        if output_name not in seen:
            seen.add(output_name)
            outputs.append(output_name)
    if not outputs:
        raise ValueError("no outputs selected; pass --output, --node-prefix, or --node-name")
    return outputs


def _chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0 or chunk_size >= len(items):
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _lookup_value_info(model: onnx.ModelProto, output_name: str) -> onnx.ValueInfoProto | None:
    for coll in (model.graph.output, model.graph.value_info, model.graph.input):
        for vi in coll:
            if vi.name == output_name:
                copy = onnx.ValueInfoProto()
                copy.CopyFrom(vi)
                return copy
    for init in model.graph.initializer:
        if init.name == output_name:
            vi = onnx.helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
            return vi
    return None


def _write_full_model_with_outputs(model_path: Path, out_path: Path, keep_outputs: list[str]) -> None:
    model = onnx.load(str(model_path))
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    except Exception:
        # Some graphs still run fine even if shape inference is partial.
        pass

    existing = {vi.name for vi in model.graph.output}
    missing: list[str] = []
    for output_name in keep_outputs:
        if output_name in existing:
            continue
        vi = _lookup_value_info(model, output_name)
        if vi is None:
            # Fall back to an unspecified tensor. ORT can still resolve the output
            # at runtime even if shape inference did not populate value_info.
            vi = onnx.helper.make_tensor_value_info(output_name, TensorProto.UNDEFINED, None)
        model.graph.output.append(vi)
        existing.add(output_name)
        missing.append(output_name)

    onnx.save(model, str(out_path))


def _freeze_dp_random_zeros(model_path: Path, out_path: Path, feed: dict[str, np.ndarray]) -> None:
    model = onnx.load(str(model_path))
    input_seq = np.asarray(feed["input"])
    batch = int(input_seq.shape[0])
    seq_len = int(input_seq.shape[1])
    zero_tensor = numpy_helper.from_array(np.zeros((batch, 2, seq_len), dtype=np.float32), name="dp_random_frozen")

    replaced = False
    new_nodes = []
    for node in model.graph.node:
        if node.name == "/dp/RandomNormalLike" and node.op_type == "RandomNormalLike":
            new_nodes.append(
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=list(node.output),
                    name="/dp/RandomNormalLike_Frozen",
                    value=zero_tensor,
                )
            )
            replaced = True
        else:
            new_nodes.append(node)

    if not replaced:
        raise RuntimeError("did not find /dp/RandomNormalLike in Piper ONNX model")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save(model, str(out_path))


def run_submodel(
    model_path: Path,
    feed: dict[str, np.ndarray],
    keep_outputs: list[str],
    *,
    provider: str,
    seed: int,
    disable_fast_reduction: bool,
    chunk_size: int,
    graph_mode: str,
    opt_level: str,
    freeze_dp_random_zeros: bool,
    extract_per_output: bool,
) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="ort-piper-tts-debug-") as td:
        source_model = model_path
        if freeze_dp_random_zeros:
            source_model = Path(td) / "freeze_dp_random.onnx"
            _freeze_dp_random_zeros(model_path, source_model, feed)
        prev = os.environ.get("ORT_ROCM_DISABLE_FAST_REDUCTION")
        try:
            if provider == "rocm" and disable_fast_reduction:
                os.environ["ORT_ROCM_DISABLE_FAST_REDUCTION"] = "1"
            elif provider == "rocm":
                os.environ.pop("ORT_ROCM_DISABLE_FAST_REDUCTION", None)
            out: dict[str, np.ndarray] = {}
            providers = ["CPUExecutionProvider"] if provider == "cpu" else ["ROCMExecutionProvider", "CPUExecutionProvider"]
            for chunk in _chunks(keep_outputs, chunk_size):
                ort.set_seed(seed)
                np.random.seed(seed)
                so = ort.SessionOptions()
                if opt_level == "disable":
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                elif opt_level == "basic":
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                elif opt_level == "extended":
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                else:
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                runnable_model = Path(td) / f"debug_model_{abs(hash(tuple(chunk)))}.onnx"
                if graph_mode == "extract":
                    extract_outputs = chunk if not extract_per_output else [chunk[0]]
                    utils.extract_model(str(source_model), str(runnable_model), ["input", "input_lengths", "scales"], extract_outputs)
                else:
                    _write_full_model_with_outputs(source_model, runnable_model, chunk if not extract_per_output else [chunk[0]])
                sess = ort.InferenceSession(str(runnable_model), sess_options=so, providers=providers)
                primary_outputs = chunk if not extract_per_output else [chunk[0]]
                values = sess.run(primary_outputs, feed)
                out.update({name: value for name, value in zip(primary_outputs, values)})
                if extract_per_output and len(chunk) > 1:
                    for output_name in chunk[1:]:
                        single_model = Path(td) / f"debug_model_{abs(hash(output_name))}.onnx"
                        if graph_mode == "extract":
                            utils.extract_model(str(source_model), str(single_model), ["input", "input_lengths", "scales"], [output_name])
                        else:
                            _write_full_model_with_outputs(source_model, single_model, [output_name])
                        single_sess = ort.InferenceSession(str(single_model), sess_options=so, providers=providers)
                        out[output_name] = single_sess.run([output_name], feed)[0]
            return out
        finally:
            if prev is None:
                os.environ.pop("ORT_ROCM_DISABLE_FAST_REDUCTION", None)
            else:
                os.environ["ORT_ROCM_DISABLE_FAST_REDUCTION"] = prev


def compare_outputs(cpu: dict[str, np.ndarray], rocm: dict[str, np.ndarray], keep_outputs: list[str], max_report: int) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    first_rocm_nan: str | None = None
    first_mismatch: str | None = None
    for name in keep_outputs:
        a = np.asarray(cpu[name])
        b = np.asarray(rocm[name])
        try:
            close = bool(np.allclose(a, b, equal_nan=True))
        except Exception:
            close = bool(np.array_equal(a, b))
        cpu_nan = bool(np.issubdtype(a.dtype, np.floating) and np.isnan(a).any())
        rocm_nan = bool(np.issubdtype(b.dtype, np.floating) and np.isnan(b).any())
        if first_rocm_nan is None and rocm_nan:
            first_rocm_nan = name
        if first_mismatch is None and (not close or cpu_nan or rocm_nan):
            first_mismatch = name
        if not close or cpu_nan or rocm_nan:
            mismatches.append(
                {
                    "name": name,
                    "cpu_shape": list(a.shape),
                    "rocm_shape": list(b.shape),
                    "cpu_dtype": str(a.dtype),
                    "rocm_dtype": str(b.dtype),
                    "cpu_nan": cpu_nan,
                    "rocm_nan": rocm_nan,
                    "allclose": close,
                    "cpu_sample": a.reshape(-1)[:16].tolist(),
                    "rocm_sample": b.reshape(-1)[:16].tolist(),
                }
            )
    return {
        "selected_output_count": len(keep_outputs),
        "mismatch_count": len(mismatches),
        "first_mismatch": first_mismatch,
        "first_rocm_nan": first_rocm_nan,
        "mismatches": mismatches[:max_report],
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    case_file = Path(args.case_file).expanduser().resolve() if args.case_file else default_case_file(model_path)
    keep_outputs = collect_outputs(model_path, args.node_prefixes, args.node_names, args.outputs)
    case = load_case(case_file, args.case_label)
    feed = build_feed(case)
    cpu = run_submodel(
        model_path,
        feed,
        keep_outputs,
        provider="cpu",
        seed=args.seed,
        disable_fast_reduction=False,
        chunk_size=max(0, int(args.chunk_size)),
        graph_mode=args.graph_mode,
        opt_level=args.opt_level,
        freeze_dp_random_zeros=bool(args.freeze_dp_random_zeros),
        extract_per_output=bool(args.extract_per_output),
    )
    rocm = run_submodel(
        model_path,
        feed,
        keep_outputs,
        provider="rocm",
        seed=args.seed,
        disable_fast_reduction=bool(args.disable_fast_reduction),
        chunk_size=max(0, int(args.chunk_size)),
        graph_mode=args.graph_mode,
        opt_level=args.opt_level,
        freeze_dp_random_zeros=bool(args.freeze_dp_random_zeros),
        extract_per_output=bool(args.extract_per_output),
    )
    result = {
        "model": str(model_path),
        "case_file": str(case_file),
        "case_label": case.get("label", ""),
        "seed": args.seed,
        "disable_fast_reduction": bool(args.disable_fast_reduction),
        "graph_mode": args.graph_mode,
        "opt_level": args.opt_level,
        "freeze_dp_random_zeros": bool(args.freeze_dp_random_zeros),
        "extract_per_output": bool(args.extract_per_output),
        "comparison": compare_outputs(cpu, rocm, keep_outputs, args.max_report),
    }
    text = json.dumps(result, indent=2)
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
