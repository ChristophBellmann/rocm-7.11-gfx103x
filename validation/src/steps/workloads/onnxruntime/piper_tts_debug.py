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
from onnx import helper
from onnx import numpy_helper
from onnx import utils
import onnxruntime as ort


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def validation_root() -> Path:
    return repo_root() / "validation"


def staged_model_root() -> Path:
    return validation_root() / "workspace" / "cache" / "models" / "onnxruntime_tts"


def ensure_staged_model_path(path: Path) -> Path:
    staged_root = staged_model_root()
    candidate = path.expanduser().resolve(strict=False)
    original = path.expanduser()
    if original.is_symlink():
        raise ValueError(f"Piper ONNX model must be copied into {staged_root}, not symlinked: {original}")
    try:
        candidate.relative_to(staged_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Piper ONNX model must live under {staged_root}; external paths are not supported: {original}") from exc
    return candidate


def default_model_path() -> Path:
    model_dir = staged_model_root()
    models = sorted((p for p in model_dir.glob("*.onnx") if p.is_file() and not p.is_symlink()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        first_symlink = next((p for p in sorted(model_dir.glob("*.onnx"), key=lambda p: p.name) if p.is_symlink()), None)
        if first_symlink is not None:
            raise FileNotFoundError(f"no local staged Piper ONNX model in {model_dir}; replace symlink with copied file: {first_symlink.name}")
        raise FileNotFoundError(f"no staged Piper ONNX model in {model_dir}")
    return ensure_staged_model_path(models[0])


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
        help="Replace Piper RandomNormalLike nodes with dynamic zero tensors for deterministic debug runs",
    )
    ap.add_argument(
        "--write-frozen-model",
        default="",
        help="If set together with --freeze-dp-random-zeros, write the rewritten model here and exit",
    )
    ap.add_argument(
        "--extract-per-output",
        action="store_true",
        help="For extract mode, build one extracted submodel per requested output instead of a shared submodel",
    )
    ap.add_argument(
        "--exact-chain-repro",
        choices=[
            "dp_shape_cast",
            "dp_flow0_mul1",
            "dp_flow3_branch",
            "dp_flow3_split_path",
            "dp_flow5_branch",
            "dp_flow5_split_path",
            "dp_flow7_branch",
            "dp_flow7_slice24_path",
            "dp_flow7_add20_path",
            "dp_flow7_unsqueeze29_path",
            "dp_flow7_gathernd1_path",
            "dp_flow7_scatternd7_path",
            "dp_flow7_scatternd4_path",
            "dp_flow7_sub1_path",
            "dp_flow7_softmax1_path",
            "dp_flow7_pad2_path",
            "dp_flow7_transpose_path",
            "dp_flow7_proj_path",
            "dp_flow7_convmul15_path",
            "dp_flow7_convadd6_path",
            "dp_flow7_convmul14_path",
            "dp_flow7_convadd9_path",
            "dp_flow7_convadd3_path",
            "dp_flow7_convmul5_path",
            "dp_flow7_convmul4_path",
            "dp_flow7_convmul3_path",
            "dp_flow7_convsep1_path",
            "dp_flow7_convadd2_path",
            "dp_flow7_convmul9_path",
            "dp_flow7_convmul13_path",
            "dp_flow7_convadd_path",
            "dp_flow7_convmul7_path",
            "dp_flow7_convmul6_path",
            "dp_flow7_convadd4_path",
            "dp_flow7_convmul8_path",
            "dp_flow7_norm11_transpose1_path",
            "dp_flow7_norm11_add1_path",
            "dp_flow7_converf2_path",
            "dp_flow7_norm20_transpose1_path",
            "dp_flow7_norm22_transpose1_path",
            "dp_flow7_convadd8_path",
            "dp_convs_add2_path",
            "dp_mul_path",
            "dp_pre_conv_path",
            "dp_convs_mul4_path",
            "enc_p_encoder_mul2_path",
            "dp_convs_mul15_path",
            "dp_convs_mul3_path",
            "dp_flow7_norm21_transpose_path",
            "dp_flow7_norm20_add1_path",
            "dp_flow7_norm21_add1_path",
            "dp_flow7_norm20_mul_path",
            "dp_flow7_norm20_div_path",
            "dp_flow7_norm20_sub_path",
            "dp_flow7_norm20_sqrt_path",
            "dp_flow7_norm20_transpose_path",
            "dp_flow7_norm20_reducemean_path",
            "dp_flow7_convs1x1_0_conv_path",
            "dp_flow7_convmul2_path",
            "dp_flow7_convmul1_path",
            "dp_flow7_convadd1_path",
            "dp_flow7_norm10_transpose1_path",
            "dp_flow7_norm10_add1_path",
            "dp_flow7_converf_path",
            "dp_flow7_norm10_mul_path",
            "dp_flow7_norm10_div_path",
            "dp_flow7_norm10_sub_path",
            "dp_flow7_norm10_sqrt_path",
            "dp_flow7_norm10_add_path",
            "dp_flow7_norm10_reducemean1_path",
            "dp_flow7_norm10_pow_path",
            "dp_flow7_norm10_reducemean_path",
            "dp_flow7_norm10_transpose_path",
            "dp_flow7_convdiv_path",
            "dp_flow7_norm21_mul_path",
            "dp_flow7_norm21_div_path",
            "dp_flow7_norm21_sub_path",
            "dp_flow7_norm22_mul_path",
            "dp_flow7_norm22_div_path",
            "dp_flow7_norm22_sub_path",
            "dp_flow7_norm22_sqrt_path",
            "dp_flow7_converf5_path",
            "dp_flow7_convdiv5_path",
            "dp_flow7_convmul12_path",
            "dp_flow7_convmul11_path",
            "dp_flow7_convs1x1_2_conv_path",
            "dp_flow7_cast16_path",
            "dp_flow7_gatherelements1_path",
            "dp_flow7_mul33_path",
            "dp_flow7_sub13_path",
            "dp_flow7_mul_gates",
            "dp_flow7_gather_logits",
        ],
        default="",
        help="Build an exact mini-repro for a named internal chain using captured full-graph boundary tensors",
    )
    ap.add_argument(
        "--artifact-dir",
        default="",
        help="Optional directory for exact-chain mini-repro artifacts",
    )
    ap.add_argument(
        "--rocm-provider-option",
        action="append",
        dest="rocm_provider_options",
        default=[],
        help="ROCm EP provider option as key=value; can be passed multiple times",
    )
    ap.add_argument("--out-json", default="", help="Optional path for the JSON result")
    return ap.parse_args()


def _slug(text: str) -> str:
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in text.strip()]
    out = "".join(keep).strip("._-")
    return out or "case"


_EXACT_CHAIN_PRESETS: dict[str, dict[str, list[str] | str]] = {
    "dp_shape_cast": {
        "description": "Top-level Piper duration/shape chain through /Cast_output_0",
        "node_names": [
            "/dp/Split",
            "/Exp",
            "/Mul",
            "/Mul_1",
            "/Ceil",
            "/ReduceSum",
            "/Clip",
            "/Cast",
        ],
        "output_names": [
            "/dp/Split_output_0",
            "/Exp_output_0",
            "/Mul_output_0",
            "/Mul_1_output_0",
            "/Ceil_output_0",
            "/ReduceSum_output_0",
            "/Clip_output_0",
            "/Cast_output_0",
        ],
    },
    "dp_flow0_mul1": {
        "description": "First mismatching /dp flow boundary feeding the top-level shape chain",
        "node_names": [
            "/dp/flows.0/Sub",
            "/dp/flows.0/Mul",
            "/dp/flows.0/Mul_1",
        ],
        "output_names": [
            "/dp/flows.0/Sub_output_0",
            "/dp/flows.0/Mul_output_0",
            "/dp/flows.0/Mul_1_output_0",
        ],
    },
    "dp_flow3_branch": {
        "description": "Branch-level repro for the first proven upstream mismatch around /dp/flows.3",
        "node_names": [
            "/dp/flows.3/Split",
            "/dp/flows.3/ScatterND_9",
            "/dp/flows.3/Concat_20",
            "/dp/flows.3/Mul_35",
        ],
        "output_names": [
            "/dp/flows.3/Split_output_0",
            "/dp/flows.3/ScatterND_9_output_0",
            "/dp/flows.3/Concat_20_output_0",
            "/dp/flows.3/Mul_35_output_0",
        ],
    },
    "dp_flow3_split_path": {
        "description": "Left branch feeding /dp/flows.3/Split from /dp/flows.4/Slice",
        "node_names": [
            "/dp/flows.4/Slice",
            "/dp/flows.3/Split",
        ],
        "output_names": [
            "/dp/flows.4/Slice_output_0",
            "/dp/flows.3/Split_output_0",
        ],
    },
    "dp_flow5_branch": {
        "description": "Branch-level repro for the next recursive mismatch around /dp/flows.5",
        "node_names": [
            "/dp/flows.5/Split",
            "/dp/flows.5/ScatterND_9",
            "/dp/flows.5/Concat_20",
            "/dp/flows.5/Mul_35",
        ],
        "output_names": [
            "/dp/flows.5/Split_output_0",
            "/dp/flows.5/ScatterND_9_output_0",
            "/dp/flows.5/Concat_20_output_0",
            "/dp/flows.5/Mul_35_output_0",
        ],
    },
    "dp_flow5_split_path": {
        "description": "Left branch feeding /dp/flows.5/Split from /dp/flows.6/Slice",
        "node_names": [
            "/dp/flows.6/Slice",
            "/dp/flows.5/Split",
        ],
        "output_names": [
            "/dp/flows.6/Slice_output_0",
            "/dp/flows.5/Split_output_0",
        ],
    },
    "dp_flow7_branch": {
        "description": "Branch-level repro for the terminal recursive mismatch around /dp/flows.7",
        "node_names": [
            "/dp/flows.7/Split",
            "/dp/flows.7/ScatterND_9",
            "/dp/flows.7/Concat_20",
            "/dp/flows.7/Mul_35",
        ],
        "output_names": [
            "/dp/flows.7/Split_output_0",
            "/dp/flows.7/ScatterND_9_output_0",
            "/dp/flows.7/Concat_20_output_0",
            "/dp/flows.7/Mul_35_output_0",
        ],
    },
    "dp_flow7_slice24_path": {
        "description": "Focused repro for the terminal /dp/flows.7 Slice_24 branch",
        "node_names": [
            "/dp/flows.7/Reshape_28",
            "/dp/flows.7/Unsqueeze_31",
            "/dp/flows.7/Slice_24",
        ],
        "output_names": [
            "/dp/flows.7/Reshape_28_output_0",
            "/dp/flows.7/Unsqueeze_31_output_0",
            "/dp/flows.7/Slice_24_output_0",
        ],
    },
    "dp_flow7_add20_path": {
        "description": "Focused repro for the terminal /dp/flows.7 Add_20 branch",
        "node_names": [
            "/dp/flows.7/Div_3",
            "/dp/flows.7/Gather_25",
            "/dp/flows.7/Mul_34",
            "/dp/flows.7/GatherElements",
            "/dp/flows.7/Gather_24",
            "/dp/flows.7/Add_20",
        ],
        "output_names": [
            "/dp/flows.7/Div_3_output_0",
            "/dp/flows.7/Gather_25_output_0",
            "/dp/flows.7/Mul_34_output_0",
            "/dp/flows.7/GatherElements_output_0",
            "/dp/flows.7/Gather_24_output_0",
            "/dp/flows.7/Add_20_output_0",
        ],
    },
    "dp_flow7_unsqueeze29_path": {
        "description": "Focused repro for the /dp/flows.7 Sub_7 -> Unsqueeze_29 index path",
        "node_names": [
            "/dp/flows.7/ReduceSum",
            "/dp/flows.7/Sub_7",
            "/dp/flows.7/Unsqueeze_29",
        ],
        "output_names": [
            "/dp/flows.7/ReduceSum_output_0",
            "/dp/flows.7/Sub_7_output_0",
            "/dp/flows.7/Unsqueeze_29_output_0",
        ],
    },
    "dp_flow7_gathernd1_path": {
        "description": "Focused repro for the /dp/flows.7 GatherND_1 -> Unsqueeze_28 path",
        "node_names": [
            "/dp/flows.7/GatherND_1",
            "/dp/flows.7/Unsqueeze_28",
        ],
        "output_names": [
            "/dp/flows.7/GatherND_1_output_0",
            "/dp/flows.7/Unsqueeze_28_output_0",
        ],
    },
    "dp_flow7_scatternd7_path": {
        "description": "Focused repro for the /dp/flows.7 ScatterND_7 compare-mask path",
        "node_names": [
            "/dp/flows.7/ScatterND_5",
            "/dp/flows.7/Concat_16",
            "/dp/flows.7/Reshape_24",
            "/dp/flows.7/ScatterND_6",
            "/dp/flows.7/Concat_18",
            "/dp/flows.7/Reshape_26",
            "/dp/flows.7/ScatterND_7",
        ],
        "output_names": [
            "/dp/flows.7/ScatterND_5_output_0",
            "/dp/flows.7/Concat_16_output_0",
            "/dp/flows.7/Reshape_24_output_0",
            "/dp/flows.7/ScatterND_6_output_0",
            "/dp/flows.7/Concat_18_output_0",
            "/dp/flows.7/Reshape_26_output_0",
            "/dp/flows.7/ScatterND_7_output_0",
        ],
    },
    "dp_flow7_scatternd4_path": {
        "description": "Focused repro for the /dp/flows.7 ScatterND_4 path",
        "node_names": [
            "/dp/flows.7/Mul_17",
            "/dp/flows.7/Add_12",
            "/dp/flows.7/Expand_17",
            "/dp/flows.7/Unsqueeze_17",
            "/dp/flows.7/Expand_18",
            "/dp/flows.7/Unsqueeze_18",
            "/dp/flows.7/Concat_12",
            "/dp/flows.7/ConstantOfShape_20",
            "/dp/flows.7/Shape_43",
            "/dp/flows.7/Expand_16",
            "/dp/flows.7/Shape_46",
            "/dp/flows.7/Slice_14",
            "/dp/flows.7/Concat_13",
            "/dp/flows.7/Reshape_20",
            "/dp/flows.7/ScatterND_4",
        ],
        "output_names": [
            "/dp/flows.7/Mul_17_output_0",
            "/dp/flows.7/Add_12_output_0",
            "/dp/flows.7/Expand_17_output_0",
            "/dp/flows.7/Unsqueeze_17_output_0",
            "/dp/flows.7/Expand_18_output_0",
            "/dp/flows.7/Unsqueeze_18_output_0",
            "/dp/flows.7/Concat_12_output_0",
            "/dp/flows.7/Expand_16_output_0",
            "/dp/flows.7/Concat_13_output_0",
            "/dp/flows.7/Reshape_20_output_0",
            "/dp/flows.7/ScatterND_4_output_0",
        ],
    },
    "dp_flow7_sub1_path": {
        "description": "Focused repro for the /dp/flows.7 Gather_11 -> Sub_1 path",
        "node_names": [
            "/dp/flows.7/Gather_11",
            "/dp/flows.7/Cast_7",
            "/dp/flows.7/Mul_9",
            "/dp/flows.7/Sub_1",
        ],
        "output_names": [
            "/dp/flows.7/Gather_11_output_0",
            "/dp/flows.7/Cast_7_output_0",
            "/dp/flows.7/Mul_9_output_0",
            "/dp/flows.7/Sub_1_output_0",
        ],
    },
    "dp_flow7_softmax1_path": {
        "description": "Focused repro for the /dp/flows.7 Slice_1 -> GatherND_3 -> Softmax_1 path",
        "node_names": [
            "/dp/flows.7/Slice_1",
            "/dp/flows.7/Div_1",
            "/dp/flows.7/GatherND_3",
            "/dp/flows.7/Softmax_1",
        ],
        "output_names": [
            "/dp/flows.7/Slice_1_output_0",
            "/dp/flows.7/Div_1_output_0",
            "/dp/flows.7/GatherND_3_output_0",
            "/dp/flows.7/Softmax_1_output_0",
        ],
    },
    "dp_flow7_pad2_path": {
        "description": "Focused repro for the /dp/flows.7 Mul_16 -> Pad_2 path",
        "node_names": [
            "/dp/flows.7/Mul_16",
            "/dp/flows.7/Add_11",
            "/dp/flows.7/CumSum_1",
            "/dp/flows.7/Pad_2",
        ],
        "output_names": [
            "/dp/flows.7/Mul_16_output_0",
            "/dp/flows.7/Add_11_output_0",
            "/dp/flows.7/CumSum_1_output_0",
            "/dp/flows.7/Pad_2_output_0",
        ],
    },
    "dp_flow7_transpose_path": {
        "description": "Focused repro for the /dp/flows.7 Mul -> Reshape -> Transpose path",
        "node_names": [
            "/dp/flows.7/Mul",
            "/dp/flows.7/Reshape",
            "/dp/flows.7/Transpose",
        ],
        "output_names": [
            "/dp/flows.7/Mul_output_0",
            "/dp/flows.7/Reshape_output_0",
            "/dp/flows.7/Transpose_output_0",
        ],
    },
    "dp_flow7_proj_path": {
        "description": "Focused repro for the /dp/flows.7 conv/proj -> reshape -> transpose path",
        "node_names": [
            "/dp/flows.7/convs/Mul_15",
            "/dp/flows.7/proj/Conv",
            "/dp/flows.7/Mul",
            "/dp/flows.7/Reshape",
            "/dp/flows.7/Transpose",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_15_output_0",
            "/dp/flows.7/proj/Conv_output_0",
            "/dp/flows.7/Mul_output_0",
            "/dp/flows.7/Reshape_output_0",
            "/dp/flows.7/Transpose_output_0",
        ],
    },
    "dp_flow7_convmul15_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add_9 -> Mul_15 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_15",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_15_output_0",
        ],
    },
    "dp_flow7_convadd6_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add_6 path",
        "node_names": [
            "/dp/flows.7/convs/Add_6",
        ],
        "output_names": [
            "/dp/flows.7/convs/Add_6_output_0",
        ],
    },
    "dp_flow7_convmul14_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_14 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_14",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_14_output_0",
        ],
    },
    "dp_flow7_convadd9_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add_9 path",
        "node_names": [
            "/dp/flows.7/convs/Add_9",
        ],
        "output_names": [
            "/dp/flows.7/convs/Add_9_output_0",
        ],
    },
    "dp_flow7_convadd3_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add_3 path",
        "node_names": [
            "/dp/flows.7/convs/Add_3",
        ],
        "output_names": [
            "/dp/flows.7/convs/Add_3_output_0",
        ],
    },
    "dp_flow7_convmul5_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_5 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_5",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_5_output_0",
        ],
    },
    "dp_flow7_convmul4_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_4 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_4",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_4_output_0",
        ],
    },
    "dp_flow7_convmul3_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_3 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_3",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_3_output_0",
        ],
    },
    "dp_flow7_convsep1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/convs_sep.1/Conv path",
        "node_names": [
            "/dp/flows.7/convs/convs_sep.1/Conv",
        ],
        "output_names": [
            "/dp/flows.7/convs/convs_sep.1/Conv_output_0",
        ],
    },
    "dp_flow7_convadd2_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add_2 path",
        "node_names": [
            "/dp/flows.7/convs/Add_2",
        ],
        "output_names": [
            "/dp/flows.7/convs/Add_2_output_0",
        ],
    },
    "dp_flow7_convmul9_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_9 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_9",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_9_output_0",
        ],
    },
    "dp_flow7_convmul13_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_13 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_13",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_13_output_0",
        ],
    },
    "dp_flow7_convadd_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add path",
        "node_names": [
            "/dp/flows.7/convs/Add",
        ],
        "output_names": [
            "/dp/flows.7/convs/Add_output_0",
        ],
    },
    "dp_flow7_convmul7_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_6 -> Mul_7 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_6",
            "/dp/flows.7/convs/Mul_7",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_6_output_0",
            "/dp/flows.7/convs/Mul_7_output_0",
        ],
    },
    "dp_flow7_convmul6_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_6 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_6",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_6_output_0",
        ],
    },
    "dp_flow7_convadd4_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add_4 path",
        "node_names": [
            "/dp/flows.7/convs/Add_4",
        ],
        "output_names": [
            "/dp/flows.7/convs/Add_4_output_0",
        ],
    },
    "dp_flow7_convmul8_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_8 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_8",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_8_output_0",
        ],
    },
    "dp_flow7_norm11_transpose1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.1/Transpose_1 path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.1/Transpose_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.1/Transpose_1_output_0",
        ],
    },
    "dp_flow7_norm11_add1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.1 path through Add_1",
        "node_names": [
            "/dp/flows.7/convs/norms_1.1/Transpose",
            "/dp/flows.7/convs/norms_1.1/ReduceMean",
            "/dp/flows.7/convs/norms_1.1/Sub",
            "/dp/flows.7/convs/norms_1.1/Pow",
            "/dp/flows.7/convs/norms_1.1/ReduceMean_1",
            "/dp/flows.7/convs/norms_1.1/Add",
            "/dp/flows.7/convs/norms_1.1/Sqrt",
            "/dp/flows.7/convs/norms_1.1/Div",
            "/dp/flows.7/convs/norms_1.1/Mul",
            "/dp/flows.7/convs/norms_1.1/Add_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.1/Transpose_output_0",
            "/dp/flows.7/convs/norms_1.1/ReduceMean_output_0",
            "/dp/flows.7/convs/norms_1.1/Sub_output_0",
            "/dp/flows.7/convs/norms_1.1/Pow_output_0",
            "/dp/flows.7/convs/norms_1.1/ReduceMean_1_output_0",
            "/dp/flows.7/convs/norms_1.1/Add_output_0",
            "/dp/flows.7/convs/norms_1.1/Sqrt_output_0",
            "/dp/flows.7/convs/norms_1.1/Div_output_0",
            "/dp/flows.7/convs/norms_1.1/Mul_output_0",
            "/dp/flows.7/convs/norms_1.1/Add_1_output_0",
        ],
    },
    "dp_flow7_converf2_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Div_2 -> Erf_2 path",
        "node_names": [
            "/dp/flows.7/convs/Div_2",
            "/dp/flows.7/convs/Erf_2",
        ],
        "output_names": [
            "/dp/flows.7/convs/Div_2_output_0",
            "/dp/flows.7/convs/Erf_2_output_0",
        ],
    },
    "dp_flow7_norm20_transpose1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.0/Transpose_1 path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.0/Transpose_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.0/Transpose_1_output_0",
        ],
    },
    "dp_flow7_norm22_transpose1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.2/Transpose_1 path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.2/Transpose_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.2/Transpose_1_output_0",
        ],
    },
    "dp_flow7_convadd8_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add_8 path",
        "node_names": [
            "/dp/flows.7/convs/Add_8",
        ],
        "output_names": [
            "/dp/flows.7/convs/Add_8_output_0",
        ],
    },
    "dp_convs_add2_path": {
        "description": "Focused repro for the /dp/convs Add_2 path",
        "node_names": [
            "/dp/convs/Add_2",
        ],
        "output_names": [
            "/dp/convs/Add_2_output_0",
        ],
    },
    "dp_mul_path": {
        "description": "Focused repro for the top-level /dp/Mul path",
        "node_names": [
            "/dp/Mul",
        ],
        "output_names": [
            "/dp/Mul_output_0",
        ],
    },
    "dp_pre_conv_path": {
        "description": "Focused repro for the top-level /dp/pre/Conv path",
        "node_names": [
            "/dp/pre/Conv",
        ],
        "output_names": [
            "/dp/pre/Conv_output_0",
        ],
    },
    "dp_convs_mul4_path": {
        "description": "Focused repro for the top-level /dp/convs/Mul_4 path",
        "node_names": [
            "/dp/convs/Mul_4",
        ],
        "output_names": [
            "/dp/convs/Mul_4_output_0",
        ],
    },
    "enc_p_encoder_mul2_path": {
        "description": "Focused repro for the /enc_p/encoder/Mul_2 path",
        "node_names": [
            "/enc_p/encoder/Mul_2",
        ],
        "output_names": [
            "/enc_p/encoder/Mul_2_output_0",
        ],
    },
    "dp_convs_mul15_path": {
        "description": "Focused repro for the /dp/convs/Mul_15 path",
        "node_names": [
            "/dp/convs/Mul_15",
        ],
        "output_names": [
            "/dp/convs/Mul_15_output_0",
        ],
    },
    "dp_convs_mul3_path": {
        "description": "Focused repro for the /dp/convs/Mul_3 path",
        "node_names": [
            "/dp/convs/Mul_3",
        ],
        "output_names": [
            "/dp/convs/Mul_3_output_0",
        ],
    },
    "dp_flow7_norm21_add1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.1/Add_1 path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.1/Add_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.1/Add_1_output_0",
        ],
    },
    "dp_flow7_norm20_add1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.0/Add_1 path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.0/Add_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.0/Add_1_output_0",
        ],
    },
    "dp_flow7_norm20_mul_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.0/Mul path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.0/Mul",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.0/Mul_output_0",
        ],
    },
    "dp_flow7_norm20_div_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.0/Div path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.0/Div",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.0/Div_output_0",
        ],
    },
    "dp_flow7_norm20_sub_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.0/Sub path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.0/Sub",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.0/Sub_output_0",
        ],
    },
    "dp_flow7_norm20_sqrt_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.0/Sqrt path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.0/Sqrt",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.0/Sqrt_output_0",
        ],
    },
    "dp_flow7_norm20_transpose_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.0/Transpose path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.0/Transpose",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.0/Transpose_output_0",
        ],
    },
    "dp_flow7_norm20_reducemean_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.0/ReduceMean path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.0/ReduceMean",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.0/ReduceMean_output_0",
        ],
    },
    "dp_flow7_convs1x1_0_conv_path": {
        "description": "Focused repro for the /dp/flows.7 convs/convs_1x1.0/Conv path",
        "node_names": [
            "/dp/flows.7/convs/convs_1x1.0/Conv",
        ],
        "output_names": [
            "/dp/flows.7/convs/convs_1x1.0/Conv_output_0",
        ],
    },
    "dp_flow7_convmul2_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_2 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_2",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_2_output_0",
        ],
    },
    "dp_flow7_convmul1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_1 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_1_output_0",
        ],
    },
    "dp_flow7_convadd1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Add_1 path",
        "node_names": [
            "/dp/flows.7/convs/Add_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/Add_1_output_0",
        ],
    },
    "dp_flow7_norm10_transpose1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Transpose_1 path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Transpose_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Transpose_1_output_0",
        ],
    },
    "dp_flow7_norm10_add1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Add_1 path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Add_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Add_1_output_0",
        ],
    },
    "dp_flow7_norm10_mul_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Mul path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Mul",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Mul_output_0",
        ],
    },
    "dp_flow7_norm10_div_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Div path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Div",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Div_output_0",
        ],
    },
    "dp_flow7_norm10_sub_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Sub path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Sub",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Sub_output_0",
        ],
    },
    "dp_flow7_norm10_sqrt_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Sqrt path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Sqrt",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Sqrt_output_0",
        ],
    },
    "dp_flow7_norm10_add_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Add path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Add",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Add_output_0",
        ],
    },
    "dp_flow7_norm10_reducemean1_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/ReduceMean_1 path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/ReduceMean_1",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/ReduceMean_1_output_0",
        ],
    },
    "dp_flow7_norm10_pow_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Pow path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Pow",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Pow_output_0",
        ],
    },
    "dp_flow7_norm10_reducemean_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/ReduceMean path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/ReduceMean",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/ReduceMean_output_0",
        ],
    },
    "dp_flow7_norm10_transpose_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_1.0/Transpose path",
        "node_names": [
            "/dp/flows.7/convs/norms_1.0/Transpose",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_1.0/Transpose_output_0",
        ],
    },
    "dp_flow7_converf_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Erf path",
        "node_names": [
            "/dp/flows.7/convs/Erf",
        ],
        "output_names": [
            "/dp/flows.7/convs/Erf_output_0",
        ],
    },
    "dp_flow7_convdiv_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Div path",
        "node_names": [
            "/dp/flows.7/convs/Div",
        ],
        "output_names": [
            "/dp/flows.7/convs/Div_output_0",
        ],
    },
    "dp_flow7_norm21_transpose_path": {
        "description": "Focused repro for the /dp/flows.7 convs/convs_1x1.1/Conv -> norms_2.1/Transpose path",
        "node_names": [
            "/dp/flows.7/convs/convs_1x1.1/Conv",
            "/dp/flows.7/convs/norms_2.1/Transpose",
        ],
        "output_names": [
            "/dp/flows.7/convs/convs_1x1.1/Conv_output_0",
            "/dp/flows.7/convs/norms_2.1/Transpose_output_0",
        ],
    },
    "dp_flow7_norm21_mul_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.1/Mul path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.1/Mul",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.1/Mul_output_0",
        ],
    },
    "dp_flow7_norm21_div_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.1/Div path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.1/Div",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.1/Div_output_0",
        ],
    },
    "dp_flow7_norm21_sub_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.1/Sub path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.1/Sub",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.1/Sub_output_0",
        ],
    },
    "dp_flow7_norm22_mul_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.2/Mul path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.2/Mul",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.2/Mul_output_0",
        ],
    },
    "dp_flow7_norm22_div_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.2/Div path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.2/Div",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.2/Div_output_0",
        ],
    },
    "dp_flow7_norm22_sub_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.2/Sub path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.2/Sub",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.2/Sub_output_0",
        ],
    },
    "dp_flow7_norm22_sqrt_path": {
        "description": "Focused repro for the /dp/flows.7 convs/norms_2.2/Sqrt path",
        "node_names": [
            "/dp/flows.7/convs/norms_2.2/Sqrt",
        ],
        "output_names": [
            "/dp/flows.7/convs/norms_2.2/Sqrt_output_0",
        ],
    },
    "dp_flow7_converf5_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Erf_5 path",
        "node_names": [
            "/dp/flows.7/convs/Erf_5",
        ],
        "output_names": [
            "/dp/flows.7/convs/Erf_5_output_0",
        ],
    },
    "dp_flow7_convdiv5_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Div_5 path",
        "node_names": [
            "/dp/flows.7/convs/Div_5",
        ],
        "output_names": [
            "/dp/flows.7/convs/Div_5_output_0",
        ],
    },
    "dp_flow7_convmul12_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_12 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_12",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_12_output_0",
        ],
    },
    "dp_flow7_convmul11_path": {
        "description": "Focused repro for the /dp/flows.7 convs/Mul_11 path",
        "node_names": [
            "/dp/flows.7/convs/Mul_11",
        ],
        "output_names": [
            "/dp/flows.7/convs/Mul_11_output_0",
        ],
    },
    "dp_flow7_convs1x1_2_conv_path": {
        "description": "Focused repro for the /dp/flows.7 convs/convs_1x1.2/Conv path",
        "node_names": [
            "/dp/flows.7/convs/convs_1x1.2/Conv",
            "/dp/flows.7/convs/norms_2.2/Transpose",
        ],
        "output_names": [
            "/dp/flows.7/convs/convs_1x1.2/Conv_output_0",
            "/dp/flows.7/convs/norms_2.2/Transpose_output_0",
        ],
    },
    "dp_flow7_cast16_path": {
        "description": "Focused repro for the /dp/flows.7 GreaterOrEqual_1 -> Cast_16 path",
        "node_names": [
            "/dp/flows.7/GatherND_1",
            "/dp/flows.7/Unsqueeze_28",
            "/dp/flows.7/ScatterND_7",
            "/dp/flows.7/GreaterOrEqual_1",
            "/dp/flows.7/Cast_16",
        ],
        "output_names": [
            "/dp/flows.7/GatherND_1_output_0",
            "/dp/flows.7/Unsqueeze_28_output_0",
            "/dp/flows.7/ScatterND_7_output_0",
            "/dp/flows.7/GreaterOrEqual_1_output_0",
            "/dp/flows.7/Cast_16_output_0",
        ],
    },
    "dp_flow7_gatherelements1_path": {
        "description": "Focused repro for the /dp/flows.7 ScatterND_3 -> GatherElements_1 path",
        "node_names": [
            "/dp/flows.7/Slice_11",
            "/dp/flows.7/Slice_12",
            "/dp/flows.7/Sub_3",
            "/dp/flows.7/GatherElements_1",
        ],
        "output_names": [
            "/dp/flows.7/Slice_11_output_0",
            "/dp/flows.7/Slice_12_output_0",
            "/dp/flows.7/Sub_3_output_0",
            "/dp/flows.7/GatherElements_1_output_0",
        ],
    },
    "dp_flow7_mul33_path": {
        "description": "Focused repro for the /dp/flows.7 Gather/Sub -> Mul_33 value path",
        "node_names": [
            "/dp/flows.7/Gather_27",
            "/dp/flows.7/Neg",
            "/dp/flows.7/GatherND_1",
            "/dp/flows.7/Gather_26",
            "/dp/flows.7/Sub_8",
            "/dp/flows.7/Mul_30",
            "/dp/flows.7/Mul_33",
        ],
        "output_names": [
            "/dp/flows.7/Gather_27_output_0",
            "/dp/flows.7/Neg_output_0",
            "/dp/flows.7/GatherND_1_output_0",
            "/dp/flows.7/Gather_26_output_0",
            "/dp/flows.7/Sub_8_output_0",
            "/dp/flows.7/Mul_30_output_0",
            "/dp/flows.7/Mul_33_output_0",
        ],
    },
    "dp_flow7_sub13_path": {
        "description": "Focused repro for the /dp/flows.7 variance path ending at Sub_13",
        "node_names": [
            "/dp/flows.7/Mul_29",
            "/dp/flows.7/Mul_27",
            "/dp/flows.7/Sub_11",
            "/dp/flows.7/Neg_1",
            "/dp/flows.7/Pow",
            "/dp/flows.7/Mul_32",
            "/dp/flows.7/Sub_12",
            "/dp/flows.7/Sqrt",
            "/dp/flows.7/Sub_13",
        ],
        "output_names": [
            "/dp/flows.7/Mul_29_output_0",
            "/dp/flows.7/Mul_27_output_0",
            "/dp/flows.7/Sub_11_output_0",
            "/dp/flows.7/Neg_1_output_0",
            "/dp/flows.7/Pow_output_0",
            "/dp/flows.7/Mul_32_output_0",
            "/dp/flows.7/Sub_12_output_0",
            "/dp/flows.7/Sqrt_output_0",
            "/dp/flows.7/Sub_13_output_0",
        ],
    },
    "dp_flow7_mul_gates": {
        "description": "Focused repro for the suspicious /dp/flows.7 Mul gate path",
        "node_names": [
            "/dp/flows.7/Sub_1",
            "/dp/flows.7/Softmax",
            "/dp/flows.7/Mul_10",
            "/dp/flows.7/Softmax_1",
            "/dp/flows.7/Mul_16",
        ],
        "output_names": [
            "/dp/flows.7/Sub_1_output_0",
            "/dp/flows.7/Softmax_output_0",
            "/dp/flows.7/Mul_10_output_0",
            "/dp/flows.7/Softmax_1_output_0",
            "/dp/flows.7/Mul_16_output_0",
        ],
    },
    "dp_flow7_gather_logits": {
        "description": "Focused repro for the /dp/flows.7 Slice/Div/GatherND logits path",
        "node_names": [
            "/dp/flows.7/Slice",
            "/dp/flows.7/Div",
            "/dp/flows.7/Slice_1",
            "/dp/flows.7/Div_1",
            "/dp/flows.7/Transpose_3",
            "/dp/flows.7/GatherND_2",
            "/dp/flows.7/GatherND_3",
        ],
        "output_names": [
            "/dp/flows.7/Slice_output_0",
            "/dp/flows.7/Div_output_0",
            "/dp/flows.7/Slice_1_output_0",
            "/dp/flows.7/Div_1_output_0",
            "/dp/flows.7/Transpose_3_output_0",
            "/dp/flows.7/GatherND_2_output_0",
            "/dp/flows.7/GatherND_3_output_0",
        ],
    },
}


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


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _parse_provider_options(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key, sep, value = text.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"invalid --rocm-provider-option {item!r}; expected key=value")
        out[key] = value.strip()
    return out


def _providers(provider: str, rocm_provider_options: dict[str, str]) -> list[str] | list[tuple[str, dict[str, str]]]:
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if rocm_provider_options:
        return [("ROCMExecutionProvider", rocm_provider_options), "CPUExecutionProvider"]
    return ["ROCMExecutionProvider", "CPUExecutionProvider"]


def _rocm_lib_hint() -> str:
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


def _node_by_name(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {node.name: node for node in model.graph.node if node.name}


def _initializer_by_name(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {init.name: init for init in model.graph.initializer}


def _exact_chain_spec(model: onnx.ModelProto, preset_name: str) -> dict[str, Any]:
    preset = _EXACT_CHAIN_PRESETS[preset_name]
    want_names = list(preset["node_names"])
    node_map = _node_by_name(model)
    missing = [name for name in want_names if name not in node_map]
    if missing:
        raise ValueError(f"missing nodes for exact-chain preset {preset_name}: {missing}")
    selected_nodes = [node for node in model.graph.node if node.name in set(want_names)]
    produced = {output_name for node in selected_nodes for output_name in node.output if output_name}
    initializer_names = set(_initializer_by_name(model))
    boundary_inputs: list[str] = []
    for node in selected_nodes:
        for input_name in node.input:
            if not input_name or input_name in produced or input_name in initializer_names:
                continue
            boundary_inputs.append(input_name)
    return {
        "preset": preset_name,
        "description": str(preset["description"]),
        "node_names": [node.name for node in selected_nodes],
        "boundary_inputs": _unique_keep_order(boundary_inputs),
        "output_names": _unique_keep_order(list(preset["output_names"])),
    }


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


def _freeze_dp_random_zeros(model_path: Path, out_path: Path, feed: dict[str, np.ndarray] | None = None) -> None:
    model = onnx.load(str(model_path))

    replaced = 0
    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "RandomNormalLike":
            shape_name = f"{node.output[0]}__shape"
            zero_value = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name=f"{node.output[0]}__zero_value")
            new_nodes.append(
                onnx.helper.make_node(
                    "Shape",
                    inputs=[node.input[0]],
                    outputs=[shape_name],
                    name=f"{node.name}_FrozenShape",
                )
            )
            new_nodes.append(
                onnx.helper.make_node(
                    "ConstantOfShape",
                    inputs=[shape_name],
                    outputs=list(node.output),
                    name=f"{node.name}_FrozenZeros",
                    value=zero_value,
                )
            )
            replaced += 1
        else:
            new_nodes.append(node)

    if not replaced:
        raise RuntimeError("did not find RandomNormalLike in Piper ONNX model")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save(model, str(out_path))


def _session_options(opt_level: str) -> ort.SessionOptions:
    so = ort.SessionOptions()
    if opt_level == "disable":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif opt_level == "basic":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == "extended":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


def _run_model_outputs(
    model_path: Path,
    feed: dict[str, np.ndarray],
    output_names: list[str],
    *,
    provider: str,
    seed: int,
    disable_fast_reduction: bool,
    opt_level: str,
    rocm_provider_options: dict[str, str],
) -> dict[str, np.ndarray]:
    prev = os.environ.get("ORT_ROCM_DISABLE_FAST_REDUCTION")
    try:
        if provider == "rocm" and disable_fast_reduction:
            os.environ["ORT_ROCM_DISABLE_FAST_REDUCTION"] = "1"
        elif provider == "rocm":
            os.environ.pop("ORT_ROCM_DISABLE_FAST_REDUCTION", None)
        providers = _providers(provider, rocm_provider_options)
        ort.set_seed(seed)
        np.random.seed(seed)
        sess = ort.InferenceSession(str(model_path), sess_options=_session_options(opt_level), providers=providers)
        values = sess.run(output_names, feed)
        return {name: np.asarray(value) for name, value in zip(output_names, values)}
    finally:
        if prev is None:
            os.environ.pop("ORT_ROCM_DISABLE_FAST_REDUCTION", None)
        else:
            os.environ["ORT_ROCM_DISABLE_FAST_REDUCTION"] = prev


def _copy_node(node: onnx.NodeProto) -> onnx.NodeProto:
    out = onnx.NodeProto()
    out.CopyFrom(node)
    return out


def _copy_initializer(init: onnx.TensorProto) -> onnx.TensorProto:
    out = onnx.TensorProto()
    out.CopyFrom(init)
    return out


def _value_info_from_array(name: str, value: np.ndarray) -> onnx.ValueInfoProto:
    tensor = numpy_helper.from_array(np.asarray(value), name=name)
    return helper.make_tensor_value_info(name, tensor.data_type, list(np.asarray(value).shape))


def _build_exact_chain_model(
    model: onnx.ModelProto,
    chain_spec: dict[str, Any],
    captured_cpu: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    node_names = set(str(name) for name in (chain_spec.get("node_names") or []))
    nodes = [_copy_node(node) for node in model.graph.node if node.name in node_names]
    init_map = _initializer_by_name(model)
    initializer_names = sorted(
        {
            input_name
            for node in model.graph.node
            if node.name in node_names
            for input_name in node.input
            if input_name in init_map
        }
    )
    initializers = [_copy_initializer(init_map[name]) for name in initializer_names]
    inputs = [_value_info_from_array(name, captured_cpu[name]) for name in chain_spec["boundary_inputs"]]
    outputs = [_value_info_from_array(name, captured_cpu[name]) for name in chain_spec["output_names"]]
    graph = helper.make_graph(
        nodes,
        f"exact_chain_{chain_spec['preset']}",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    out_model = helper.make_model(graph, producer_name="piper_tts_debug")
    out_model.ir_version = model.ir_version
    del out_model.opset_import[:]
    for opset in model.opset_import:
        copied = onnx.OperatorSetIdProto()
        copied.CopyFrom(opset)
        out_model.opset_import.append(copied)
    onnx.save(out_model, str(out_path))


def _resolve_exact_chain_artifact_dir(
    preset_name: str,
    case_label: str,
    *,
    artifact_dir: str,
    out_json: str,
) -> Path:
    if artifact_dir:
        out = Path(artifact_dir).expanduser().resolve()
    elif out_json:
        out = Path(out_json).expanduser().resolve().parent / f"{preset_name}_{_slug(case_label)}"
    else:
        out = validation_root() / "workspace" / "debug" / f"exact_chain_{preset_name}_{_slug(case_label)}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_exact_chain_repro(
    model_path: Path,
    model: onnx.ModelProto,
    chain_spec: dict[str, Any],
    full_cpu: dict[str, np.ndarray],
    full_rocm: dict[str, np.ndarray],
    *,
    case_label: str,
    seed: int,
    disable_fast_reduction: bool,
    opt_level: str,
    artifact_dir: Path,
    max_report: int,
    rocm_provider_options: dict[str, str],
) -> dict[str, Any]:
    cpu_boundary_inputs = {name: np.asarray(full_cpu[name]) for name in chain_spec["boundary_inputs"]}
    rocm_boundary_inputs = {name: np.asarray(full_rocm[name]) for name in chain_spec["boundary_inputs"]}
    mini_model_path = artifact_dir / f"{chain_spec['preset']}.onnx"
    cpu_feed_npz = artifact_dir / f"{chain_spec['preset']}.cpu_boundary_inputs.npz"
    rocm_feed_npz = artifact_dir / f"{chain_spec['preset']}.rocm_boundary_inputs.npz"
    report_json = artifact_dir / f"{chain_spec['preset']}.report.json"

    _build_exact_chain_model(model, chain_spec, full_cpu, mini_model_path)
    np.savez(cpu_feed_npz, **cpu_boundary_inputs)
    np.savez(rocm_feed_npz, **rocm_boundary_inputs)

    mini_cpu = _run_model_outputs(
        mini_model_path,
        cpu_boundary_inputs,
        chain_spec["output_names"],
        provider="cpu",
        seed=seed,
        disable_fast_reduction=False,
        opt_level=opt_level,
        rocm_provider_options={},
    )
    mini_rocm = _run_model_outputs(
        mini_model_path,
        cpu_boundary_inputs,
        chain_spec["output_names"],
        provider="rocm",
        seed=seed,
        disable_fast_reduction=disable_fast_reduction,
        opt_level=opt_level,
        rocm_provider_options=rocm_provider_options,
    )

    result = {
        "preset": chain_spec["preset"],
        "description": chain_spec["description"],
        "case_label": case_label,
        "node_names": chain_spec["node_names"],
        "boundary_inputs": chain_spec["boundary_inputs"],
        "output_names": chain_spec["output_names"],
        "mini_model": str(mini_model_path),
        "cpu_boundary_inputs_npz": str(cpu_feed_npz),
        "rocm_boundary_inputs_npz": str(rocm_feed_npz),
        "full_boundary_compare": compare_outputs(full_cpu, full_rocm, list(chain_spec["boundary_inputs"]), max_report),
        "full_chain_compare": compare_outputs(full_cpu, full_rocm, list(chain_spec["output_names"]), max_report),
        "mini_cpu_boundary_compare": compare_outputs(mini_cpu, mini_rocm, list(chain_spec["output_names"]), max_report),
    }
    report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report_json"] = str(report_json)
    return result


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
    rocm_provider_options: dict[str, str],
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
            providers = _providers(provider, rocm_provider_options)
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
    rocm_provider_options = _parse_provider_options(args.rocm_provider_options)
    model_path = ensure_staged_model_path(Path(args.model))
    if args.write_frozen_model:
        if not bool(args.freeze_dp_random_zeros):
            raise ValueError("--write-frozen-model requires --freeze-dp-random-zeros")
        out_path = Path(args.write_frozen_model).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _freeze_dp_random_zeros(model_path, out_path, None)
        print(json.dumps({"model": str(model_path), "frozen_model": str(out_path)}, indent=2))
        return 0
    case_file = Path(args.case_file).expanduser().resolve() if args.case_file else default_case_file(model_path)
    model = onnx.load(str(model_path))
    exact_chain = _exact_chain_spec(model, args.exact_chain_repro) if args.exact_chain_repro else None
    requested_outputs: list[str] = []
    if args.outputs or args.node_prefixes or args.node_names:
        requested_outputs = collect_outputs(model_path, args.node_prefixes, args.node_names, args.outputs)
    elif exact_chain is not None:
        requested_outputs = list(exact_chain["output_names"])
    else:
        raise ValueError("no outputs selected; pass --output, --node-prefix, or --node-name")
    keep_outputs = list(requested_outputs)
    if exact_chain is not None:
        keep_outputs = _unique_keep_order(keep_outputs + list(exact_chain["boundary_inputs"]) + list(exact_chain["output_names"]))
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
        rocm_provider_options={},
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
        rocm_provider_options=rocm_provider_options,
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
        "rocm_provider_options": rocm_provider_options,
        "comparison": compare_outputs(cpu, rocm, requested_outputs, args.max_report),
        "hip_lib": _rocm_lib_hint(),
    }
    if exact_chain is not None:
        artifact_dir = _resolve_exact_chain_artifact_dir(
            exact_chain["preset"],
            str(case.get("label", "")),
            artifact_dir=args.artifact_dir,
            out_json=args.out_json,
        )
        result["exact_chain_repro"] = build_exact_chain_repro(
            model_path,
            model,
            exact_chain,
            cpu,
            rocm,
            case_label=str(case.get("label", "")),
            seed=args.seed,
            disable_fast_reduction=bool(args.disable_fast_reduction),
            opt_level=args.opt_level,
            artifact_dir=artifact_dir,
            max_report=args.max_report,
            rocm_provider_options=rocm_provider_options,
        )
    text = json.dumps(result, indent=2)
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
