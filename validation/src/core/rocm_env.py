from __future__ import annotations

import os
from pathlib import Path


def activated_env(base_env: dict[str, str], rocm_dist: Path) -> dict[str, str]:
    env = dict(base_env)
    env["ROCM_PATH"] = str(rocm_dist)
    env.setdefault("HIP_PATH", str(rocm_dist))
    env.setdefault("HSA_PATH", str(rocm_dist))
    env["PATH"] = f"{rocm_dist}/bin:{rocm_dist}/llvm/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{rocm_dist}/lib:{rocm_dist}/lib64:{rocm_dist}/lib/host-math/lib:{rocm_dist}/lib/rocm_sysdeps/lib:{rocm_dist}/llvm/lib:"
        f"{env.get('LD_LIBRARY_PATH', '')}"
    )
    if "HIP_DEVICE_LIB_PATH" not in env:
        p1 = rocm_dist / "lib" / "llvm" / "amdgcn" / "bitcode"
        p2 = rocm_dist / "amdgcn" / "bitcode"
        env["HIP_DEVICE_LIB_PATH"] = str(p1 if p1.is_dir() else p2)

    # WORKAROUND: HIP CLR (therock-7.13) doesn't link libamdhip64.so against
    # librocm_smi64.so. PyTorch/libtorch_hip.so references rsmi_init but can't
    # resolve it. Preload the SMI library to make the symbol available.
    rsmi_lib = rocm_dist / "lib" / "librocm_smi64.so"
    if rsmi_lib.is_file():
        existing = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{rsmi_lib}:{existing}" if existing else str(rsmi_lib)

    return env


def deactivated_env(env: dict[str, str], rocm_dist: Path) -> dict[str, str]:
    """
    Remove in-tree ROCm entries from PATH/LD_LIBRARY_PATH.

    Some third-party Python wheels (notably ROCm-enabled PyTorch) are tightly
    coupled to a specific ROCm version and may crash if forced to load the
    in-tree dist libraries via LD_LIBRARY_PATH. For such steps, keep the rest
    of the environment but drop the in-tree prefixes.
    """
    out = dict(env)

    path = out.get("PATH", "")
    drop_path = {f"{rocm_dist}/bin", f"{rocm_dist}/llvm/bin"}
    out["PATH"] = ":".join([p for p in path.split(":") if p and p not in drop_path])

    ld = out.get("LD_LIBRARY_PATH", "")
    drop_ld = {
        f"{rocm_dist}/lib",
        f"{rocm_dist}/lib64",
        f"{rocm_dist}/lib/host-math/lib",
        f"{rocm_dist}/lib/rocm_sysdeps/lib",
        f"{rocm_dist}/llvm/lib",
    }
    out["LD_LIBRARY_PATH"] = ":".join([p for p in ld.split(":") if p and p not in drop_ld])

    # Let tools discover their own ROCm root (e.g. system /opt/rocm) if needed.
    for k in ("ROCM_PATH", "HIP_PATH", "HSA_PATH", "HIP_DEVICE_LIB_PATH"):
        out.pop(k, None)
    return out


def which(exe: str, env: dict[str, str]) -> str | None:
    path = env.get("PATH", os.environ.get("PATH", ""))
    for p in path.split(":"):
        candidate = Path(p) / exe
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None
