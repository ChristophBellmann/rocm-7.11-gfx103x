from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def validation_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return validation_root().parent


def in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def ensure_venv() -> Path:
    venv_dir = validation_root() / "workspace" / "envs" / "py"
    py = venv_dir / "bin" / "python"
    if py.exists():
        return py

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    lock = validation_root() / "requirements-lock.txt"
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(lock)])
    return py


def reexec_in_venv(script_path: Path, argv: list[str]) -> int | None:
    if in_venv() or os.environ.get("ROCM_VALIDATION_NO_BOOTSTRAP", "") == "1":
        return None
    py = ensure_venv()
    env = os.environ.copy()
    env["THEROCK_VALIDATION_BOOTSTRAPPED"] = "1"
    env.setdefault("PYTHONPYCACHEPREFIX", str(validation_root() / "__pycache__"))
    return subprocess.call([str(py), str(script_path)] + argv, env=env)


def run_cli(command: str, argv: list[str], script_path: Path) -> int:
    rc = reexec_in_venv(script_path, argv)
    if rc is not None:
        return rc

    src = validation_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(repo_root())

    from cli.main import main as cli_main  # noqa: E402

    return cli_main([command] + argv)
