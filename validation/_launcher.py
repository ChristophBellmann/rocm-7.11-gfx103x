#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def exec_script(script_name: str, argv: list[str] | None = None) -> "NoReturn":
    validation_root = Path(__file__).resolve().parent
    script_path = validation_root / "scripts" / script_name
    args = [sys.executable, str(script_path), *(sys.argv[1:] if argv is None else argv)]
    os.execv(sys.executable, args)
