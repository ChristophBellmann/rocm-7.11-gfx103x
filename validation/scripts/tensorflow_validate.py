#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _bootstrap import reexec_in_venv, repo_root, validation_root


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rc = reexec_in_venv(Path(__file__).resolve(), argv)
    if rc is not None:
        return rc

    ap = argparse.ArgumentParser(
        prog="tensorflow_validate.py",
        description="Build TensorFlow ROCm wheel (plus minimal ROCm sanity) and print a compact summary.",
    )
    ap.add_argument("--build-dirs", default=None, help="Comma-separated build dirs to validate (default: auto).")
    ap.add_argument("--no-downloads", action="store_true", help="Disable downloads (TensorFlow build step will SKIP).")
    ap.add_argument("--log", action="store_true", help="Write logs under validation/workspace/runs/ (default: off).")
    args = ap.parse_args(argv)

    validate_args: list[str] = ["--profile", "tensorflow", "--yes", "--no-power"]
    if args.build_dirs:
        validate_args += ["--build-dirs", args.build_dirs]
    if args.no_downloads:
        validate_args += ["--no-downloads"]
    if args.log:
        validate_args += ["--log"]

    src = validation_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(repo_root())

    from core.config import load_config  # noqa: E402
    from core.context import Context  # noqa: E402
    from core.reporting.summary import print_summary  # noqa: E402
    from steps.plan import build_plan, run_plan  # noqa: E402

    cfg = load_config(profile="tensorflow")
    if args.build_dirs:
        cfg["run"]["build_dirs"] = [x.strip() for x in args.build_dirs.split(",") if x.strip()]
    if args.no_downloads or os.environ.get("ROCM_VALIDATION_NO_DOWNLOADS", "") == "1":
        cfg["run"]["downloads_enabled"] = False
        cfg["run"]["ask_before_downloads"] = False
    cfg["run"]["power_monitor"] = False

    ctx = Context.from_repo(cfg=cfg, enable_logs=bool(args.log))
    plan = build_plan(cfg)
    results = run_plan(ctx, cfg, plan)
    print_summary(ctx, results)
    rc = 0 if all(r.status != "FAIL" for r in results) else 1

    wheel_res = next((r for r in results if "TensorFlow (ROCm) wheel build" in r.name), None)
    func_res = next((r for r in results if "TensorFlow matmul (GPU)" in r.name), None)
    if wheel_res is None and func_res is None:
        return rc

    primary = func_res or wheel_res
    build_dir = primary.build_dir if primary is not None else ""

    wheel_status = wheel_res.status if wheel_res else ""
    wheel_metric = (wheel_res.metric or "").strip() if wheel_res else ""
    func_status = func_res.status if func_res else ""
    func_metric = (func_res.metric or "").strip() if func_res else ""

    if func_status == "OK" and wheel_status in {"", "OK", "SKIP"}:
        overall_status = "OK"
    elif func_status == "FAIL" or wheel_status == "FAIL":
        overall_status = "FAIL"
    elif func_status:
        overall_status = func_status
    else:
        overall_status = wheel_status

    print("")
    print("==== tensorflow validation ====")
    print(f"run_dir : {ctx.run_root}")
    print(f"build   : {build_dir}")
    print("profile : tensorflow")
    print(f"overall : {overall_status}")
    if wheel_res is not None:
        print(f"wheel   : {wheel_status}")
        if wheel_metric:
            print(f"wheel_m : {wheel_metric}")
    if func_res is not None:
        print(f"load    : {func_status}")
        if func_metric:
            print(f"load_m  : {func_metric}")

    if overall_status != "OK":
        print("")
        print("next steps:")
        print("- Re-run with logs: `python3 validation/scripts/tensorflow_validate.py --log`")
        print("- Monitor systemd build: `validation/scripts/tensorflow_rocm/monitor_tensorflow_rocm_build.sh --once`")
        print("- Check TensorFlow build log: `tail -n 200 validation/workspace/builds/tensorflow_rocm/build_start.log`")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
