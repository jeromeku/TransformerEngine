#!/usr/bin/env python3
"""
Minimal NCU helper for experiments/.

This is a slimmed‑down version of Wafer's ncu_tool.py focused on:
  - discovering `ncu`
  - running a profile against a given command
  - dumping a human‑readable summary from a `.ncu-rep` file

Usage (from repo root):
  python tools/ncu_tool_min.py check
  python tools/ncu_tool_min.py run --cmd "./experiments/bin/tma_test" --name tma_test
  python tools/ncu_tool_min.py summary --report .wafer/ncu/tma_test.ncu-rep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _find_ncu() -> Optional[str]:
    """Best‑effort search for Nsight Compute CLI (`ncu`)."""
    # 1) PATH
    ncu = shutil.which("ncu")
    if ncu:
        return ncu

    # 2) Common CUDA install locations (covers many dev setups)
    candidates: List[str] = []

    # Respect CUDA_HOME if set
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "ncu"))

    # Typical system installs
    candidates.extend(
        [
            "/usr/local/cuda/bin/ncu",
            "/usr/local/cuda-13.0/bin/ncu",
            "/usr/local/cuda-12.0/bin/ncu",
            "/usr/local/cuda-11.8/bin/ncu",
            "/opt/nvidia/nsight-compute/ncu",
            "/usr/bin/ncu",
        ]
    )

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def cmd_check() -> dict:
    """Return a small status blob describing NCU availability."""
    ncu = _find_ncu()
    if not ncu:
        return {
            "installed": False,
            "message": "ncu not found on PATH or common CUDA locations.",
            "hint": "Install Nsight Compute or ensure `ncu` is on PATH.",
        }

    try:
        out = subprocess.run(
            [ncu, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = (out.stdout or out.stderr or "").strip().splitlines()[0]
    except Exception:
        version = "unknown"

    return {
        "installed": True,
        "path": ncu,
        "version": version,
    }


def cmd_run_profile(cmd: str, name: str, output_dir: str) -> dict:
    """
    Run NCU against `cmd` and store a .ncu-rep under `output_dir`.

    This mirrors Wafer's `cmd_run` but is intentionally simple:
      ncu -o <output_dir>/<name>.ncu-rep <cmd...>
    """
    ncu = _find_ncu()
    if not ncu:
        return {
            "success": False,
            "error": "ncu not found. Run `check` for details.",
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / f"{name}.ncu-rep"

    # Use shlex.split for safety but avoid importing if not needed
    import shlex

    cmd_list = shlex.split(cmd)
    ncu_cmd = [ncu, "-o", str(rep_path)] + cmd_list

    try:
        proc = subprocess.run(
            ncu_cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "NCU command timed out (600s).",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to run ncu: {exc}",
        }

    if proc.returncode != 0:
        msg = proc.stderr or proc.stdout or f"NCU exited with code {proc.returncode}"
        return {
            "success": False,
            "error": msg,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    return {
        "success": True,
        "report": str(rep_path),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def cmd_summary(report: str, output: Optional[str]) -> dict:
    """
    Generate a human‑readable summary from a .ncu-rep file.

    This is a thin wrapper around:
      ncu --import <report> --page summary
    plus optional writing to a text file.
    """
    ncu = _find_ncu()
    if not ncu:
        return {
            "success": False,
            "error": "ncu not found. Run `check` for details.",
        }

    rep_path = Path(report)
    if not rep_path.is_file():
        return {
            "success": False,
            "error": f"Report not found: {report}",
        }

    try:
        proc = subprocess.run(
            [ncu, "--import", str(rep_path), "--page", "summary"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "ncu --import summary timed out (300s).",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to run ncu --import: {exc}",
        }

    if proc.returncode != 0:
        msg = proc.stderr or proc.stdout or f"NCU exited with code {proc.returncode}"
        return {
            "success": False,
            "error": msg,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    summary_text = proc.stdout
    out_path: Optional[str] = None
    if output:
        out_file = Path(output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(summary_text)
        out_path = str(out_file)

    return {
        "success": True,
        "summary_path": out_path,
        "summary": summary_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal NCU helper for experiments/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Check if ncu is available")

    run_p = subparsers.add_parser("run", help="Run NCU on a command")
    run_p.add_argument("--cmd", required=True, help="Command to profile (quoted)")
    run_p.add_argument("--name", required=True, help="Base name for report file")
    run_p.add_argument(
        "--output-dir",
        default=".wafer/ncu",
        help="Directory to store .ncu-rep (default: .wafer/ncu)",
    )

    summary_p = subparsers.add_parser("summary", help="Generate summary from .ncu-rep")
    summary_p.add_argument("--report", required=True, help="Path to .ncu-rep file")
    summary_p.add_argument(
        "--output",
        help="Optional path to write summary text (e.g. .wafer/ncu/<name>_summary.txt)",
    )

    args = parser.parse_args()

    if args.command == "check":
        result = cmd_check()
    elif args.command == "run":
        result = cmd_run_profile(args.cmd, args.name, args.output_dir)
    elif args.command == "summary":
        result = cmd_summary(args.report, args.output)
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

