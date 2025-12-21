#!/usr/bin/env python3
"""
Compile a CUDA file to PTX/SASS with line info and build source<->assembly maps.

This is a self-contained, minimal version of Wafer's compiler_explorer_tool.py
specialized for this repo. It:
  - runs nvcc -ptx / -cubin with --generate-line-info
  - uses nvdisasm -g to get SASS (if available)
  - parses .loc and //## File "...", line N markers into JSON mappings

Usage (from repo root):
  python tools/ptx_sass_map.py compile \\
      --src experiments/tma_test.cu \\
      --arch sm_80 \\
      --out-dir .wafer/ptx-sass
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional


def _repo_root() -> Path:
    """Assume tools/ lives directly under the repo root."""
    return Path(__file__).resolve().parents[1]


def _find_nvcc() -> Optional[str]:
    """Best‑effort search for nvcc."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/usr/local/cuda-11.8/bin/nvcc",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def _find_nvdisasm() -> Optional[str]:
    """Find nvdisasm (optional, only needed for SASS)."""
    nvdisasm = shutil.which("nvdisasm")
    if nvdisasm:
        return nvdisasm

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvdisasm"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    for path in [
        "/usr/local/cuda/bin/nvdisasm",
        "/usr/local/cuda-13.0/bin/nvdisasm",
        "/usr/local/cuda-12.0/bin/nvdisasm",
        "/usr/local/cuda-11.8/bin/nvdisasm",
    ]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def _include_paths_for_repo(src: Path) -> List[str]:
    """
    Approximate the include setup used by experiments/Makefile:
      - CUTLASS include dirs
      - experiments/ for local headers
    """
    root = _repo_root()
    cutlass_root = root / "3rdparty" / "cutlass"
    experiments_dir = root / "experiments"

    includes: List[Path] = [experiments_dir]
    includes.append(cutlass_root / "include")
    includes.append(cutlass_root / "tools" / "util" / "include")
    includes.append(cutlass_root / "examples" / "common")

    # Also include the source directory itself
    includes.append(src.parent)

    return [str(p) for p in includes if p.is_dir()]


def parse_ptx_line_mapping(ptx_content: str) -> Dict[str, Any]:
    """
    Build source<->PTX line mapping using .loc directives.

    .loc format: `.loc <file_id> <line_number> <column>`
    """
    mapping: Dict[str, Any] = {
        "source_to_asm": {},  # line -> [ptx_line_numbers]
        "asm_to_source": {},  # ptx_line_number -> line
    }

    lines = ptx_content.splitlines()
    current_source_line: Optional[int] = None

    for ptx_line_num, line in enumerate(lines, start=1):
        loc_match = re.match(r"\s*\.loc\s+\d+\s+(\d+)\s+\d+", line)
        if loc_match:
            current_source_line = int(loc_match.group(1))
            continue

        stripped = line.strip()
        if not current_source_line or not stripped:
            continue
        if stripped.startswith("//"):
            continue

        # Treat non‑directive lines (and .pragma) as instructions
        if not stripped.startswith(".") or stripped.startswith(".pragma"):
            mapping["asm_to_source"][ptx_line_num] = current_source_line
            src_map = mapping["source_to_asm"].setdefault(current_source_line, [])
            if ptx_line_num not in src_map:
                src_map.append(ptx_line_num)

    return mapping


def parse_sass_line_mapping(sass_content: str) -> Dict[str, Any]:
    """
    Build source<->SASS line mapping using nvdisasm -g markers:
      //## File "/path/to/file.cu", line N
    """
    mapping: Dict[str, Any] = {
        "source_to_asm": {},
        "asm_to_source": {},
    }

    lines = sass_content.splitlines()
    current_source_line: Optional[int] = None

    for sass_line_num, line in enumerate(lines, start=1):
        match = re.search(r"//##.*line\s+(\d+)", line, re.IGNORECASE)
        if match:
            current_source_line = int(match.group(1))
            continue

        stripped = line.strip()
        if not current_source_line or not stripped:
            continue
        if stripped.startswith("//"):
            continue

        # SASS instructions usually look like: /*addr*/ OP ...
        if re.match(r"/\*[0-9a-fA-Fx]+\*/", stripped):
            mapping["asm_to_source"][sass_line_num] = current_source_line
            src_map = mapping["source_to_asm"].setdefault(current_source_line, [])
            if sass_line_num not in src_map:
                src_map.append(sass_line_num)

    return mapping


def cmd_compile(src: str, arch: str, out_dir: str) -> Dict[str, Any]:
    """Compile a CUDA source file to PTX/SASS and return mapping info."""
    nvcc = _find_nvcc()
    if not nvcc:
        return {
            "success": False,
            "error": "nvcc not found. Install CUDA Toolkit or ensure nvcc is on PATH.",
        }

    nvdisasm = _find_nvdisasm()

    src_path = Path(src)
    if not src_path.is_file():
        return {
            "success": False,
            "error": f"Source file not found: {src}",
        }

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    base = src_path.stem
    ptx_path = out_dir_path / f"{base}.{arch}.ptx"
    cubin_path = out_dir_path / f"{base}.{arch}.cubin"
    sass_path = out_dir_path / f"{base}.{arch}.sass"
    map_path = out_dir_path / f"{base}.{arch}.map.json"

    include_paths = _include_paths_for_repo(src_path)
    include_flags: List[str] = []
    for inc in include_paths:
        include_flags.extend(["-I", inc])

    # Generate PTX with line information
    ptx_cmd = [
        nvcc,
        "-ptx",
        "--generate-line-info",
        "-std=c++17",
        f"-arch={arch}",
        *include_flags,
        str(src_path),
        "-o",
        str(ptx_path),
    ]

    try:
        ptx_res = subprocess.run(
            ptx_cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "PTX compilation timed out."}
    except Exception as exc:
        return {"success": False, "error": f"Failed to invoke nvcc for PTX: {exc}"}

    if ptx_res.returncode != 0:
        return {
            "success": False,
            "error": ptx_res.stderr or ptx_res.stdout or "PTX compilation failed.",
            "stage": "ptx",
        }

    ptx_text = ptx_path.read_text()
    ptx_mapping = parse_ptx_line_mapping(ptx_text)

    sass_text: Optional[str] = None
    sass_mapping: Optional[Dict[str, Any]] = None

    if nvdisasm:
        cubin_cmd = [
            nvcc,
            "-cubin",
            "--generate-line-info",
            "-std=c++17",
            f"-arch={arch}",
            *include_flags,
            str(src_path),
            "-o",
            str(cubin_path),
        ]
        try:
            cubin_res = subprocess.run(
                cubin_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to invoke nvcc for cubin: {exc}",
                "stage": "cubin",
            }

        if cubin_res.returncode == 0:
            sass_cmd = [nvdisasm, "-c", "-g", str(cubin_path)]
            try:
                sass_res = subprocess.run(
                    sass_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except Exception:
                sass_res = None

            if sass_res and sass_res.returncode == 0:
                sass_text = sass_res.stdout
                sass_path.write_text(sass_text)
                sass_mapping = parse_sass_line_mapping(sass_text)

    # Persist mapping JSON to disk for easy inspection in VSCode
    mapping_payload = {
        "ptx_file": str(ptx_path),
        "sass_file": str(sass_path) if sass_text else None,
        "ptx_mapping": ptx_mapping,
        "sass_mapping": sass_mapping,
    }
    map_path.write_text(json.dumps(mapping_payload, indent=2))

    return {
        "success": True,
        "ptx_file": str(ptx_path),
        "sass_file": str(sass_path) if sass_text else None,
        "map_file": str(map_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PTX/SASS compiler + source mapping helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_p = subparsers.add_parser("compile", help="Compile CUDA to PTX/SASS and build mappings")
    compile_p.add_argument("--src", required=True, help="Path to CUDA source (.cu)")
    compile_p.add_argument(
        "--arch",
        default="sm_80",
        help="Target SM architecture (e.g. sm_80, sm_90). Default: sm_80",
    )
    compile_p.add_argument(
        "--out-dir",
        default=".wafer/ptx-sass",
        help="Output directory for PTX/SASS and mapping JSON (default: .wafer/ptx-sass)",
    )

    args = parser.parse_args()

    if args.command == "compile":
        result = cmd_compile(args.src, args.arch, args.out_dir)
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

