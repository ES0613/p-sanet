#!/usr/bin/env python3
"""Validate the supported Python process architecture on Windows.

P-SANET uses binary wheels pinned for Windows x86-64. Windows 11 ARM64 hosts
are supported through the operating system's x64 emulation, which requires an
x64 Python process. Native ``win-arm64`` Python cannot load ``win-amd64``
PyTorch and RDKit wheels.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import struct
import sys
import sysconfig
from typing import Mapping, Sequence


SUPPORTED_PYTHON = (3, 11)
SUPPORTED_WINDOWS_PLATFORM = "win-amd64"


class WindowsPlatformError(RuntimeError):
    """Raised when the active Python cannot run the pinned Windows stack."""


def normalize_platform_tag(value: str) -> str:
    """Normalize a sysconfig platform tag for stable comparisons."""
    return value.strip().lower().replace("_", "-")


def validate_windows_runtime(
    *,
    sys_platform: str,
    platform_tag: str,
    python_version: Sequence[int],
    pointer_bits: int,
) -> str:
    """Return the supported mode or raise an actionable platform error."""
    normalized_tag = normalize_platform_tag(platform_tag)
    major_minor = tuple(int(part) for part in python_version[:2])
    if sys_platform != "win32":
        raise WindowsPlatformError(
            "The Windows installer must be run from Windows. "
            f"Observed sys.platform={sys_platform!r}."
        )
    if major_minor != SUPPORTED_PYTHON:
        raise WindowsPlatformError(
            "P-SANET requires the x86-64 build of Python 3.11 on Windows. "
            f"Observed Python {major_minor[0]}.{major_minor[1]}."
        )
    if normalized_tag == "win-arm64":
        raise WindowsPlatformError(
            "Native Windows ARM64 Python is not supported in this release because "
            "the pinned PyTorch 2.5.1 and RDKit 2025.09.6 stack does not provide "
            "a complete win-arm64 wheel set. On a Windows 11 ARM64 computer, "
            "install the x86-64 build of Python 3.11 and run P-SANET through "
            "Windows x64 emulation."
        )
    if normalized_tag != SUPPORTED_WINDOWS_PLATFORM or pointer_bits != 64:
        raise WindowsPlatformError(
            "P-SANET requires a 64-bit x86-64 Python process on Windows. "
            f"Observed sysconfig platform {platform_tag!r} and {pointer_bits}-bit pointers."
        )
    return "windows-x64-native-or-emulated"


def runtime_information(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Collect diagnostics without importing any third-party package."""
    environment = os.environ if environ is None else environ
    return {
        "sys_platform": sys.platform,
        "sysconfig_platform": sysconfig.get_platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "process_machine": platform.machine(),
        "pointer_bits": struct.calcsize("P") * 8,
        "processor_architecture": environment.get("PROCESSOR_ARCHITECTURE", ""),
        "processor_architew6432": environment.get("PROCESSOR_ARCHITEW6432", ""),
        "runner_arch": environment.get("RUNNER_ARCH", ""),
    }


def require_supported_windows_python() -> dict[str, object]:
    """Validate the active interpreter and return printable diagnostics."""
    information = runtime_information()
    mode = validate_windows_runtime(
        sys_platform=str(information["sys_platform"]),
        platform_tag=str(information["sysconfig_platform"]),
        python_version=sys.version_info,
        pointer_bits=int(information["pointer_bits"]),
    )
    information["support_mode"] = mode
    return information


def format_runtime_information(information: Mapping[str, object]) -> str:
    """Format diagnostics for installation logs."""
    order = (
        "support_mode",
        "sys_platform",
        "sysconfig_platform",
        "python_version",
        "python_executable",
        "process_machine",
        "pointer_bits",
        "processor_architecture",
        "processor_architew6432",
        "runner_arch",
    )
    return "\n".join(f"{key}: {information.get(key, '')}" for key in order)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that Windows is using x64 Python 3.11. "
            "Windows 11 ARM64 hosts must use x64 emulation."
        )
    )
    parser.add_argument("--json", action="store_true", help="Print diagnostics as JSON.")
    args = parser.parse_args()
    try:
        information = require_supported_windows_python()
    except WindowsPlatformError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(information, indent=2, sort_keys=True))
    else:
        print(format_runtime_information(information))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
