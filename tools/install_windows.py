#!/usr/bin/env python3
"""Install the pinned P-SANET runtime on supported Windows Python.

Both native Windows x86-64 computers and Windows 11 ARM64 computers using x64
emulation run the same ``win-amd64`` Python 3.11 dependency stack. Native
``win-arm64`` Python is rejected by the preflight before pip changes the
environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


TOOLS_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOLS_DIRECTORY.parent
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIRECTORY))

from windows_preflight import (  # noqa: E402
    WindowsPlatformError,
    format_runtime_information,
    require_supported_windows_python,
)


def installation_commands(
    *,
    python_executable: str,
    repository_root: Path,
    source_archive: Path | None = None,
) -> list[list[str]]:
    """Build the reproducible command sequence for unit testing and execution."""
    commands = [
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip==25.0.1",
        ],
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(repository_root / "requirements-lock.txt"),
        ],
    ]
    fast_transformers = [
        python_executable,
        str(repository_root / "tools" / "install_fast_transformers.py"),
    ]
    if source_archive is not None:
        fast_transformers.extend(
            ["--source-archive", str(source_archive.expanduser().resolve())]
        )
    commands.append(fast_transformers)
    commands.append(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            str(repository_root),
        ]
    )
    return commands


def install(source_archive: Path | None = None) -> None:
    """Run the pinned Windows installation sequence."""
    information = require_supported_windows_python()
    print(format_runtime_information(information), flush=True)
    for command in installation_commands(
        python_executable=sys.executable,
        repository_root=REPOSITORY_ROOT,
        source_archive=source_archive,
    ):
        subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install P-SANET for Windows x86-64, including Windows 11 ARM64 "
            "computers running x64 Python through operating-system emulation."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run the architecture and Python-version preflight without installing.",
    )
    parser.add_argument(
        "--fast-transformers-source-archive",
        type=Path,
        default=None,
        help="Use an offline copy of the verified pytorch-fast-transformers source archive.",
    )
    args = parser.parse_args()
    try:
        information = require_supported_windows_python()
        print(format_runtime_information(information))
        if not args.check_only:
            for command in installation_commands(
                python_executable=sys.executable,
                repository_root=REPOSITORY_ROOT,
                source_archive=args.fast_transformers_source_archive,
            ):
                subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
    except WindowsPlatformError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "Windows platform preflight passed."
        if args.check_only
        else "P-SANET Windows installation completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
