from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _load(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, TOOLS / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


PREFLIGHT = _load("windows_preflight", "windows_preflight.py")
INSTALLER = _load("install_windows", "install_windows.py")


def test_windows_preflight_accepts_x64_python_311() -> None:
    mode = PREFLIGHT.validate_windows_runtime(
        sys_platform="win32",
        platform_tag="win-amd64",
        python_version=(3, 11, 15),
        pointer_bits=64,
    )
    assert mode == "windows-x64-native-or-emulated"


def test_windows_preflight_rejects_native_arm64_python() -> None:
    with pytest.raises(PREFLIGHT.WindowsPlatformError, match="x86-64 build of Python 3.11"):
        PREFLIGHT.validate_windows_runtime(
            sys_platform="win32",
            platform_tag="win-arm64",
            python_version=(3, 11, 15),
            pointer_bits=64,
        )


def test_windows_preflight_rejects_wrong_python_version() -> None:
    with pytest.raises(PREFLIGHT.WindowsPlatformError, match="Python 3.11"):
        PREFLIGHT.validate_windows_runtime(
            sys_platform="win32",
            platform_tag="win-amd64",
            python_version=(3, 12, 0),
            pointer_bits=64,
        )


def test_windows_preflight_rejects_non_windows_and_32_bit_python() -> None:
    with pytest.raises(PREFLIGHT.WindowsPlatformError, match="must be run from Windows"):
        PREFLIGHT.validate_windows_runtime(
            sys_platform="darwin",
            platform_tag="macosx-15.0-arm64",
            python_version=(3, 11, 15),
            pointer_bits=64,
        )
    with pytest.raises(PREFLIGHT.WindowsPlatformError, match="64-bit x86-64"):
        PREFLIGHT.validate_windows_runtime(
            sys_platform="win32",
            platform_tag="win32",
            python_version=(3, 11, 15),
            pointer_bits=32,
        )


def test_windows_install_command_sequence_uses_one_interpreter(tmp_path: Path) -> None:
    commands = INSTALLER.installation_commands(
        python_executable="X:/Python311/python.exe",
        repository_root=tmp_path,
    )
    assert len(commands) == 4
    assert all(command[0] == "X:/Python311/python.exe" for command in commands)
    assert commands[0][-1] == "pip==25.0.1"
    assert commands[1][-1] == str(tmp_path / "requirements-lock.txt")
    assert commands[2][1] == str(tmp_path / "tools" / "install_fast_transformers.py")
    assert commands[3][-1] == str(tmp_path)
