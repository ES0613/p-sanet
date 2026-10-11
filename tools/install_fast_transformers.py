#!/usr/bin/env python3
"""Install the pinned PyTorch Fast Transformers dependency reproducibly.

PyPI publishes version 0.4.0 only as a source distribution.  The upstream
build uses Unix compiler flags and cannot be built unchanged with Windows
MSVC.  The pinned SMI-TED-Light loader uses only pure-Python unmasked linear
attention, so the Windows path installs the verified upstream Python sources
without the unused native extensions.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


VERSION = "0.4.0"
SOURCE_FILENAME = f"pytorch-fast-transformers-{VERSION}.tar.gz"
SOURCE_URL = (
    "https://files.pythonhosted.org/packages/ea/bc/"
    "00f597fefeab6341114c41045c1b232c38436738e0e8eac1bc9d5e9d5962/"
    f"{SOURCE_FILENAME}"
)
SOURCE_SHA256 = "d1826bc31b9dfbcd018998b897667e89fc6566bd3f8c424cda9f0943544f7e90"

UPSTREAM_EXTENSION_SETTING = "        ext_modules=get_extensions(),"
WINDOWS_EXTENSION_SETTING = "        ext_modules=[],"
WINDOWS_ATTENTION_INIT = """\
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

\"\"\"Pure-Python attention modules required by pinned SMI-TED-Light.

The upstream package imports optional clustered/local implementations here,
which in turn require native extensions. SMI-TED-Light uses unmasked linear
attention only; FullAttention is retained as the builder's safe default.
\"\"\"

from .attention_layer import AttentionLayer
from .full_attention import FullAttention
from .linear_attention import LinearAttention

__all__ = [\"AttentionLayer\", \"FullAttention\", \"LinearAttention\"]
"""
WINDOWS_BUILDERS_INIT = """\
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

\"\"\"Builders required by the pinned SMI-TED-Light linear-attention path.\"\"\"

# Import the retained attention implementations so they register themselves.
from ..attention import FullAttention, LinearAttention
del FullAttention, LinearAttention

from .attention_builders import (
    AttentionBuilder,
    RecurrentAttentionBuilder,
    RecurrentCrossAttentionBuilder,
)
from .transformer_builders import (
    TransformerEncoderBuilder,
    RecurrentEncoderBuilder,
    TransformerDecoderBuilder,
    RecurrentDecoderBuilder,
)

__all__ = [
    \"AttentionBuilder\",
    \"RecurrentAttentionBuilder\",
    \"RecurrentCrossAttentionBuilder\",
    \"TransformerEncoderBuilder\",
    \"RecurrentEncoderBuilder\",
    \"TransformerDecoderBuilder\",
    \"RecurrentDecoderBuilder\",
]
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    observed = sha256_file(path)
    if observed != SOURCE_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {SOURCE_SHA256}, observed {observed}."
        )


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe path in source archive: {member.name!r}.")
        handle.extractall(destination)


def patch_windows_source(source_root: Path) -> None:
    """Restrict a verified upstream tree to SMI-TED's pure-Python path."""
    setup_path = source_root / "setup.py"
    setup_text = setup_path.read_text(encoding="utf-8")
    count = setup_text.count(UPSTREAM_EXTENSION_SETTING)
    if count != 1:
        raise RuntimeError(
            "The verified upstream setup.py no longer has the expected extension setting."
        )
    setup_path.write_text(
        setup_text.replace(UPSTREAM_EXTENSION_SETTING, WINDOWS_EXTENSION_SETTING),
        encoding="utf-8",
    )

    attention_init = source_root / "fast_transformers" / "attention" / "__init__.py"
    if not attention_init.is_file():
        raise RuntimeError("The verified upstream attention package is missing.")
    attention_init.write_text(WINDOWS_ATTENTION_INIT, encoding="utf-8")

    builders_init = source_root / "fast_transformers" / "builders" / "__init__.py"
    if not builders_init.is_file():
        raise RuntimeError("The verified upstream builders package is missing.")
    builders_init.write_text(WINDOWS_BUILDERS_INIT, encoding="utf-8")


def prepare_source(archive: Path, destination: Path, windows_compat: bool) -> Path:
    verify_archive(archive)
    destination.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, destination)
    source_root = destination / f"pytorch-fast-transformers-{VERSION}"
    if not source_root.is_dir():
        raise RuntimeError(f"Expected source directory was not extracted: {source_root}.")
    if windows_compat:
        patch_windows_source(source_root)
    return source_root


def _download_archive(destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(SOURCE_URL, temporary)
        verify_archive(temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verify_installed_runtime() -> None:
    code = (
        "import importlib.metadata as m; "
        "from fast_transformers.attention import AttentionLayer, FullAttention, LinearAttention; "
        "from fast_transformers.builders.transformer_builders import BaseTransformerEncoderBuilder; "
        "from fast_transformers.feature_maps import GeneralizedRandomFeatures; "
        "from fast_transformers.masking import LengthMask; "
        f"assert m.version('pytorch-fast-transformers') == '{VERSION}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def install(source_archive: Path | None = None, windows_compat: bool | None = None) -> None:
    if windows_compat is None:
        windows_compat = sys.platform == "win32"
    if windows_compat and sys.platform == "win32":
        from windows_preflight import require_supported_windows_python

        require_supported_windows_python()
    with tempfile.TemporaryDirectory(prefix="physaft-fast-transformers-") as temporary:
        work = Path(temporary)
        archive = work / SOURCE_FILENAME
        if source_archive is None:
            _download_archive(archive)
        else:
            shutil.copy2(source_archive.expanduser().resolve(), archive)
        source_root = prepare_source(archive, work / "source", windows_compat)
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            str(source_root),
        ]
        environment = os.environ.copy()
        environment.setdefault("MAX_JOBS", "2")
        subprocess.run(command, check=True, env=environment)
    _verify_installed_runtime()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=None,
        help="Use an offline copy of the official 0.4.0 source archive.",
    )
    args = parser.parse_args()
    install(args.source_archive)
    mode = "Windows pure-Python compatibility" if sys.platform == "win32" else "upstream native"
    print(f"Installed pytorch-fast-transformers {VERSION} ({mode} mode).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
