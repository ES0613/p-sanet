from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "install_fast_transformers.py"
SPEC = importlib.util.spec_from_file_location("install_fast_transformers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_windows_patch_disables_native_extensions_and_unsafe_attention_imports(tmp_path):
    source = tmp_path / "pytorch-fast-transformers-0.4.0"
    attention = source / "fast_transformers" / "attention"
    builders = source / "fast_transformers" / "builders"
    attention.mkdir(parents=True)
    builders.mkdir(parents=True)
    (source / "setup.py").write_text(
        "setup(\\n" + MODULE.UPSTREAM_EXTENSION_SETTING + "\\n)\\n",
        encoding="utf-8",
    )
    (attention / "__init__.py").write_text(
        "from .clustered_attention import ClusteredAttention\n",
        encoding="utf-8",
    )
    (builders / "__init__.py").write_text(
        "from ..attention import ClusteredAttention, CausalLinearAttention\n",
        encoding="utf-8",
    )

    MODULE.patch_windows_source(source)

    setup_text = (source / "setup.py").read_text(encoding="utf-8")
    attention_text = (attention / "__init__.py").read_text(encoding="utf-8")
    builders_text = (builders / "__init__.py").read_text(encoding="utf-8")
    assert MODULE.WINDOWS_EXTENSION_SETTING in setup_text
    assert MODULE.UPSTREAM_EXTENSION_SETTING not in setup_text
    assert "LinearAttention" in attention_text
    assert "ClusteredAttention" not in attention_text
    assert "LocalAttention" not in attention_text
    assert "LinearAttention" in builders_text
    assert "ClusteredAttention" not in builders_text
    assert "CausalLinearAttention" not in builders_text


def test_archive_hash_rejects_unverified_content(tmp_path):
    archive = tmp_path / MODULE.SOURCE_FILENAME
    archive.write_bytes(b"not the pinned upstream archive")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        MODULE.verify_archive(archive)
