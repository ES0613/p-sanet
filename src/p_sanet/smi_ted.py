"""Pinned download and loading of IBM Research SMI-TED-Light."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import os
import shutil
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    SMI_TED_FILES,
    SMI_TED_REMOTE_PATHS,
    SMI_TED_REPOSITORY,
    SMI_TED_REVISION,
)
from .exceptions import ModelAssetError
from .utils import sha256_file


def default_smi_ted_dir() -> Path:
    configured = os.environ.get("SMI_TED_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "p_sanet" / "smi_ted_light"


def verify_smi_ted(directory: Path) -> dict[str, str]:
    directory = Path(directory).expanduser().resolve()
    hashes: dict[str, str] = {}
    for filename, expected in SMI_TED_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise ModelAssetError(
                f"Missing SMI-TED asset {filename!r} in {directory}. "
                "Run 'p-sanet download-smi-ted'."
            )
        observed = sha256_file(path)
        if observed != expected:
            raise ModelAssetError(
                f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}."
            )
        hashes[filename] = observed
    return hashes


def download_smi_ted(destination: Path | None = None, source_dir: Path | None = None) -> Path:
    """Install the pinned third-party assets, then verify every SHA-256."""
    destination = Path(destination or default_smi_ted_dir()).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for filename in SMI_TED_FILES:
        target = destination / filename
        temporary = target.with_suffix(target.suffix + ".part")
        if source_dir is not None:
            source = Path(source_dir).expanduser().resolve() / filename
            if not source.is_file():
                raise ModelAssetError(f"Offline source is missing {source}.")
            shutil.copy2(source, temporary)
        else:
            remote_path = SMI_TED_REMOTE_PATHS[filename]
            url = (
                f"https://huggingface.co/{SMI_TED_REPOSITORY}/resolve/"
                f"{SMI_TED_REVISION}/{remote_path}?download=true"
            )
            try:
                urllib.request.urlretrieve(url, temporary)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                raise ModelAssetError(f"Could not download {url}: {exc}") from exc
        observed = sha256_file(temporary)
        expected = SMI_TED_FILES[filename]
        if observed != expected:
            temporary.unlink(missing_ok=True)
            raise ModelAssetError(
                f"Downloaded {filename} failed SHA-256 verification: expected {expected}, observed {observed}."
            )
        temporary.replace(target)
    verify_smi_ted(destination)
    return destination


def load_encoder(directory: Path | None = None) -> Any:
    directory = Path(directory or default_smi_ted_dir()).expanduser().resolve()
    verify_smi_ted(directory)
    spec = importlib.util.spec_from_file_location("p_sanet_smi_ted_load", directory / "load.py")
    if spec is None or spec.loader is None:
        raise ModelAssetError(f"Could not import SMI-TED loader from {directory}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return module.load_smi_ted(
            folder=str(directory),
            ckpt_filename="smi-ted-Light_40.pt",
            vocab_filename="bert_vocab_curated.txt",
        )


def encode_smiles(encoder: Any, nonisomeric_canonical_smiles: str) -> np.ndarray:
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        embedding = encoder.encode(
            [nonisomeric_canonical_smiles],
            batch_size=1,
            return_torch=True,
        )
    values = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
    if values.shape != (1, 768) or not np.all(np.isfinite(values)):
        raise ModelAssetError(f"SMI-TED returned invalid embedding shape/content: {values.shape!r}.")
    return values[0]


__all__ = [
    "default_smi_ted_dir",
    "download_smi_ted",
    "encode_smiles",
    "load_encoder",
    "verify_smi_ted",
]
