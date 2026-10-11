from pathlib import Path

from physaft_thermo.constants import SMI_TED_FILES, SMI_TED_REMOTE_PATHS, SMI_TED_REVISION
from physaft_thermo.smi_ted import download_smi_ted


def test_download_uses_pinned_remote_paths(tmp_path: Path, monkeypatch) -> None:
    observed_urls: dict[str, str] = {}

    def fake_urlretrieve(url: str, target: Path) -> None:
        filename = Path(target).name.removesuffix(".part")
        observed_urls[filename] = url
        Path(target).write_bytes(filename.encode("utf-8"))

    def fake_sha256(path: Path) -> str:
        filename = Path(path).name.removesuffix(".part")
        return SMI_TED_FILES[filename]

    monkeypatch.setattr(
        "physaft_thermo.smi_ted.urllib.request.urlretrieve",
        fake_urlretrieve,
    )
    monkeypatch.setattr("physaft_thermo.smi_ted.sha256_file", fake_sha256)

    destination = download_smi_ted(tmp_path)

    assert destination == tmp_path.resolve()
    for filename, remote_path in SMI_TED_REMOTE_PATHS.items():
        assert observed_urls[filename].endswith(
            f"/{SMI_TED_REVISION}/{remote_path}?download=true"
        )
