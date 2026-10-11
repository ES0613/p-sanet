import pytest

from physaft_thermo.cli import main


def test_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "PhySAFT Thermo" in output
    assert "PC-SAFT property inference" in output
    assert "calibrate" not in output


def test_predict_cli_has_model_selection_and_no_legacy_option(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["predict", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--models" in output
    assert "--strict" in output
    assert "--properties" in output
    assert "--no-calibration" not in output
