# PhySAFT Esper

PhySAFT Esper predicts selected pure-component PC-SAFT properties from a SMILES string and a temperature grid: vapor pressure, saturated liquid and vapor densities, heat of vaporization, and critical properties. Five neural networks trained on quantities derived from the published Esper PCP-SAFT parameter data predict Gchg/RT. Each network has its own epsilon(T) fit and property calculation; results are summarized across models.

```text
SMILES -> 100-conformer RDKit geometry -> SMI-TED-Light embedding
       -> five Gchg/RT networks -> five epsilon(T) fits
       -> five PC-SAFT saturation, vaporization, or critical calculations
       -> ensemble summaries
```

This is an inference release. It bundles converted model weights, not the source training tables or training pipeline. It does not provide calibrated pressure-error estimates. Between-model spread is a descriptive statistic, not a prediction interval.

## Quick start

Use Python 3.11. The pinned SMI-TED-Light assets are downloaded separately (approximately 1.1 GB) and verified by SHA-256.

### Ubuntu x86-64 and macOS

```bash
conda env create -f environment.yml
conda activate physaft-esper-pvap
python tools/install_fast_transformers.py
python -m pip install --no-deps -e .
physaft-esper-pvap download-smi-ted
physaft-esper-pvap predict --smiles 'CCO' --t-min 280 --t-max 320 --n-points 5 --output example_output
physaft-esper-pvap predict --smiles 'CCO' --t-min 280 --t-max 320 --n-points 5 --properties heat_of_vaporization --output heat_output
physaft-esper-pvap predict --smiles 'CCO' --t-min 280 --t-max 320 --n-points 5 --properties critical_properties --output critical_output
```

On macOS, install `llvm` and `libomp` with Homebrew before building `pytorch-fast-transformers` if they are absent. The checked Linux image is x86-64; other compatible Linux distributions may use the Conda recipe or Docker. Linux ARM64 is not part of the validated platform matrix. CPU-only, headless Linux hosts are supported.

### Windows x86-64 and Windows 11 ARM64

Use the **x86-64 build of Python 3.11** on both hosts. Windows 11 ARM64 runs this x64 Python environment under operating-system emulation. Native ARM64 Python is not supported by the pinned dependency stack. In PowerShell:

```powershell
& "C:\path\to\Python311-x64\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python tools\windows_preflight.py
python tools\install_windows.py
physaft-esper-pvap download-smi-ted
physaft-esper-pvap --help
```

The preflight must report `sysconfig_platform: win-amd64`. On an ARM64 computer, use the full x64 Python path when creating the environment.

## Python API and batch CLI

```python
from physaft_esper_pvap import predict, predict_range

result = predict_range("CCO", 280.0, 320.0, n_points=5)
result.write("example_output")
result = predict("CCO", [280.0, 290.0, 300.0, 310.0, 320.0])
heat = predict_range("CCO", 280.0, 320.0, n_points=5, properties=["heat_of_vaporization"])
critical = predict_range("CCO", 280.0, 320.0, n_points=5, properties=["critical_properties"])
```

```bash
physaft-esper-pvap batch --input examples/batch_input.csv --output batch_output --strict
```

Use `--properties` with one or more comma-separated choices: `vapor_pressure`, `liquid_density`, `vapor_density`, `heat_of_vaporization`, and `critical_properties`. With no selection, the three saturation properties are calculated. Heat of vaporization internally solves saturation states first. Critical properties are calculated over a fixed model search domain; the supplied temperature grid is recorded but does not bound that search.

The batch CSV needs `smiles`, `t_min_k`, and `t_max_k`; `n_points` is optional. Inputs must be single-fragment molecules with at least five distinct, finite, positive temperatures. `--strict` returns a nonzero exit code if any requested model or temperature fails. Normal mode records partial results with explicit statuses.

## Output

A saturation or vaporization run writes `geometry.json`, `gchg_rt.csv`, `epsilon_fits.csv`, `pvap_by_model.csv`, `ensemble_summary.csv`, and `run_metadata.json`. The saturation files include pressure and both phase densities in mol/m³ and kg/m³. The ensemble summary includes arithmetic pressure mean and sample standard deviation, mean and sample standard deviation of ln(P), geometric-mean pressure, phase-density summaries, and successful-model counts. Selecting heat of vaporization adds `thermodynamics_by_model.csv` and `thermodynamics_summary.csv` with J/mol and kJ/mol values.

Selecting critical properties writes `critical_properties_by_model.csv`, `critical_properties_summary.csv`, candidate and validation CSVs, and `critical_gchg_diagnostics.svg`. Accepted exact roots and approximate closest-point results have distinct statuses. Unsolved states remain missing with a failure status. The five-model spread is descriptive and uncalibrated.

The model and inference methods are described in [METHODS](docs/METHODS.md), [MODEL_CARD](docs/MODEL_CARD.md), and [REPRODUCIBILITY](docs/REPRODUCIBILITY.md). The source publication and third-party assets are identified in [DATA_AVAILABILITY](docs/DATA_AVAILABILITY.md) and [THIRD_PARTY_NOTICES](docs/THIRD_PARTY_NOTICES.md).

## Platform checks

```bash
python -m pytest -q
python tools/release_audit.py .
physaft-esper-pvap --help
```

The full integration check additionally requires the separately downloaded SMI-TED-Light assets. A Linux CPU Docker image can be built with `docker build -t physaft-esper-pvap .`.

## Publish to GitHub

This directory is an independent Git repository on branch `main`. To publish it to the existing `ES0613/physaft-thermo` repository:

1. In a terminal, enter this directory and run `python tools/release_audit.py .` and `git status --short`. If Git lacks your author identity, set `git config user.name "Your Name"` and `git config user.email "your GitHub email"` for this repository.
2. Run `git add -A`, then `git commit -m "Release PhySAFT Esper property inference"`.
3. Run `git remote add origin https://github.com/ES0613/physaft-thermo.git`, followed by `git push -u origin main`. Use your normal GitHub authentication flow if prompted.
4. Open the repository page and check that the five safetensors weights, `README.md`, and workflow files appear. If the release should be public, change the repository visibility from Private to Public in GitHub Settings. The collaborator repository is separate and is not changed by these commands.

## Scope and citation

This release addresses pure-component research use. It does not establish accuracy for mixtures, associating fluids, ionic systems, safety design, or temperatures outside the relevant training domain. The source data were derived from the Esper PCP-SAFT publication; see [DATA_AVAILABILITY](docs/DATA_AVAILABILITY.md) for the DOI. Cite this software using `CITATION.cff` and the source publication.

Copyright (c) 2026 Yi-Cheng Shen and Shiang-Tai Lin. Code and converted inference assets are released under BSD-3-Clause. SMI-TED-Light is fetched from IBM Research and is not redistributed here.
