# PhySAFT: Physics-Guided Molecular Thermodynamics

**PhySAFT Thermo** is the public, runnable example of the pure-component thermodynamic framework accompanying **Physics-Guided Learning for Reliable Molecular Thermodynamics Beyond Observed Chemical Space** by **Yi-Cheng Shen, Shu Wang, and Shiang-Tai Lin**. From a SMILES string and temperatures, it predicts vapor pressure, saturated liquid and vapor densities, residual enthalpy and entropy, heat and entropy of vaporization, and critical properties using five neural-network models and PC-SAFT.

The paper has two companion repositories with the same underlying physics-guided architecture:

| Repository | Model training source | Distribution |
| --- | --- | --- |
| **`physaft-thermo`** (this repository) | Quantities derived from publicly available PCP-SAFT pure-substance parameters | Public inference code and five trained model weights |
| **`pcsaft-ml-pvap`** | NIST ThermoData Engine (TDE) data | Restricted to authorized collaborators; its NIST-trained weights and calibration assets are not distributed here |

The authors do not publicly release the NIST TDE-trained models because of the applicable data-use restrictions. This repository lets readers run the proposed inference architecture using models trained from published parameter data. The authors expect that researchers using their own appropriately licensed databases can train models with potentially comparable performance, subject to data quality, coverage, and independent validation. The package includes inference code and converted weights, not the training tables or a complete retraining pipeline.

## Workflow

```text
SMILES -> 100-conformer RDKit geometry + SMI-TED-Light embedding
       -> five independent Gchg/RT networks
       -> per-model epsilon(T) -> PC-SAFT saturation and caloric properties
       -> direct Gchg/RT critical matching -> critical properties
       -> per-model results and five-model summaries
```

Each model has its own predicted Gchg/RT curve and thermodynamic calculation. The five-model spread is an uncalibrated description of model disagreement, not a pressure-error estimate or prediction interval. This repository does not include the restricted release's calibrated vapor-pressure error bands or its Aspen/TDE Wagner comparison workflow.

## Reviewer quick start

For step-by-step setup on Ubuntu x86-64, macOS Apple Silicon, Windows x86-64, or Windows 11 ARM64 with x64 Python, see [REVIEWER_SETUP](docs/REVIEWER_SETUP.md). Use Python 3.11. The pinned SMI-TED-Light assets are downloaded separately (approximately 1.1 GB) and verified by SHA-256.

### Ubuntu x86-64 and macOS

```bash
conda env create -f environment.yml
conda activate physaft-thermo
python tools/install_fast_transformers.py
python -m pip install --no-deps -e .
physaft-thermo download-smi-ted
physaft-thermo predict --smiles 'CCO' --t-min 280 --t-max 320 --n-points 5 --output example_output --strict
```

The SMI-TED-Light checkpoint, loader, and vocabulary come from a pinned IBM Research revision and are hash-verified before use; they are not redistributed in this repository. `tools/install_fast_transformers.py` verifies the official source distribution of the legacy attention dependency after PyTorch has been installed. Linux builds its native extension. macOS installs only the pure-Python attention modules used by SMI-TED-Light, so no Homebrew compiler setup is needed. The checked Linux image is x86-64; other compatible Linux distributions may use the Conda recipe or Docker. Linux ARM64 is not part of the validated platform matrix. CPU-only, headless Linux hosts are supported.

### Windows x86-64 and Windows 11 ARM64

Use the **x86-64 build of Python 3.11** on both hosts. Windows 11 ARM64 runs this x64 Python environment under operating-system emulation. Native ARM64 Python is not supported by the pinned dependency stack. In PowerShell:

```powershell
& "C:\path\to\Python311-x64\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python tools\windows_preflight.py
python tools\install_windows.py
physaft-thermo download-smi-ted
physaft-thermo --help
```

The preflight must report `sysconfig_platform: win-amd64`. On an ARM64 computer, use the full x64 Python path when creating the environment. The Windows installer installs the pinned runtime and the same verified pure-Python SMI-TED attention subset used on macOS.

## Select thermodynamic properties

The default prediction calculates vapor pressure and both saturated phase densities. Select properties with comma-separated `--properties` in the CLI or a list in Python:

| Property name | Reported result |
| --- | --- |
| `vapor_pressure` | Saturation pressure in Pa |
| `liquid_density`, `vapor_density` | Saturated molar and mass densities in mol/m³ and kg/m³ |
| `enthalpy` | Saturated liquid and vapor residual enthalpies in J/mol |
| `entropy` | Saturated liquid and vapor residual entropies in J/(mol K) |
| `heat_of_vaporization` | Difference of saturated vapor and liquid residual enthalpies in J/mol and kJ/mol |
| `entropy_of_vaporization` | Difference of saturated vapor and liquid residual entropies in J/(mol K) |
| `critical_properties` | Per-model Tc, Pc, critical density, Zc, and molar volume |

```bash
physaft-thermo predict --smiles 'CCCCC' --t-min 280 --t-max 350 --n-points 5 \
  --properties vapor_pressure,liquid_density,vapor_density,enthalpy,entropy,heat_of_vaporization,entropy_of_vaporization,critical_properties \
  --output thermodynamic_output --strict
```

Caloric properties reuse internally solved saturation states. `enthalpy` and `entropy` report **residual**, not absolute, phase properties relative to an ideal gas at the same temperature and pressure. Their vaporization differences cancel that reference. Critical properties use direct ML Gchg/RT matching over the model's fixed temperature search domain; the input grid is recorded but does not bound that search. A critical-only request does not calculate saturation on the input grid. Failed states retain explicit statuses, and `--strict` exits nonzero if a requested model or temperature fails. Critical states labeled `closest_point_no_intersection` are approximate constructions, identified separately from accepted roots.

## Python API and batch CLI

```python
from physaft_thermo import predict, predict_range

result = predict_range("CCO", 280.0, 320.0, n_points=5)
result.write("example_output")

same_grid = predict("CCO", [280.0, 290.0, 300.0, 310.0, 320.0])
caloric = predict_range("CCO", 280.0, 320.0, n_points=5,
                        properties=["enthalpy", "entropy", "heat_of_vaporization",
                                    "entropy_of_vaporization"])
critical = predict_range("CCO", 280.0, 320.0, n_points=5,
                         properties=["critical_properties"])
print(caloric.thermodynamics_summary)
print(critical.critical_properties_by_model)
```

```bash
physaft-thermo batch --input examples/batch_input.csv --output batch_output --strict
```

The batch CSV needs `smiles`, `t_min_k`, and `t_max_k`; `n_points` is optional. Inputs must be single-fragment molecules with at least five distinct, finite, positive temperatures. Normal mode records partial results with explicit statuses. The same `--properties` choice works with `batch`.

## Output files

- `geometry.json`: canonical SMILES, mean van der Waals area/volume, PC-SAFT `m` and `sigma`, and conformer diagnostics.
- `gchg_rt.csv`: each model's predicted Gchg/RT over the requested temperature grid when saturation is calculated.
- `epsilon_fits.csv`: per-model `epsilon(T)` coefficients and fit status.
- `pvap_by_model.csv`: vapor pressure, both phase densities, fugacity residual, convergence flag, and solver status.
- `ensemble_summary.csv`: pressure and phase-density means, sample standard deviations, geometric-mean pressure, and successful-model counts. The spread is descriptive and uncalibrated.
- `thermodynamics_by_model.csv`, `thermodynamics_summary.csv`: selected residual enthalpy, residual entropy, heat of vaporization, or entropy of vaporization, with ensemble summaries when requested.
- `critical_properties_by_model.csv`, `critical_properties_summary.csv`: critical-property values, model statuses, and ensemble summary when requested.
- `critical_gchg_curves.csv`, `critical_candidates.csv`, `critical_vle_validation.csv`, `critical_branch_choices.csv`, and `critical_gchg_diagnostics.svg`: critical-root and validation diagnostics when requested.
- `run_metadata.json`: inputs, model manifest, verified SMI-TED hashes, runtime versions, and calculation-stage definitions.

Files for unrequested calculation stages are omitted. Caloric runs retain the saturation files used to obtain the coexistence states. No calibrated pressure-error columns or experimental comparison plots are produced.

The implementation details are described in [METHODS](docs/METHODS.md), [MODEL_CARD](docs/MODEL_CARD.md), and [REPRODUCIBILITY](docs/REPRODUCIBILITY.md). The source publication and third-party assets are identified in [DATA_AVAILABILITY](docs/DATA_AVAILABILITY.md) and [THIRD_PARTY_NOTICES](docs/THIRD_PARTY_NOTICES.md).

## Reproducibility and tests

```bash
python -m pytest -q
python tools/release_audit.py .
physaft-thermo --help
```

The full end-to-end test additionally requires verified SMI-TED-Light assets:

```bash
physaft-thermo download-smi-ted
python -m pytest -q -m integration
```

The GitHub Actions matrix checks Ubuntu x86-64, macOS ARM64, Windows x86-64, and Windows 11 ARM64 with x64 Python. The separate full-integration workflow exercises actual SMI-TED-Light inference. A Linux x86-64 CPU Docker image can be built with `docker build -t physaft-thermo .`. Dependency versions and model-asset hashes are documented in [REPRODUCIBILITY](docs/REPRODUCIBILITY.md).

## Scope and limitations

- The implementation targets pure components. Mixtures, associating systems, ionic species, and process-safety decisions require separate validation.
- The effective non-associating PC-SAFT calculation does not establish accuracy for every chemical family or extrapolated temperature. Unsolved saturation states remain missing with a status.
- Inference uses exactly 100 RDKit conformers and can take longer than descriptor-only prediction.
- Critical results marked `closest_point_no_intersection` are approximate; solver and validation failures remain explicit.
- Five-model standard deviations describe disagreement among these public weights. They are not calibrated uncertainty intervals or substitutes for validation against an independent reference dataset.
- This release supplies no training records, retraining scripts, NIST TDE-trained weights, calibration model, or Wagner reference coefficients.

## Citation and availability

The research framework is described in **Physics-Guided Learning for Reliable Molecular Thermodynamics Beyond Observed Chemical Space** by **Yi-Cheng Shen, Shu Wang, and Shiang-Tai Lin**. Cite the paper when its formal citation becomes available. For this software release, use [CITATION.cff](CITATION.cff).

The public weights were trained from quantities derived from the published PCP-SAFT parameter source: *PCP-SAFT Parameters of Pure Substances Using Large Experimental Databases*, [DOI: 10.1021/acs.iecr.3c02255](https://doi.org/10.1021/acs.iecr.3c02255). See [DATA_AVAILABILITY](docs/DATA_AVAILABILITY.md) for provenance and distribution scope. The separate NIST TDE-trained companion repository is available only to authorized collaborators.

Copyright (c) 2026 Yi-Cheng Shen and Shiang-Tai Lin. The code and converted inference assets are released under [BSD-3-Clause](LICENSE). SMI-TED-Light is fetched from IBM Research and is not redistributed here; see [THIRD_PARTY_NOTICES](docs/THIRD_PARTY_NOTICES.md).
