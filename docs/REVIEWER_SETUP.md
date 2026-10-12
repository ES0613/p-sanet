# Reviewer setup: from a new computer to PC-SAFT predictions

This guide runs the public PhySAFT Thermo inference release on a CPU. It uses Python 3.11, the five bundled model weights, and separately downloaded, SHA-256-verified SMI-TED-Light assets. No training data, GPU, or Aspen installation is required.

| Computer | Python process | Section |
| --- | --- | --- |
| Ubuntu x86-64 | Linux x86-64 Python 3.11 | A |
| macOS Apple Silicon | Native ARM64 Python 3.11 | B |
| Windows x86-64 | Windows x64 Python 3.11 (`win-amd64`) | C |
| Windows 11 ARM64 | Windows x64 Python 3.11 through operating-system emulation | D |

These are the platforms covered by the project CI. Other x86-64 Linux distributions may work with equivalent system packages or the Linux Dockerfile; Linux ARM64 and native Windows ARM64 Python are not validated. An internet connection is needed for the Python dependencies and the approximately 1.1 GB SMI-TED-Light download. Allow additional disk space for the Python environment.

## Before installation

The repository is [ES0613/physaft-thermo](https://github.com/ES0613/physaft-thermo). If access is restricted, accept the GitHub invitation before cloning. Alternatively, sign in on the repository page, choose **Code > Download ZIP**, extract it, and use the extracted directory in place of the `git clone` steps below. Run project commands from the directory containing `pyproject.toml`, `requirements-lock.txt`, `tools/`, and `examples/`. A clone has no extra publishing subdirectory.

## A. Ubuntu x86-64

Use Terminal. Install the compiler and Git, then verify the architecture:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates
uname -m
```

`uname -m` should report `x86_64`. Install [Miniforge](https://github.com/conda-forge/miniforge) if Conda is not already available:

```bash
cd "$HOME"
curl -fL -o Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Do not run the installer over an existing `~/miniforge3`; source that installation's `conda.sh` instead. Create the environment and install the repository:

```bash
conda create -n physaft-thermo --platform linux-64 --override-channels -c conda-forge python=3.11.15 pip -y
conda activate physaft-thermo
git clone https://github.com/ES0613/physaft-thermo.git
cd physaft-thermo
python -m pip install --upgrade pip==25.0.1
python -m pip install -r requirements-lock.txt
python tools/install_fast_transformers.py
python -m pip install --no-deps -e .
```

The pinned fast-transformers source builds its native extension on Linux after PyTorch is installed. Continue with section E.

## B. macOS Apple Silicon

Use a native Terminal session. `uname -m` should report `arm64`; if it reports `x86_64` on an M-series Mac, disable **Open using Rosetta** for Terminal and reopen it. Install Apple's command-line tools if Git is unavailable:

```bash
uname -m
xcode-select --install
```

An “already installed” message is fine. Install [Miniforge for macOS ARM64](https://github.com/conda-forge/miniforge) if native ARM64 Conda is not already available:

```bash
cd "$HOME"
curl -fL -o Miniforge3-MacOSX-arm64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Use the existing Conda installation if `~/miniforge3` already exists. Then:

```bash
conda create -n physaft-thermo --platform osx-arm64 --override-channels -c conda-forge python=3.11.15 pip -y
conda activate physaft-thermo
python -c "import platform; assert platform.machine() == 'arm64'"
git clone https://github.com/ES0613/physaft-thermo.git
cd physaft-thermo
python -m pip install --upgrade pip==25.0.1
python -m pip install -r requirements-lock.txt
python tools/install_fast_transformers.py
python -m pip install --no-deps -e .
```

The installer verifies the upstream fast-transformers archive and installs the pure-Python attention modules required by SMI-TED-Light. No Homebrew LLVM or OpenMP setup is needed. Continue with section E.

## C. Windows x86-64

Use **Miniforge Prompt (CMD)**. Install the [Windows x86-64 Miniforge release](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe) first, then create an x64 environment:

```bat
conda create -n physaft-thermo --platform win-64 --override-channels -c conda-forge python=3.11.15 pip git -y
conda activate physaft-thermo
python -c "import sysconfig; assert sysconfig.get_platform() == 'win-amd64'"
cd /d "%USERPROFILE%"
git clone https://github.com/ES0613/physaft-thermo.git
cd physaft-thermo
python tools\windows_preflight.py
python tools\install_windows.py
```

The installer installs the pinned runtime, the verified pure-Python attention dependency, and this package. Continue with section E.

## D. Windows 11 ARM64

Install the **same Windows x86-64 Miniforge release** linked in section C. Run the x64 Miniforge Prompt so the entire Python environment uses Windows x64 emulation:

```bat
conda create -n physaft-thermo --platform win-64 --override-channels -c conda-forge python=3.11.15 pip git -y
conda activate physaft-thermo
python -c "import sysconfig; assert sysconfig.get_platform() == 'win-amd64'"
cd /d "%USERPROFILE%"
git clone https://github.com/ES0613/physaft-thermo.git
cd physaft-thermo
python tools\windows_preflight.py
python tools\install_windows.py
```

The preflight must report `sysconfig_platform: win-amd64` and a 64-bit pointer size. A native `win-arm64` Python process is unsupported. Continue with section E.

## E. Download assets and run predictions

Stay in the repository root with the environment activated. These commands work in the Ubuntu/macOS shell and Windows Miniforge Prompt:

```text
python --version
physaft-thermo download-smi-ted
physaft-thermo --help
physaft-thermo predict --smiles CCO --t-min 280 --t-max 320 --n-points 5 --output reviewer_vle --strict
physaft-thermo predict --smiles CCO --t-min 280 --t-max 320 --n-points 5 --properties heat_of_vaporization --output reviewer_heat --strict
physaft-thermo predict --smiles CCO --t-min 280 --t-max 320 --n-points 5 --properties enthalpy,entropy,heat_of_vaporization,entropy_of_vaporization --output reviewer_caloric --strict
physaft-thermo predict --smiles CCO --t-min 280 --t-max 320 --n-points 5 --properties critical_properties --output reviewer_critical --strict
```

The download verifies the loader, vocabulary, and checkpoint hashes. By default, it installs them under the current user's `~/.cache/physaft_thermo/smi_ted_light` directory (the corresponding user-profile location on Windows). Subsequent predictions reuse these assets. A successful strict command exits with code 0.

`reviewer_vle/ensemble_summary.csv` contains the five-model summary of vapor pressure and saturated liquid/vapor densities. `reviewer_vle/pvap_by_model.csv` contains the individual model states, solver statuses, and pressure in Pa. `reviewer_heat/thermodynamics_summary.csv` reports heat of vaporization in J/mol and kJ/mol. `reviewer_caloric/thermodynamics_summary.csv` additionally reports the requested residual phase enthalpies and entropies and entropy of vaporization. `reviewer_critical/critical_properties_summary.csv` reports the critical-property ensemble, while `critical_properties_by_model.csv` records each model's status. Critical-property search does not use the supplied temperature interval as its search bounds.

The default prediction requests the three saturation properties. Use `--properties` to select any combination of `vapor_pressure`, `liquid_density`, `vapor_density`, `enthalpy`, `entropy`, `heat_of_vaporization`, `entropy_of_vaporization`, and `critical_properties`, separated by commas. The caloric properties internally solve saturated states. Phase enthalpy and entropy are residual values relative to an ideal gas at the same temperature and pressure. To process a CSV of SMILES and temperature ranges, run:

```text
physaft-thermo batch --input examples/batch_input.csv --output reviewer_batch --strict
```

The package does not include a Wagner comparison command, experimental correlation coefficients, or calibrated vapor-pressure error bands. The reported five-model spread describes disagreement among the models; it is not a prediction interval. Review [METHODS.md](METHODS.md), [MODEL_CARD.md](MODEL_CARD.md), and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) before interpreting results.

## Later sessions and common issues

Reopen a terminal, activate the same `physaft-thermo` Conda environment, and enter the cloned repository. On Ubuntu/macOS, source the Conda initialization script first if `conda` is unavailable. On Windows, use Miniforge Prompt.

| Symptom | Check |
| --- | --- |
| GitHub clone reports “repository not found” | Confirm access to the repository or use the authenticated browser's Download ZIP option. |
| `physaft-thermo` command is missing | Activate the project environment and repeat the package installation step from the repository root. |
| SMI-TED files are missing or fail hash verification | Rerun `physaft-thermo download-smi-ted`; verify the network connection and available disk space. |
| Windows preflight reports `win-arm64` | Create a `win-64` Conda environment from x64 Miniforge as shown above. |
| `--strict` exits with code 2 | Inspect each output CSV's `status` and `message` columns; one or more requested model states failed. |

For validation and platform details, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md). This guide documents the inference release and does not require the separate collaborator repository.
