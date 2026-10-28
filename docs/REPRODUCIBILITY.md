# Reproducibility and validation

## Reference runtime

Python 3.11.15; RDKit 2025.09.6; PyTorch 2.5.1; NumPy 2.4.2; SciPy 1.17.1; pandas 3.0.1; transformers 4.46.3; tokenizers 0.20.3; safetensors 0.7.0; pytorch-fast-transformers 0.4.0. The direct runtime packages are pinned in `environment.yml` and `requirements-lock.txt`. The SMI-TED-Light loader, vocabulary, and checkpoint are pinned by revision and SHA-256 in the package constants.

`pytorch-fast-transformers` must be installed after PyTorch because its source build imports PyTorch. Linux builds the original native extension. macOS and Windows install a source-verified, pure-Python compatibility build for the modules required by the pinned SMI-TED-Light loader. This avoids a macOS Clang/libc++ conflict with the pinned PyTorch headers. Windows 11 ARM64 hosts run an x64 Python process and x64 wheels through Windows emulation; native ARM64 Python is not supported.

## Checks

```bash
python -m pytest -q
python tools/release_audit.py .
p-sanet --help
```

The full integration test needs `SMI_TED_HOME` pointing to verified SMI-TED-Light assets. Install them with `p-sanet download-smi-ted`, then run `python -m pytest -q -m integration`. Fixed-input safetensors predictions were checked against the source neural-network state dictionaries with maximum absolute difference at most `1e-6` before publication. The package checks public weight hashes on load.

The public CI checks Ubuntu x86-64, macOS ARM64, Windows x86-64, and Windows 11 ARM64 with x64 Python. The separately triggered integration workflow downloads SMI-TED-Light and runs the full inference path. The Linux Dockerfile targets x86-64 CPU hosts. Other Linux distributions require an equivalent runtime or Docker. Linux ARM64 has not been validated.

## Solver acceptance

The epsilon fit requires `1e-8 < epsilon(T) <= 5000 K` at every requested temperature. Accepted vapor-pressure states require absolute liquid-vapor log-fugacity difference at most `1e-5` and packing-fraction gap above `1e-12`. The code records a status for unsuccessful states and never substitutes a guessed equilibrium pressure.

Residual phase enthalpy and entropy, heat of vaporization, and entropy of vaporization are evaluated from the accepted saturated phase densities using analytic PC-SAFT Helmholtz derivatives, including the epsilon(T) chain rule. The critical solver reports each model's acceptance or approximation status and writes its target curves, candidates, VLE validation, and diagnostic SVG. Use `--properties` to select the requested outputs.
