# Third-party notices

## SMI-TED-Light

SMI-TED-Light is retrieved from the official IBM Research Hugging Face repository and is not redistributed in this release.

- Repository: <https://huggingface.co/ibm-research/materials.smi-ted>
- Pinned revision: `d823faeb77bf4d6ea7f0795d5cebd5ff2fb32344`
- Model DOI: <https://doi.org/10.57967/hf/2778>
- Upstream license displayed by the repository: Apache-2.0
- `smi-ted-Light_40.pt` SHA-256: `baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375`

The download command also verifies the pinned loader and vocabulary hashes recorded in `physaft_esper_pvap.constants`. Users should review the upstream license and model documentation.

## PyTorch Fast Transformers

SMI-TED-Light uses selected modules from `pytorch-fast-transformers` 0.4.0, distributed by the Idiap Research Institute under the MIT License.

- Repository: <https://github.com/idiap/fast-transformers>
- PyPI release: <https://pypi.org/project/pytorch-fast-transformers/0.4.0/>
- Official source-distribution SHA-256: `d1826bc31b9dfbcd018998b897667e89fc6566bd3f8c424cda9f0943544f7e90`

PyPI provides only a source distribution. On Windows, `tools/install_fast_transformers.py` verifies that archive, disables native extensions not used by the pinned SMI-TED-Light linear-attention path, and narrows `fast_transformers.attention` imports to the pure-Python full and linear implementations. Windows ARM64 computers use an x64 Python 3.11 process under Windows x64 emulation because the pinned PyTorch and RDKit stack does not provide a complete native `win-arm64` wheel set. No third-party archive or compiled binary is redistributed by this repository.

Other Python dependencies retain their respective licenses; see `pyproject.toml` and `environment.yml` for the exact direct dependency set.
