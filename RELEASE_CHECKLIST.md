# Release checklist

For a documentation-only update with unchanged code, dependencies, CI workflows, and model assets, review the edited text and run the release audit. The full SMI-TED-Light integration test is for functional releases and need not be repeated solely for documentation changes.

- [ ] Run `python -m pytest -q` in a clean Python 3.11 environment.
- [ ] Run the full SMI-TED-Light integration test and inspect every output file.
- [ ] Check one saturation/heat-of-vaporization run and one critical-properties run with all five weights; inspect per-model statuses and units.
- [ ] Confirm CI passes on Ubuntu x86-64, macOS ARM64, Windows x86-64, and Windows 11 ARM64 using x64 Python.
- [ ] Build and smoke-test the Linux CPU Docker image.
- [ ] Run `python tools/release_audit.py .` and confirm it passes.
- [ ] Confirm the model manifest and five weight hashes remain unchanged.
- [ ] Confirm BSD-3-Clause applies to the code and converted weights, and review third-party notices.
- [ ] Commit and push the checked contents to the existing `ES0613/p-sanet` repository.
