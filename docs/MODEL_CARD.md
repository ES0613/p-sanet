# PhySAFT Esper model card

## Intended use

Five converted neural-network weights predict Gchg/RT for research-oriented pure-component PC-SAFT property inference. The release estimates saturation pressure and phase densities, heat of vaporization, and constructed critical properties per model with ensemble summaries. It does not provide a calibrated pressure error, confidence interval, or accuracy guarantee for a specific chemical family.

## Inputs and assets

Each model uses a 768-dimensional SMI-TED-Light embedding, `[V, A, V/A]` geometry descriptors, and `[T, 1/T]` features. The package contains five safetensors weights and JSON scalers/architecture metadata. The source data are quantities derived from published Esper PCP-SAFT parameters. The source checkpoint files, training records, and local paths are not bundled. SMI-TED-Light is downloaded separately from a pinned IBM Research revision.

## Limitations

The inference geometry path uses deterministic 100-conformer ETKDGv3/UFF descriptors. A subset of the original training structures was curated with a different geometry protocol; performance on such structures needs separate assessment. The effective non-associating PC-SAFT model does not establish reliability for associating fluids, ionic species, mixtures, or extrapolation beyond the model's chemical and temperature domains. Critical states marked `closest_point_no_intersection` are approximate constructions where the model curve did not meet the critical target. Solver failure is reported explicitly. Ensemble spread is not a calibrated uncertainty estimate.
