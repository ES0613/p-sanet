# Data provenance and availability

The five bundled weights were trained on Gchg/RT quantities derived from the published Esper PCP-SAFT pure-substance parameter data. Source publication: *PCP-SAFT Parameters of Pure Substances Using Large Experimental Databases*, DOI: [10.1021/acs.iecr.3c02255](https://doi.org/10.1021/acs.iecr.3c02255). The model target is `Gsol/RT - Gcav/RT`, where the solvation term uses Esper parameters and the cavity term uses the geometry-based PC-SAFT construction described in the methods.

This repository distributes inference code, converted weights, and small example inputs. It does not redistribute the source supporting-information archive, derived training tables, or experimental vapor-pressure correlations. Users should obtain the publication's data from the publisher under its applicable terms. No training or experimental records are needed during inference.
