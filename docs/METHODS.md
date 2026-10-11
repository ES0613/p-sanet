# Methods implemented in this release

## Molecular inputs

RDKit validates one molecular fragment and retains specified stereochemistry for 3D geometry. The SMI-TED-Light encoder receives canonical non-isomeric SMILES, matching the training preprocessing. Exactly 100 ETKDGv3 conformers are generated deterministically and UFF optimized. RDKit DoubleCubicLatticeVolume uses atomic VDW radii scaled by 0.856 and zero probe radius. Mean area A and volume V yield `sigma = 6V/A` and `m = A^3/(36 pi V^2)`.

Some training compounds used a separate curated geometry protocol. Their geometry distribution is not guaranteed to be reproduced by the standard inference path; the model card describes this limitation.

## Network and epsilon(T)

Five independent ChargeFusionNet networks receive a 768-dimensional SMI-TED embedding, standardized `[V, A, V/A]`, and standardized `[T, 1/T]`. Each network returns dimensionless Gchg/RT after inverse target scaling. Each model is independently fitted to `epsilon(T) = epsilon0 + epsilon1/T + epsilon2*T` by matching its predicted Gchg/RT to the PC-SAFT dispersion quantity `zdsp + Adsp` at packing fraction 0.4. Constant starts of 200, 300, 450, and 800 K initialize the bounded Powell fit; the lower-MAE valid result is retained. Every requested epsilon must satisfy `1e-8 < epsilon(T) <= 5000 K`.

## Vapor pressure

No experimental vapor pressure, Wagner/DIPPR coefficient, or experimental critical temperature is used as a pressure guess. For each fitted model and temperature, the solver scans a PC-SAFT pressure isotherm for the three-density-root region and solves liquid-vapor fugacity equality with bracketed Brent iteration. It accepts a state only with absolute log-fugacity residual at most `1e-5` and liquid-vapor packing-fraction gap above `1e-12`. Failed states remain missing with a status. The reported five-model pressure spread is uncalibrated.

## Saturation heat of vaporization with epsilon(T)

Let `a=Ares/(RT)` and `a_T=(∂a/∂T)_rho`, at fixed molecular number density and composition. For the non-associating EOS:

```text
epsilon(T) = e0 + e1/T + e2*T
epsilon'(T) = -e1/T² + e2
q_T = epsilon'(T)/T - epsilon(T)/T²
d(T) = sigma * [1 - 0.12 exp(-3 epsilon(T)/T)]
d_T = -0.36 sigma exp(-3 epsilon(T)/T) * [epsilon(T)/T² - epsilon'(T)/T]
Hres = RT * [-T*a_T + Z - 1]
Sres = R * [-T*a_T - a + ln Z]
Gres = Hres - T*Sres = RT * [a + Z - 1 - ln Z].
```

The analytic derivative includes hard-sphere, hard-chain and dispersion terms: temperature dependence of collision diameters, packing-fraction moments, contact values, `I1`, `I2`, `C1`, and both energy moments. Epsilon is clamped to its fitted endpoints outside the fit range with zero epsilon derivative; the endpoint derivative is taken from the interior. Scalar epsilon remains supported. Association derivatives are explicitly unsupported.

Residual H/S use an ideal gas at the same T/P as reference. At accepted pure-component saturation, that reference cancels between phases: `ΔHvap=Hres_vap-Hres_liq` and `ΔSvap=Sres_vap-Sres_liq`. The code reuses accepted pressure/phase densities and reports the identity residual `|ΔHvap/T-ΔSvap|`; it does not estimate ΔSvap merely by dividing ΔHvap by T. Mass density uses RDKit molecular weight with the EOS constants `R=8.314` and `N_A=6.0221367e23`.

## Critical properties

The reduced critical point solves `(dP/deta)_T=0` and `(d²P/deta²)_T=0` by bounded multistart least squares, choosing the highest physically admissible reduced-temperature root. `T_tilde_c` and `eta_c` depend on m; sigma cancels.

The fixed PC-SAFT charging target is constructed at **eta_ref=0.4**, not eta_c:

```text
q_c = 1/T_tilde_c
g_c = g_PC-SAFT(m, q_c) = A_disp_reduced + Z_disp
g_ML(Tc) = g_c.
```

Direct ML roots use a 513-point logarithmic scan on the fixed training-temperature range **11.5002–3728 K**, followed by Brent solves. Admission requires an autograd-positive bracket and positive central differences at relative steps 0.001, 0.003 and 0.01. Absolute root residual tolerance is `2e-5` in Gchg/RT. All candidates and rejection reasons are retained.

At each candidate, ML epsilon is uniquely inverted over `1e-8–5000 K` using a 65-point scan plus Brent at `T/Tc=0.80, 0.85, 0.90, 0.95, 0.99`. All continuation temperatures must lie inside the model domain, with `epsilon(T)/T > q_c`. VLE validation uses the strict isotherm-bracket solver without the multistart fallback. All five continuation states must converge with fugacity residual `<=1e-5`, decreasing density gap, increasing saturation pressure and eta_c bracketed by the final three phase pairs. Linear fits to normalized gap² and pressure use the final three states. Strong evidence requires both R²≥0.95, relative Tc termination error≤0.02 and Pc error≤0.05; supportive evidence uses R²≥0.85, Tc error≤0.05 and Pc error≤0.10.

Among strong/supportive roots, ensemble branch combinations minimize trimmed then full sample standard deviation of ln(Tc), using the benchmark's 10% separation rule and deterministic candidate ordering. Approximate no-root models do not influence assignment of genuine root branches. Unresolved branches or failed VLE gates remain failed.

For an accepted root, `epsilon_required=Tc/T_tilde_c`, `rho_c=eta_c/[(pi/6)*m*d³]`, `Pc=Zc*rho_c*RTc`, and `Vc=1/rho_c` in molar units. Molecular weight converts molar to mass density.

If there is no intersection anywhere in the fixed domain, minimize `|g_ML(T)-g_c|` over both endpoints and every sampled local-minimum basin, refined in log temperature. Ties select lower T. The selected temperature supplies approximate Tc; Pc/rho_c/Zc/Vc use the same reduced critical construction and `epsilon_required`. That epsilon is imposed by the construction and is not claimed to equal ML-inverted or fitted epsilon at a nonzero gap. All constructed properties are marked approximate. Non-finite evaluations, failed Brent/minimization, and intersections failing physical/branch gates are not converted to this fallback. Tangencies are classified as intersections and fail the positive-slope gate.
