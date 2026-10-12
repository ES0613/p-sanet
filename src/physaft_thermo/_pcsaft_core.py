"""PC-SAFT numerical core extracted from the research notebook.

This standalone module retains the equation implementation for numerical
traceability. Validated pure-component functions live in ``pcsaft.py`` and
``solver.py``. The public workflow uses the non-associating pure-component path.
"""

import numpy as np
import math
from scipy.optimize import root



class Component:
    def __init__(
        self,
        PCSAFTm,
        PCSAFTsig,
        PCSAFTeps=None,
        PCSAFTkAB=0,
        PCSAFTeAB=0,
        *,
        PCSAFTeps_coeffs=None,
        PCSAFTeps_Tmin=None,
        PCSAFTeps_Tmax=None,
    ):
        """PC-SAFT component parameters.

        ``PCSAFTeps`` retains the original scalar epsilon/k [K] interface.
        Alternatively, ``PCSAFTeps_coeffs=(e0, e1, e2)`` defines
        epsilon(T)/k = e0 + e1/T + e2*T, with units K, K^2, and 1.
        Coefficient mode uses component-wise endpoint clamping.
        """
        self.PCSAFTm = float(PCSAFTm)
        self.PCSAFTsig = float(PCSAFTsig)
        self.PCSAFTkAB = PCSAFTkAB
        self.PCSAFTeAB = PCSAFTeAB
        self.PCSAFTsigT = None
        self.PCSAFTsigTdT = None
        self.PCSAFTepsdT = None
        self.PCSAFTeps_region = None

        if (PCSAFTeps is None) == (PCSAFTeps_coeffs is None):
            raise ValueError(
                "Provide exactly one of PCSAFTeps or PCSAFTeps_coeffs."
            )

        if PCSAFTeps_coeffs is None:
            epsilon = float(PCSAFTeps)
            if not np.isfinite(epsilon) or epsilon <= 0.0:
                raise ValueError("PCSAFTeps must be finite and positive.")
            self.PCSAFTeps = epsilon
            self.PCSAFTeps_coeffs = None
            self.PCSAFTeps_Tmin = None
            self.PCSAFTeps_Tmax = None
        else:
            if len(PCSAFTeps_coeffs) != 3:
                raise ValueError("PCSAFTeps_coeffs must contain (e0, e1, e2).")
            coeffs = tuple(float(value) for value in PCSAFTeps_coeffs)
            if not all(np.isfinite(value) for value in coeffs):
                raise ValueError("All epsilon(T) coefficients must be finite.")
            if PCSAFTeps_Tmin is None or PCSAFTeps_Tmax is None:
                raise ValueError("Coefficient mode requires PCSAFTeps_Tmin and PCSAFTeps_Tmax.")
            tmin = float(PCSAFTeps_Tmin)
            tmax = float(PCSAFTeps_Tmax)
            if not np.isfinite(tmin) or not np.isfinite(tmax) or not (0.0 < tmin < tmax):
                raise ValueError("Require finite bounds satisfying 0 < Tmin < Tmax.")
            e0, e1, e2 = coeffs
            check_temperatures = [tmin, tmax]
            if e2 != 0.0 and e1 / e2 > 0.0:
                stationary = math.sqrt(e1 / e2)
                if tmin < stationary < tmax:
                    check_temperatures.append(stationary)
            values = [e0 + e1 / temperature + e2 * temperature for temperature in check_temperatures]
            if not all(np.isfinite(value) and value > 0.0 for value in values):
                raise ValueError("epsilon(T)/k must remain finite and positive over [Tmin, Tmax].")
            self.PCSAFTeps = None
            self.PCSAFTeps_coeffs = coeffs
            self.PCSAFTeps_Tmin = tmin
            self.PCSAFTeps_Tmax = tmax

    def evaluate_epsilon_over_k(self, T):
        """Return epsilon(T)/k, its derivative, and the active clamp region.

        At Tmin and Tmax the derivative is the inward one-sided derivative.
        Outside the fitted interval epsilon/k is constant, so its derivative is zero.
        """
        temperature = float(T)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Temperature must be finite and positive.")

        if self.PCSAFTeps_coeffs is None:
            epsilon = float(self.PCSAFTeps)
            derivative = 0.0
            region = "constant"
        else:
            e0, e1, e2 = self.PCSAFTeps_coeffs
            if temperature < self.PCSAFTeps_Tmin:
                evaluation_temperature = self.PCSAFTeps_Tmin
                derivative = 0.0
                region = "below_Tmin_clamped"
            elif temperature > self.PCSAFTeps_Tmax:
                evaluation_temperature = self.PCSAFTeps_Tmax
                derivative = 0.0
                region = "above_Tmax_clamped"
            else:
                evaluation_temperature = temperature
                derivative = -e1 / temperature**2 + e2
                if temperature == self.PCSAFTeps_Tmin:
                    region = "at_Tmin_inward_derivative"
                elif temperature == self.PCSAFTeps_Tmax:
                    region = "at_Tmax_inward_derivative"
                else:
                    region = "within_range"
            epsilon = e0 + e1 / evaluation_temperature + e2 * evaluation_temperature

        if not np.isfinite(epsilon) or epsilon <= 0.0 or not np.isfinite(derivative):
            raise ValueError("Evaluated epsilon(T)/k and its derivative must be finite; epsilon must be positive.")
        self.PCSAFTeps = float(epsilon)
        self.PCSAFTepsdT = float(derivative)
        self.PCSAFTeps_region = region
        return self.PCSAFTeps, self.PCSAFTepsdT, region




class EOS:
    def __init__(self, ncomp, T, x, comp, kij=None):
        self.ncomp = ncomp
        self.T = T
        self.x = x
        self.comp = comp
        self.kij = kij if kij is not None else np.zeros((ncomp, ncomp))
        self.Na = 6.0221367E23                                            # Avogadro's number
        self.R = 8.314                                                    # gas constant J/mol K
        self.KBOL = self.R / self.Na                                      # Boltzmann constant

        self.PCSAFTap = np.array([
            [0.91056314451539, -0.30840169182720, -0.09061483509767],
            [0.63612814494991, 0.18605311591713, 0.45278428063920],
            [2.68613478913903, -2.50300472586548, 0.59627007280101],
            [-26.5473624914884, 21.4197936296668, -1.72418291311787],
            [97.7592087835073, -65.2558853303492, -4.13021125311661],
            [-159.591540865600, 83.3186804808856, 13.7766318697211],
            [91.2977740839123, -33.7469229297323, -8.67284703679646]
        ])

        self.PCSAFTbp = np.array([
            [0.72409469413165, -0.57554980753450, 0.09768831158356],
            [1.11913959304690 * 2.0, 0.34975477607218 * 2.0, -0.12787874908050 * 2.0],
            [-1.33419498282114 * 3.0, 1.29752244631769 * 3.0, -3.05195205099107 * 3.0],
            [-5.25089420371162 * 4.0, -4.30386791194303 * 4.0, 5.16051899359931 * 4.0],
            [5.37112827253230 * 5.0, 38.5344528930499 * 5.0, -7.76088601041257 * 5.0],
            [34.4252230677698 * 6.0, -26.9710769414608 * 6.0, 15.6044623461691 * 6.0],
            [-50.8003365888685 * 7.0, -23.6010990650801 * 7.0, -4.23812936930675 * 7.0]
        ])

        self.sigij = np.zeros((ncomp, ncomp))
        self.epsij = np.zeros((ncomp, ncomp))
        self.depsij_dT = np.zeros((ncomp, ncomp))
        self.eABij = np.zeros((ncomp, ncomp))
        self.kABij = np.zeros((ncomp, ncomp))
        self.sigTij = np.zeros((ncomp, ncomp))
        self.dijab = np.zeros((ncomp, ncomp))
        self.ggij = np.zeros((ncomp, ncomp))
        self.deltaAB = np.zeros((ncomp, ncomp))
        self.dapdx = np.zeros((ncomp, 7))
        self.dbpdx = np.zeros((ncomp, 7))
        self.apar = np.zeros(7)
        self.bpar = np.zeros(7)
        self.xmd3 = 0
        self.num_assoc_components = 0
        self.num_assoc_components = sum(1 for comp in self.comp if comp.PCSAFTeAB != 0)
        # print("self.num_assoc_components = ", self.num_assoc_components)
        self.xAB = np.zeros((2, ncomp))



        for i in range(self.ncomp):
            self.sigij[i][i] = comp[i].PCSAFTsig

        self.z0 = 0.0
        self.z1 = 0.0
        self.z2 = 0.0
        self.z3 = 0.0
        self.rho = 0.0
        self.Ares = 0.0
        self.ahs = 0.0
        self.ahc = 0.0
        self.Adsp = 0.0
        self.Aassoc = 0.0
        self.dAhsdT = np.nan
        self.dAhcdT = np.nan
        self.dAdspdT = np.nan
        self.dAresdT = np.nan
        self.Hres = np.nan
        self.Sres = np.nan
        self.Gresd = np.nan
        self.epsilon_regions = []

        self.Zhs = 0.0
        self.Zhc = 0.0
        self.Zdsp = 0.0
        self.Zres = 0.0
        self.Zges = 0.0
        self.charging_flag = False

        self.G_cav_pcsaft_RT = 0.0

        self.vprfu = 0
        self.lqdfu = 1


    def PCSAFT_parameter(self):
        # Determine epsilon(T)/k and the Barker-Henderson collision diameter.
        self.epsilon_regions = []
        for i in range(self.ncomp):
            epsilon_i, depsilon_i_dT, region = self.comp[i].evaluate_epsilon_over_k(self.T)
            self.epsilon_regions.append(region)
            self.epsij[i][i] = epsilon_i
            self.depsij_dT[i][i] = depsilon_i_dT
            if self.charging_flag:
                self.comp[i].PCSAFTsigT = 0.88 * self.comp[i].PCSAFTsig
                self.comp[i].PCSAFTsigTdT = 0.0
            else:
                exponential = np.exp(-3.0 * epsilon_i / self.T)
                self.comp[i].PCSAFTsigT = self.comp[i].PCSAFTsig * (1.0 - 0.12 * exponential)
                self.comp[i].PCSAFTsigTdT = -0.36 * self.comp[i].PCSAFTsig * exponential * (
                    epsilon_i / self.T**2 - depsilon_i_dT / self.T
                )

            self.sigTij[i][i] = self.comp[i].PCSAFTsigT
            self.kABij[i][i] = self.comp[i].PCSAFTkAB
            self.eABij[i][i] = self.comp[i].PCSAFTeAB


        # Combining rule
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                self.sigij[i][j] = 0.5 * (self.sigij[i][i] + self.sigij[j][j])
                self.sigTij[i][j] = 0.5 * (self.sigTij[i][i] + self.sigTij[j][j])
                self.epsij[i][j] = (1.0 - self.kij[i][j]) * np.sqrt(
                    self.epsij[i][i] * self.epsij[j][j]
                )
                self.depsij_dT[i][j] = 0.5 * self.epsij[i][j] * (
                    self.depsij_dT[i][i] / self.epsij[i][i]
                    + self.depsij_dT[j][j] / self.epsij[j][j]
                )
                self.dijab[i][j] = self.sigTij[i][i] * self.sigTij[j][j] / (self.sigTij[i][i] + self.sigTij[j][j])
                if self.num_assoc_components >= 2:
                    self.kABij[i][j] = np.sqrt(self.kABij[i][i] * self.kABij[j][j]) * \
                    ((np.sqrt(self.sigij[i][i] * self.sigij[j][j]) / self.sigij[i][j]) ** 3.0)
                    self.eABij[i][j] = 0.5 * (self.eABij[i][i] + self.eABij[j][j])

#         print(f"sigTij: {self.sigTij}")
#         print(f"sigij: {self.sigij}")
#         print(f"epsij: {self.epsij}")
#         print(f"dijab: {self.dijab}")
        # print(f"eABij: {self.eABij}")
        # print(f"kABij: {self.kABij}")



        # Calculating zeta parameter
        self.z0t = 0
        self.z1t = 0
        self.z2t = 0
        self.z3t = 0
#         for i in range(self.ncomp):
#             self.z0t += self.x[i] * self.sigij[i][i]
#             self.z1t += self.x[i] * self.sigij[i][i] * np.power(self.sigTij[i][i], 1.0)
#             self.z2t += self.x[i] * self.sigij[i][i] * np.power(self.sigTij[i][i], 2.0)
#             self.z3t += self.x[i] * self.sigij[i][i] * np.power(self.sigTij[i][i], 3.0)
        for i in range(self.ncomp):
#             print(self.x[i], self.comp[i].PCSAFTm, self.comp[i].PCSAFTsigT)
            self.z0t += self.x[i] * self.comp[i].PCSAFTm
            self.z1t += self.x[i] * self.comp[i].PCSAFTm * np.power(self.comp[i].PCSAFTsigT, 1.0)
            self.z2t += self.x[i] * self.comp[i].PCSAFTm * np.power(self.comp[i].PCSAFTsigT, 2.0)
            self.z3t += self.x[i] * self.comp[i].PCSAFTm * np.power(self.comp[i].PCSAFTsigT, 3.0)
        self.PCSAFTmm = self.z0t
        self.z0t *= np.pi / 6.0
        self.z1t *= np.pi / 6.0
        self.z2t *= np.pi / 6.0
        self.z3t *= np.pi / 6.0
#         print(f"Calculated z0t: {self.z0t}, z1t: {self.z1t}, z2t: {self.z2t}, z3t: {self.z3t}")

        # Dispersion term parameters for chain molecules
        for i in range(7):
#             print('ap ', self.PCSAFTap[i][0], self.PCSAFTap[i][1], self.PCSAFTap[i][2])
#             print('bp ', self.PCSAFTbp[i][0], self.PCSAFTbp[i][1], self.PCSAFTbp[i][2])
            self.apar[i] = self.PCSAFTap[i][0] + (1.0 - 1.0 / self.PCSAFTmm) * self.PCSAFTap[i][1] + \
                           (1.0 - 1.0 / self.PCSAFTmm) * (1.0 - 2.0 / self.PCSAFTmm) * self.PCSAFTap[i][2]
            self.bpar[i] =  self.PCSAFTbp[i][0] + (1.0 - 1.0 / self.PCSAFTmm) * self.PCSAFTbp[i][1] + \
                           (1.0 - 1.0 / self.PCSAFTmm) * (1.0 - 2.0 / self.PCSAFTmm) * self.PCSAFTbp[i][2]

#         print(f"apar: {self.apar}")
#         print(f"bpar: {self.bpar}")
        # Derivatives of apar and bpar to mole fraction
        for k in range(self.ncomp):
            for m in range(7):
                self.dapdx[k][m] = (
                    self.comp[k].PCSAFTm / self.PCSAFTmm / self.PCSAFTmm * self.PCSAFTap[m][1] +
                    (3.0 * self.comp[k].PCSAFTm / self.PCSAFTmm / self.PCSAFTmm -
                     4.0 * self.comp[k].PCSAFTm / self.PCSAFTmm / self.PCSAFTmm / self.PCSAFTmm) * self.PCSAFTap[m][2])
                self.dbpdx[k][m] = (
                    self.comp[k].PCSAFTm / self.PCSAFTmm / self.PCSAFTmm * self.PCSAFTbp[m][1] +
                    (3.0 * self.comp[k].PCSAFTm / self.PCSAFTmm / self.PCSAFTmm -
                     4.0 * self.comp[k].PCSAFTm / self.PCSAFTmm / self.PCSAFTmm / self.PCSAFTmm) * self.PCSAFTbp[m][2])

#         print(f"dapdx: {self.dapdx}")
#         print(f"dbpdx: {self.dbpdx}")

        # Van der Waals mixing rules for perturbation terms
        self.order1 = 0.0
        self.order2 = 0.0
        self.dorder1_dT = 0.0
        self.dorder2_dT = 0.0
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                weight = (
                    self.x[i] * self.x[j] * self.comp[i].PCSAFTm * self.comp[j].PCSAFTm
                    * np.power(self.sigij[i][j], 3.0)
                )
                reduced_energy = self.epsij[i][j] / self.T
                dreduced_energy_dT = self.depsij_dT[i][j] / self.T - self.epsij[i][j] / self.T**2
                self.order1 += weight * reduced_energy
                self.order2 += weight * reduced_energy**2
                self.dorder1_dT += weight * dreduced_energy_dT
                self.dorder2_dT += 2.0 * weight * reduced_energy * dreduced_energy_dT

#         print(f"order1: {self.order1}")
#         print(f"order2: {self.order2}")

        self.xmd3 = 0
        for i in range(self.ncomp):
            # print("self.x[i] = ",self.x[i])
            # print("self.comp[i].PCSAFTm = ", self.comp[i].PCSAFTm)
            # print("self.comp[i].PCSAFTsigT = ", self.comp[i].PCSAFTsigT)
            self.xmd3 += self.x[i] * self.comp[i].PCSAFTm * np.power(self.comp[i].PCSAFTsigT, 3.0)
#         print("xmd3", self.xmd3)




        return 0




    def PCSAFT_assoc(self):
        """
        Python implementation of PCSAFT_assoc().
        - For one component, use the analytic association-site solution.
        - For multiple components, solve four association equations iteratively.
        """
        if self.ncomp == 1:
            # Single component: use the analytic solution.
            # xAB(0) = xAB(1) = [1 - sqrt(1 + 4*rho*deltaAB[0][0])] / -2 / (rho*deltaAB[0][0])
            val = (1.0 - math.sqrt(1.0 + 4.0*self.rho*self.deltaAB[0][0])) / (-2.0*self.rho*self.deltaAB[0][0])

            # Store the association-site fractions in component[0].
            # Also store them in EOS.xAB[][].
            self.comp[0].PCSAFTxAB = [ val, val ]   # Store the site fractions on the Component instance.
#             print(f"deltaAB={self.deltaAB[0][0]}, xAB={val}")

            self.xAB[0][0] = val
            self.xAB[1][0] = val
#             print("xAB = ", self.xAB)
#             print(f" xAB0 = {self.xAB[0][0]}, xAB1 = {self.xAB[1][0]}")

        else:




            # Binary or multicomponent system.
            # Assume two components with two association sites each: four unknowns n[0..3].
            # Additional sites would require a higher-dimensional system; this branch handles two components.
            if self.num_assoc_components == 1:
                val = (1.0 - math.sqrt(1.0 + 4.0*self.rho*self.x[0]*self.deltaAB[0][0])) / (-2.0*self.rho*self.x[0]*self.deltaAB[0][0])
                self.xAB[0][0] = val
                self.xAB[0][1] = val
                self.xAB[1][0] = 1
                self.xAB[1][1] = 1
                # print("xAB = ", self.xAB)


            else:
                from scipy.optimize import root
                def F_assoc(n):
                    return [
                        n[0] + self.rho * n[0] * (self.x[0]*n[1]*self.deltaAB[0][0] + self.x[1]*n[3]*self.deltaAB[0][1]) - 1.0,
                        n[1] + self.rho * n[1] * (self.x[0]*n[0]*self.deltaAB[0][0] + self.x[1]*n[2]*self.deltaAB[0][1]) - 1.0,
                        n[2] + self.rho * n[2] * (self.x[0]*n[1]*self.deltaAB[1][0] + self.x[1]*n[3]*self.deltaAB[1][1]) - 1.0,
                        n[3] + self.rho * n[3] * (self.x[0]*n[0]*self.deltaAB[1][0] + self.x[1]*n[2]*self.deltaAB[1][1]) - 1.0 ]
                n_init = [0.9, 0.9, 0.9, 0.9]
                res = root(F_assoc, x0=n_init, method='hybr', tol=1e-6)

                if res.success:
                    self.xAB[0][0], self.xAB[0][1], self.xAB[1][0], self.xAB[1][1] = res.x
                    # print("Assoc converged:", res.x)
                else:
                    print("Warning: Association equations did not converge.")
                    self.xAB[0][0], self.xAB[0][1], self.xAB[1][0], self.xAB[1][1] = n_init  # fallback



                # n = [0.5, 0.5, 1, 1]  # initial guess
                # tol = 1e-15

                # # Calculate the initial residuals F[i].
                # F = [0.0, 0.0, 0.0, 0.0]
                # F[0] = n[0] + self.rho*n[0]*(
                #     self.x[0]*n[1]*self.deltaAB[0][0] +
                #     self.x[1]*n[3]*self.deltaAB[0][1]) - 1.0
                # F[1] = n[1] + self.rho*n[1]*(
                #     self.x[0]*n[0]*self.deltaAB[0][0] +
                #     self.x[1]*n[2]*self.deltaAB[0][1]) - 1.0
                # F[2] = n[2] + self.rho*n[2]*(
                #     self.x[0]*n[1]*self.deltaAB[1][0] +
                #     self.x[1]*n[3]*self.deltaAB[1][1]) - 1.0
                # F[3] = n[3] + self.rho*n[3]*(
                #     self.x[0]*n[0]*self.deltaAB[1][0] +
                #     self.x[1]*n[2]*self.deltaAB[1][1]) - 1.0

                # print("F = ", F[0], F[1], F[2], F[3])

                # # Perform Newton iterations.
                # while True:
                #     # Build the augmented 4x5 matrix.
                #     matrix = np.zeros((4,5), dtype=float)

                #     matrix[0][0] = self.rho*(self.deltaAB[0][0]*n[2]*self.x[0] + self.deltaAB[0][1]*n[3]*self.x[1]) + 1.0
                #     matrix[0][1] = self.deltaAB[0][0]*n[0]*self.rho*self.x[0]
                #     matrix[0][2] = 0.0
                #     matrix[0][3] = self.deltaAB[0][1]*n[0]*self.rho*self.x[1]
                #     matrix[0][4] = -F[0]

                #     matrix[1][0] = self.deltaAB[0][0]*n[2]*self.rho*self.x[0]
                #     matrix[1][1] = self.rho*(self.deltaAB[0][0]*n[0]*self.x[0] + self.deltaAB[0][1]*n[1]*self.x[1]) + 1.0
                #     matrix[1][2] = self.deltaAB[0][1]*n[2]*self.rho*self.x[1]
                #     matrix[1][3] = 0.0
                #     matrix[1][4] = -F[1]

                #     matrix[2][0] = 0.0
                #     matrix[2][1] = self.deltaAB[1][0]*n[1]*self.rho*self.x[0]
                #     matrix[2][2] = self.rho*(self.deltaAB[1][0]*n[2]*self.x[0] + self.deltaAB[1][1]*n[3]*self.x[1]) + 1.0
                #     matrix[2][3] = self.deltaAB[1][1]*n[1]*self.rho*self.x[1]
                #     matrix[2][4] = -F[2]

                #     matrix[3][0] = self.deltaAB[1][0]*n[3]*self.rho*self.x[0]
                #     matrix[3][1] = 0.0
                #     matrix[3][2] = self.deltaAB[1][1]*n[3]*self.rho*self.x[1]
                #     matrix[3][3] = self.rho*(self.deltaAB[1][0]*n[0]*self.x[0] + self.deltaAB[1][1]*n[1]*self.x[1]) + 1.0
                #     matrix[3][4] = -F[3]

                #     # Solve with a linear-system method.
                #     # Follow the original C++ calculation.
                #     # Use NumPy for the linear solve:
                #     A = matrix[:,:4]   # (4x4)
                #     b = -matrix[:,4]   # (4, )

                #     # Try to solve Ax = b.
                #     try:
                #         sol = np.linalg.solve(A, b)  # sol = [xx, yy, zz, uu]
                #     except np.linalg.LinAlgError:
                #         # Handle a singular system.
                #         sol = np.zeros(4, dtype=float)

                #     # Update n.
                #     for i_ in range(4):
                #         n[i_] += sol[i_]

                #     # Update F.
                #     F[0] = n[0] + self.rho*n[0]*(
                #         self.x[0]*n[1]*self.deltaAB[0][0] +
                #         self.x[1]*n[3]*self.deltaAB[0][1]) - 1.0
                #     F[1] = n[1] + self.rho*n[1]*(
                #         self.x[0]*n[0]*self.deltaAB[0][0] +
                #         self.x[1]*n[2]*self.deltaAB[0][1]) - 1.0
                #     F[2] = n[2] + self.rho*n[2]*(
                #         self.x[0]*n[1]*self.deltaAB[1][0] +
                #         self.x[1]*n[3]*self.deltaAB[1][1]) - 1.0
                #     F[3] = n[3] + self.rho*n[3]*(
                #         self.x[0]*n[0]*self.deltaAB[1][0] +
                #         self.x[1]*n[2]*self.deltaAB[1][1]) - 1.0

                #     # Check convergence.
                #     if (abs(F[0]) < tol and abs(F[1]) < tol and
                #         abs(F[2]) < tol and abs(F[3]) < tol):
                #         break

                # # Finally update xAB.
                # self.xAB[0][0] = n[0]
                # self.xAB[0][1] = n[1]
                # self.xAB[1][0] = n[2]
                # self.xAB[1][1] = n[3]
                # # More components or sites require extending this system.
                # # print(f"n= {n}, F={F}")

        # End of association calculation.
        return 0


    def PCSAFT_P(self):
        # Compressibility contributions and their derivatives.
        zhs = zhc = zdsp = 0.0
        zgesdz = zhsdz = zhcdz = zdspdz = 0.0
        dgijdz = np.zeros((self.ncomp, self.ncomp))
        dgijd2 = np.zeros((self.ncomp, self.ncomp))
        dgijdrho = np.zeros((self.ncomp, self.ncomp))
        I2 = edI1dz = edI2dz = edI1d2 = edI2d2 = 0.0
        c1_con = c2_con = c3_con = 0.0
        sumZassoc = zassoc = 0.0

        # Calculate base parameters.
        self.zms = 1.0 - self.dense
        self.rho = self.dense / self.z3t
        self.V = 1.0 / (self.rho / self.Na * 1.0E30)
        self.z0 = self.z0t * self.rho
        self.z1 = self.z1t * self.rho
        self.z2 = self.z2t * self.rho
        self.z3 = self.z3t * self.rho
#         print(f"Base parameters: zms={zms}, rho={rho}, V={self.V}, z0={z0}, z1={z1}, z2={z2}, z3={z3}")


        # Calculate gij and its derivatives.
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                self.ggij[i][j] = 1.0 / self.zms + 3.0 * self.dijab[i][j] * self.z2 / self.zms**2 + 2.0 * (self.dijab[i][j] * self.z2)**2 / self.zms**3
                dgijdz[i][j] = 1.0 / self.zms**2 + 3.0 * self.dijab[i][j] * self.z2 * (1.0 + self.z3) / (self.z3 * self.zms**3) + (self.dijab[i][j] * self.z2 / self.zms**2)**2 * (4.0 + 2.0 * self.z3) / self.z3
                dgijd2[i][j] = 2.0 / self.zms**3 + 6.0 * self.dijab[i][j] * self.z2 / self.z3 / self.zms**4 * (2.0 + self.z3) + (2.0 * self.dijab[i][j] * self.z2 / self.z3)**2 / self.zms**5 * (1.0 + 4.0 * self.z3 + self.z3**2)
#         print(f"gij derivatives: ggij={self.ggij}, dgijdz={dgijdz}, dgijd2={dgijd2}")

        # Calculate the hard-sphere compressibility term and derivative.
        zhs = self.PCSAFTmm * (self.z3 / self.zms + 3.0 * self.z1 * self.z2 / self.z0 / self.zms**2 + self.z2**3 / self.z0 * (3.0 - self.z3) / self.zms**3)
        zhsdz = self.PCSAFTmm * (1.0 / self.zms**2 + 3.0 * self.z1 * self.z2 / self.z0 / self.z3 * (1.0 + self.z3) / self.zms**3 + 6.0 * self.z2**3 / self.z0 / self.z3 / self.zms**4)
#         print(f"Hard-sphere term: zhs={zhs}, zhsdz={zhsdz}")

        # Chain compressibility term and derivative.
        for i in range(self.ncomp):
            zhc += self.x[i] * (1.0 - self.comp[i].PCSAFTm) * self.dense / self.ggij[i][i] * dgijdz[i][i]
            zhcdz += self.x[i] * (1.0 - self.comp[i].PCSAFTm) * (-self.dense * (dgijdz[i][i] / self.ggij[i][i])**2 + dgijdz[i][i] / self.ggij[i][i] + self.dense / self.ggij[i][i] * dgijd2[i][i])
#         print(f"Chain term: zhc={zhc}, zhcdz={zhcdz}")
        # Calculate the dispersion contribution.
        for m in range(7):
            I2 += self.bpar[m] * self.dense**m
            edI1dz += self.apar[m] * (m + 1) * self.dense**m
            edI2dz += self.bpar[m] * (m + 1) * self.dense**m
            edI1d2 += self.apar[m] * (m + 1) * m * self.dense**(m - 1)
            edI2d2 += self.bpar[m] * (m + 1) * m * self.dense**(m - 1)
#         print(f"Dispersion term: I2={I2}, edI1dz={edI1dz}, edI2dz={edI2dz}, edI1d2={edI1d2}, edI2d2={edI2d2}")

        # Calculate dispersion constants.
        c1_con = 1.0 / (1.0 + self.PCSAFTmm * (8.0 * self.dense - 2.0 * self.dense**2) / self.zms**4 + (1.0 - self.PCSAFTmm) * (20.0 * self.dense - 27.0 * self.dense**2 + 12.0 * self.dense**3 - 2.0 * self.dense**4) / (self.zms * (2.0 - self.dense))**2)
        c2_con = -c1_con**2 * (self.PCSAFTmm * (-4.0 * self.dense**2 + 20.0 * self.dense + 8.0) / self.zms**5 + (1.0 - self.PCSAFTmm) * (2.0 * self.dense**3 + 12.0 * self.dense**2 - 48.0 * self.dense + 40.0) / (self.zms * (2.0 - self.dense))**3)
        c3_con = 2.0 * c2_con**2 / c1_con - c1_con**2 * (self.PCSAFTmm * (-12.0 * self.dense**2 + 72.0 * self.dense + 60.0) / self.zms**6 + (1.0 - self.PCSAFTmm) * (-6.0 * self.dense**4 - 48.0 * self.dense**3 + 288.0 * self.dense**2 - 480.0 * self.dense + 264.0) / (self.zms * (2.0 - self.dense))**4)
#         print(f"Dispersion constants: c1_con={c1_con}, c2_con={c2_con}, c3_con={c3_con}")

        # Calculate zdsp and zdspdz.
        zdsp = -2.0 * np.pi * self.rho * edI1dz * self.order1 - np.pi * self.rho * self.order2 * self.PCSAFTmm * (c2_con * I2 * self.dense + c1_con * edI2dz)
        zdspdz = zdsp / self.dense - 2.0 * np.pi * self.rho * edI1d2 * self.order1 - np.pi * self.rho * self.order2 * self.PCSAFTmm * (c3_con * I2 * self.dense + 2.0 * c2_con * edI2dz + c1_con * edI2d2)
#         print(f"Dispersion compressibility factor: zdsp={zdsp}, zdspdz={zdspdz}")
        self.P_pcsaft_z1 = - 2.0 * math.pi * self.rho * edI1dz;
        self.P_pcsaft_z2 = - math.pi * self.rho * self.PCSAFTmm * (c2_con * I2 * self.dense + c1_con * edI2dz);
        # print("self.P_pcsaft_z1 = ", self.P_pcsaft_z1, " ;self.P_pcsaft_z2 = ", self.P_pcsaft_z2)

        # Association term.
        checkeAB = sum(comp.PCSAFTeAB for comp in self.comp)
#         print("checkeAB = ", checkeAB)
#         print("num_assoc_components = ", num_assoc_components)
        if checkeAB == 0:
            zassoc = 0.0
        else:
            for i in range(ncomp):
                for j in range(ncomp):
#                     print("self.sigTij[i][j] = ", self.sigTij[i][j], "self.ggij[i][j] = ", self.ggij[i][j], "self.kABij[i][j] = ", self.kABij[i][j], "self.eABij[i][j] = ", self.eABij[i][j])
                    self.deltaAB[i][j] = self.sigTij[i][j]**3 * self.ggij[i][j] * self.kABij[i][j] * (np.exp(self.eABij[i][j] / self.T) - 1.0)
                    dgijdrho[i][j] = self.z3 / self.zms**2 + self.dijab[i][j] * (3.0 * self.z2 / self.zms**2 + 6.0 * self.z2 * self.z3 / self.zms**3) + self.dijab[i][j]**2 * (4.0 * self.z2**2 / self.zms**3 + 6.0 * self.z2**2 * self.z3 / self.zms**4)
            # print(f"Association calculation: deltaAB={self.deltaAB}, dgijdrho={dgijdrho}")

            self.PCSAFT_assoc()
            sumZassoc = 0.0
            for i in range(ncomp):
                if self.comp[i].PCSAFTeAB == 0:
                    continue
                for j in range(ncomp):
                    if self.comp[i].PCSAFTeAB == 0:
                        continue
                    sumZassoc += 2 * self.x[i] * self.xAB[i][j] * self.x[j] * self.xAB[j][i] * self.deltaAB[i][j] * (1 + dgijdrho[i][j] / self.ggij[i][j])
            zassoc = -self.rho * sumZassoc / 2.0
#         print(f"Association term: zassoc={zassoc}")


        # Calculate the final compressibility factor and pressure.
        self.Zges = 1.0 + zhs + zhc + zdsp + zassoc

        # for cavity free energy 115.03.27
        self.Zhs = zhs
        self.Zhc = zhc
        self.Zdsp = zdsp


        # print("zhs = ", zhs, ";zhc = ", zhc, ";zdsp = ", zdsp, ";Zges = ", self.Zges)
        self.zdsp = zdsp
        self.Zres = zhs + zhc + zdsp + zassoc
        self.pges = self.Zges * self.rho * (self.KBOL * self.T) / 1.0e-30
        zgesdz = zhsdz + zhcdz + zdspdz
        self.pgesdz = (zgesdz * self.rho + self.Zges * self.rho / self.z3) * (self.KBOL * self.T) / 1.0e-30
        # print("Zges = ", self.Zges, ", Zres = ", self.Zres, ", pges = ", self.pges, ", pgesdz = ", self.pgesdz)

#         print(rho)
#         print(f"Final compressibility factor and pressure: Zres={self.Zres}")

        return 0

    def PCSAFT_density(self):
        i = 0
        max_i = 100

        eta_min = 1.0e-30
        eta_max = 0.9

        # strict vs relaxed convergence for low-pressure robustness
        acc_p_strict = 1.0e-8
        acc_eta_strict = 1.0e-18
        acc_p_relaxed = 1.0e-6
        acc_eta_relaxed = 1.0e-14

        x1 = max(self.densta, eta_min)

        converged = False
        converged_strict = False
        y1 = float("inf")
        dx = 0.0

        while True:
            i += 1

            self.dense = x1
            self.PCSAFT_P()

            y1 = (self.pges / self.P) - 1.0
            dydx = self.pgesdz / self.P

            if abs(dydx) < 1.0e-300:
                # Avoid numerical blow-up; accept if residual already small.
                if abs(y1) < acc_p_relaxed:
                    converged = True
                break

            dx = -y1 / dydx

            # Prevent too aggressive Newton step.
            max_step = 0.5 * max(x1, eta_min)
            if dx > max_step:
                dx = max_step
            elif dx < -max_step:
                dx = -max_step

            x_new = x1 + dx

            if x_new > eta_max:
                x_new = 0.6

            if x_new <= eta_min:
                x_new = eta_min

            if abs(y1) < acc_p_strict and abs(dx) < acc_eta_strict:
                x1 = x_new
                converged = True
                converged_strict = True
                break

            if abs(y1) < acc_p_relaxed and abs(dx) < acc_eta_relaxed:
                x1 = x_new
                converged = True
                break

            x1 = x_new

            if i > max_i:
                print(
                    "Warning: PC-SAFT density iteration failed at",
                    self.T,
                    "K and",
                    self.P,
                    "Pa",
                    "last eta =",
                    x1,
                    "residual =",
                    y1,
                    "dx =",
                    dx,
                )
                break

        self.dense = x1
        self.density_converged = converged
        self.density_converged_strict = converged_strict
        self.density_last_residual = y1
        self.density_last_dx = dx
        self.density_iterations = i
        return 0

    def PCSAFT_Ares(self):

        I1 = I2 = c1_con = 0.0
        Ahs = Ahc = Adsp = Aassoc = 0.0

        # Hard sphere contribution
        Ahs = self.PCSAFTmm * (
                3.0 * self.z1 * self.z2 / self.zms +
                self.z2 ** 3.0 / self.z3 / self.zms ** 2 +
                (self.z2 ** 3.0 / self.z3 ** 2 - self.z0) * math.log(self.zms)
            ) / self.z0

        # Chain term
        Ahc = 0.0
        for i in range(self.ncomp):
            Ahc += self.x[i] * (1.0 - self.comp[i].PCSAFTm) * math.log(self.ggij[i][i])

        # Dispersion contribution
        I1 = I2 = 0.0
        for m in range(7):
            I1 += self.apar[m] * self.dense ** m
            I2 += self.bpar[m] * self.dense ** m

        c1_con = 1.0 / (
                1.0 + self.PCSAFTmm * (8.0 * self.dense - 2.0 * self.dense ** 2.0) / self.zms ** 4.0 +
                (1.0 - self.PCSAFTmm) * (
                    20.0 * self.dense - 27.0 * self.dense ** 2.0 + 12.0 * self.dense ** 3.0 - 2.0 * self.dense ** 4.0
                ) / (self.zms * (2.0 - self.dense)) ** 2.0
            )

        Adsp = -2.0 * math.pi * self.rho * I1 * self.order1 - math.pi * self.rho * c1_con * self.PCSAFTmm * I2 * self.order2

        # Association contribution
        Aassoc = 0.0
        checkeAB = 0.0
        for i in range(self.ncomp):
            checkeAB += self.comp[i].PCSAFTeAB

        if checkeAB != 0.0:
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    if self.ncomp == 1:
                        Aassoc = 2.0 * (math.log(self.xAB[i][j]) - (self.xAB[i][j] / 2.0)) + 1.0
                    else:
                        # print("i, j = ", i, " ", j, ", eAB = ", self.comp[j].PCSAFTeAB)
                        if self.comp[i].PCSAFTeAB == 0:
                            continue
                        # print("xAB = ", self.xAB[i][j])
                        Aassoc += self.x[i] * (math.log(self.xAB[i][j]) - (self.xAB[i][j] / 2.0) + 0.5)
                        # print(f"x[{i}] {self.x[i]:e}, xAB {self.xAB[i][j]:e}, Aassoc {Aassoc:e}")

        # Calculate residual Helmholtz free energy.
        self.Ares = Ahs + Ahc + Adsp + Aassoc
        self.ahs = Ahs
        self.ahc = Ahc
        self.Adsp = Adsp
        self.Aassoc = Aassoc
        self.P_pcsaft_a1 = -2.0 * math.pi * self.rho * I1
        self.P_pcsaft_a2 = - math.pi * self.rho * c1_con * self.PCSAFTmm * I2
        # print(f"Ares {self.Ares:e} Ahs {Ahs:e} Ahc {Ahc:e} Adsp {Adsp:e} Aassoc {Aassoc:e}")
        # print("self.P_pcsaft_a1 = ", self.P_pcsaft_a1, " ;self.P_pcsaft_a2 = ", self.P_pcsaft_a2)
        return 0

    def PCSAFT_Hres(self):
        """Calculate non-associating residual H, S, and G with analytic d(ares)/dT.

        The derivative is evaluated at fixed molecular number density and composition,
        as required by Gross and Sadowski (2001), Appendix A. The association
        contribution is intentionally outside the scope of this implementation.
        """
        if self.num_assoc_components != 0:
            raise NotImplementedError(
                "PCSAFT_Hres implements the analytic temperature derivative only for "
                "non-associating PC-SAFT systems."
            )

        # Temperature derivatives of d_i and d_i*d_j/(d_i+d_j).
        sigtdt = np.asarray([component.PCSAFTsigTdT for component in self.comp], dtype=float)
        dijabdt = np.zeros((self.ncomp, self.ncomp))
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                dijabdt[i][j] = self.dijab[i][j] * (
                    sigtdt[i] / self.comp[i].PCSAFTsigT
                    + sigtdt[j] / self.comp[j].PCSAFTsigT
                    - (sigtdt[i] + sigtdt[j])
                    / (self.comp[i].PCSAFTsigT + self.comp[j].PCSAFTsigT)
                )

        # zeta_n,T at fixed rho and x_i.
        z1dt = z2dt = z3dt = 0.0
        for i in range(self.ncomp):
            z1dt += self.x[i] * self.comp[i].PCSAFTm * sigtdt[i]
            z2dt += (
                self.x[i] * self.comp[i].PCSAFTm * 2.0
                * self.comp[i].PCSAFTsigT * sigtdt[i]
            )
            z3dt += (
                self.x[i] * self.comp[i].PCSAFTm * 3.0
                * self.comp[i].PCSAFTsigT**2 * sigtdt[i]
            )
        z1dt *= math.pi / 6.0 * self.rho
        z2dt *= math.pi / 6.0 * self.rho
        z3dt *= math.pi / 6.0 * self.rho

        # g_ij^hs,T (eq A.57 with the full d_i(T) derivative).
        gijdt = np.zeros((self.ncomp, self.ncomp))
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                D = self.dijab[i][j]
                DT = dijabdt[i][j]
                gijdt[i][j] = (
                    z3dt / self.zms**2
                    + 3.0 * (DT * self.z2 + D * z2dt) / self.zms**2
                    + 6.0 * D * self.z2 * z3dt / self.zms**3
                    + 4.0 * D * self.z2 * (DT * self.z2 + D * z2dt) / self.zms**3
                    + 6.0 * (D * self.z2) ** 2 * z3dt / self.zms**4
                )

        # Hard-sphere and hard-chain reference contributions (eqs A.54-A.57).
        dAhsdT = (6.0 / math.pi / self.rho) * (
            3.0 * (z1dt * self.z2 + self.z1 * z2dt) / self.zms
            + 3.0 * self.z1 * self.z2 * z3dt / self.zms**2
            + 3.0 * self.z2**2 * z2dt / self.z3 / self.zms**2
            + self.z2**3 * (2.0 * self.z3 * z3dt - z3dt * self.zms)
            / (self.z3**2 * self.zms**3)
            + (3.0 * self.z2**2 * z2dt * self.z3 - 2.0 * self.z2**3 * z3dt)
            / self.z3**3 * math.log(self.zms)
            + (math.pi / 6.0 * self.rho * self.PCSAFTmm - self.z2**3 / self.z3**2)
            * z3dt / self.zms
        )

        dAhcdT = 0.0
        for i in range(self.ncomp):
            dAhcdT += (
                self.x[i] * (1.0 - self.comp[i].PCSAFTm)
                * gijdt[i][i] / self.ggij[i][i]
            )

        # I_1,T, I_2,T, and C_1,T = C_2*zeta_3,T.
        I1 = I2 = I1dt = I2dt = 0.0
        for power in range(7):
            I1 += self.apar[power] * self.z3**power
            I2 += self.bpar[power] * self.z3**power
            if power > 0:
                I1dt += self.apar[power] * power * self.z3 ** (power - 1) * z3dt
                I2dt += self.bpar[power] * power * self.z3 ** (power - 1) * z3dt

        c1_con = 1.0 / (
            1.0
            + self.PCSAFTmm * (8.0 * self.z3 - 2.0 * self.z3**2) / self.zms**4
            + (1.0 - self.PCSAFTmm)
            * (20.0 * self.z3 - 27.0 * self.z3**2 + 12.0 * self.z3**3 - 2.0 * self.z3**4)
            / (self.zms * (2.0 - self.z3)) ** 2
        )
        c2_con = -c1_con**2 * (
            self.PCSAFTmm * (-4.0 * self.z3**2 + 20.0 * self.z3 + 8.0) / self.zms**5
            + (1.0 - self.PCSAFTmm)
            * (2.0 * self.z3**3 + 12.0 * self.z3**2 - 48.0 * self.z3 + 40.0)
            / (self.zms * (2.0 - self.z3)) ** 3
        )
        c1_dt = c2_con * z3dt

        # Full product rule for temperature-dependent dispersion energy moments.
        dAdspdT = (
            -2.0 * math.pi * self.rho
            * (I1dt * self.order1 + I1 * self.dorder1_dT)
            - math.pi * self.rho * self.PCSAFTmm
            * (
                (c1_dt * I2 + c1_con * I2dt) * self.order2
                + c1_con * I2 * self.dorder2_dT
            )
        )

        dAresdT = dAhsdT + dAhcdT + dAdspdT
        Hres = (-self.T * dAresdT + self.Zres) * self.R * self.T
        Sres = self.R * (-self.T * dAresdT - self.Ares + math.log(self.Zges))
        Gresd = Hres - self.T * Sres

        self.sigtdt = sigtdt
        self.dijabdt = dijabdt
        self.zeta_T = np.asarray([0.0, z1dt, z2dt, z3dt])
        self.gij_dT = gijdt
        self.I1_dT = I1dt
        self.I2_dT = I2dt
        self.C1_dT = c1_dt
        self.dAhsdT = dAhsdT
        self.dAhcdT = dAhcdT
        self.dAdspdT = dAdspdT
        self.dAresdT = dAresdT
        self.Hres = Hres
        self.Sres = Sres
        self.Gresd = Gresd
        return 0


    def cal_fc_PCSAFT(self, fc):
        ncomp = self.ncomp

        # Initialize arrays
        mhs = [0.0 for _ in range(ncomp)]
        mhc = [0.0 for _ in range(ncomp)]
        mdsp = [0.0 for _ in range(ncomp)]
        massoc = [0.0 for _ in range(ncomp)]
        mpart = [0.0 for _ in range(ncomp)]
        myres = [0.0 for _ in range(ncomp)]
        lnphi = [0.0 for _ in range(ncomp)]
        m_mndx = [0.0 for _ in range(ncomp)]
        sumdassocdx = [0.0 for _ in range(ncomp)]
#         sumlnXAk = [0.0 for _ in range(ncomp)]
        sumlnXAk = [0.0, 0.0]
        dgijdx = [[[0.0 for _ in range(ncomp)] for _ in range(ncomp)] for _ in range(ncomp)]

        # Constants for phase
#         vprfu = self.vprfu
#         lqdfu = self.lqdfu

#         # Set initial density guess based on phase
#         if phase == vprfu:
#             self.densta = 1.0E-8
#         elif phase == lqdfu:
#             self.densta = 0.4
#         else:
#             pass

        # Initialize PC-SAFT parameters
        self.PCSAFT_parameter()
        self.PCSAFT_P()
        # Perform density iteration
#         self.PCSAFT_density()
        # Calculate residual Helmholtz energy
        self.PCSAFT_Ares()
        # Caloric derivatives are independent of the fugacity calculation and
        # are evaluated only when PCSAFT_Hres is explicitly requested.

        # Calculation of derivatives of Ares to mole fraction x (d(f)/d(x))
        for k in range(ncomp):
            # Derivative of d(zeta(i))/d(x)
            z0dx = self.rho * math.pi / 6.0 * self.comp[k].PCSAFTm
            z1dx = self.rho * math.pi / 6.0 * self.comp[k].PCSAFTm * self.comp[k].PCSAFTsigT ** 1.0
            z2dx = self.rho * math.pi / 6.0 * self.comp[k].PCSAFTm * self.comp[k].PCSAFTsigT ** 2.0
            z3dx = self.rho * math.pi / 6.0 * self.comp[k].PCSAFTm * self.comp[k].PCSAFTsigT ** 3.0

            # Derivative d(PCSAFTmm)/d(x)
            m_mndx[k] = self.comp[k].PCSAFTm

            # d(f)/d(x) : hard sphere contribution
            fhs_sg = (3.0 * self.z1 * self.z2 / self.zms +
                      self.z2 ** 3.0 / self.z3 / self.zms ** 2 +
                      (self.z2 ** 3.0 / self.z3 ** 2 - self.z0) * math.log(self.zms)) / self.z0

            fhs_sx = (- z0dx / self.z0 * fhs_sg +
                      (3.0 * (z1dx * self.z2 + self.z1 * z2dx) / self.zms +
                       3.0 * self.z1 * self.z2 * z3dx / self.zms ** 2 +
                       3.0 * self.z2 ** 2 * z2dx / self.z3 / self.zms ** 2 +
                       self.z2 ** 3.0 * z3dx * (3.0 * self.z3 - 1.0) / self.z3 ** 2 / self.zms ** 3 +
                       ((3.0 * self.z2 ** 2 * z2dx * self.z3 - 2.0 * self.z2 ** 3.0 * z3dx) / self.z3 ** 3 - z0dx) * math.log(self.zms) +
                       (self.z0 - self.z2 ** 3.0 / self.z3 ** 2) * z3dx / self.zms) / self.z0)

            mhs[k] = m_mndx[k] * fhs_sg + self.PCSAFTmm * fhs_sx
#             print("self.rho = ", self.rho)

#             print(f"z1234 {z0dx} {z1dx} {z2dx} {z3dx}")
#             print(f"{m_mndx[k]} {fhs_sg} {self.PCSAFTmm} {fhs_sx}")

            # d(f)/d(x) : chain term
            for i in range(ncomp):
                for j in range(ncomp):
                    dgijdx[i][j][k] = (z3dx / self.zms ** 2 +
                                       3.0 * self.dijab[i][j] * (z2dx + 2.0 * self.z2 * z3dx / self.zms) / self.zms ** 2 +
                                       self.dijab[i][j] ** 2.0 * self.z2 / self.zms ** 3 * (4.0 * z2dx + 6.0 * self.z2 * z3dx / self.zms))

            mhc[k] = 0.0
            for i in range(ncomp):
                mhc[k] += self.x[i] * (1.0 - self.comp[i].PCSAFTm) * (1.0 / self.ggij[i][i]) * dgijdx[i][i][k]
            mhc[k] += (1.0 - self.comp[k].PCSAFTm) * math.log(self.ggij[k][k])

            # d(f)/d(x) : dispersion contribution
            I1 = I2 = I1_dx = I2_dx = 0.0
            for m in range(7):
                I1 += self.apar[m] * self.dense ** m
                I2 += self.bpar[m] * self.dense ** m
                I1_dx += self.apar[m] * m * self.dense ** (m - 1) * z3dx + self.dapdx[k][m] * self.dense ** m
                I2_dx += self.bpar[m] * m * self.dense ** (m - 1) * z3dx + self.dbpdx[k][m] * self.dense ** m

            ord1dx = ord2dx = 0.0
            for i in range(ncomp):
                ord1dx += 2.0 * self.comp[k].PCSAFTm * self.x[i] * self.comp[i].PCSAFTm * self.sigij[i][k] ** 3.0 * self.epsij[i][k] / self.T
                ord2dx += 2.0 * self.comp[k].PCSAFTm * self.x[i] * self.comp[i].PCSAFTm * self.sigij[i][k] ** 3.0 * (self.epsij[i][k] / self.T) ** 2.0

            c1_con = 1.0 / (
                1.0 + self.PCSAFTmm * (8.0 * self.z3 - 2.0 * self.z3 ** 2) / self.zms ** 4 +
                (1.0 - self.PCSAFTmm) * (20.0 * self.z3 - 27.0 * self.z3 ** 2 + 12.0 * self.z3 ** 3 - 2.0 * self.z3 ** 4) /
                (self.zms * (2.0 - self.z3)) ** 2
            )

            c2_con = -c1_con ** 2 * (
                self.PCSAFTmm * (-4.0 * self.z3 ** 2 + 20.0 * self.z3 + 8.0) / self.zms ** 5 +
                (1.0 - self.PCSAFTmm) * (2.0 * self.z3 ** 3 + 12.0 * self.z3 ** 2 - 48.0 * self.z3 + 40.0) /
                (self.zms * (2.0 - self.z3)) ** 3
            )

            c1_dx = (c2_con * z3dx - c1_con ** 2 * self.comp[k].PCSAFTm * (
                (8.0 * self.z3 - 2.0 * self.z3 ** 2) / self.zms ** 4 -
                (-2.0 * self.z3 ** 4 + 12.0 * self.z3 ** 3 - 27.0 * self.z3 ** 2 + 20.0 * self.z3) /
                (self.zms * (2.0 - self.z3)) ** 2
            ))

            mdsp[k] = (
                -2.0 * math.pi * self.rho * (self.order1 * I1_dx + ord1dx * I1) -
                math.pi * self.rho * c1_con * self.PCSAFTmm * (self.order2 * I2_dx + ord2dx * I2) -
                math.pi * self.rho * (c1_con * self.comp[k].PCSAFTm + c1_dx * self.PCSAFTmm) * self.order2 * I2
            )

            # Association contribution
            checkeAB = 0.0
            for i in range(ncomp):
                checkeAB += self.comp[i].PCSAFTeAB
#             print("checkeAB = ", checkeAB)
            if self.comp[k].PCSAFTeAB is None or self.comp[k].PCSAFTeAB == 0:
                massoc[k] = 0.0
            else:
                sumdassocdx[k] = 0.0
                sumlnXAk[k] = 0.0

                for i in range(self.ncomp):
                    sumlnXAk[0] += math.log(self.xAB[0][i])
                    sumlnXAk[1] += math.log(self.xAB[1][i])

                for i in range(ncomp):
                    if self.comp[i].PCSAFTeAB == 0:
                        continue

                    for j in range(ncomp):
                        if self.comp[j].PCSAFTeAB == 0:
                            continue
                        sumdassocdx[k] += (
                            self.x[i] * self.xAB[i][j] * self.x[j] * self.xAB[j][i] * self.deltaAB[i][j] *
                            (1.0 / self.ggij[i][j]) * dgijdx[i][j][k] * 2.0
                        )

                massoc[k] = sumlnXAk[k] - self.rho * sumdassocdx[k] / 2.0

            # Summation of all contributions
            mpart[k] = mhs[k] + mhc[k] + mdsp[k] + massoc[k]
#             print(f"mpart[{k}]= {mpart[k]:e} mhs[{k}]= {mhs[k]:e} mhc[{k}]= {mhc[k]:e} mdsp[{k}]= {mdsp[k]:e} massoc[{k}]= {massoc[k]:e}")

        # print("massoc = ", massoc)
        myresq = 0.0
        for i in range(ncomp):
            myresq -= self.x[i] * mpart[i]

        for k in range(ncomp):
            myres[k] = myresq + mpart[k] + self.Ares + self.Zres
            lnphi[k] = myres[k] - math.log(self.Zges)
            fc[k] = math.exp(lnphi[k])
#             print(f"fc[{k}] = {fc[k]}")

        return 0




    def _PCSAFT_pressure_at_eta(self, eta):
        old_dense = getattr(self, "dense", None)
        try:
            if eta <= 0.0 or eta >= 0.9:
                return float("nan")
            self.dense = float(eta)
            self.PCSAFT_P()
            p = float(self.pges)
            if not np.isfinite(p):
                return float("nan")
            return p
        except Exception:
            return float("nan")
        finally:
            if old_dense is not None:
                self.dense = old_dense

    def _PCSAFT_density_roots_by_eta_scan(self, P, eta_min=1.0e-14, eta_max=0.895, n_grid=1800):
        self.P = float(P)
        if (not np.isfinite(self.P)) or self.P <= 0.0:
            return []

        eta_grid = np.unique(
            np.concatenate(
                [
                    np.logspace(math.log10(eta_min), -3.0, 240),
                    np.linspace(1.0e-3, eta_max, n_grid),
                ]
            )
        )

        roots = []
        last_eta = None
        last_val = None

        for eta in eta_grid:
            val = self._PCSAFT_pressure_at_eta(eta) - self.P
            if not np.isfinite(val):
                last_eta = None
                last_val = None
                continue

            if last_val is not None:
                if val == 0.0:
                    roots.append(float(eta))
                elif last_val == 0.0:
                    roots.append(float(last_eta))
                elif val * last_val < 0.0:
                    lo = float(last_eta)
                    hi = float(eta)
                    flo = float(last_val)

                    for _ in range(80):
                        mid = 0.5 * (lo + hi)
                        fm = self._PCSAFT_pressure_at_eta(mid) - self.P
                        if not np.isfinite(fm):
                            break
                        if abs(fm) <= max(1.0e-4, 1.0e-10 * self.P):
                            lo = hi = mid
                            break
                        if flo * fm <= 0.0:
                            hi = mid
                        else:
                            lo = mid
                            flo = fm

                    roots.append(0.5 * (lo + hi))

            last_eta = float(eta)
            last_val = float(val)

        roots = sorted(r for r in roots if np.isfinite(r) and r > 0.0 and r < eta_max)
        clustered = []
        for root in roots:
            if not clustered or abs(root - clustered[-1][-1]) > 1.0e-6:
                clustered.append([root])
            else:
                clustered[-1].append(root)

        return [sum(group) / len(group) for group in clustered]

    def _PCSAFT_lnphi_at_eta(self, eta):
        self.dense = float(eta)
        self.PCSAFT_P()
        fc = np.zeros(self.ncomp)
        self.cal_fc_PCSAFT(fc)
        if fc[0] <= 0.0 or (not np.isfinite(fc[0])):
            return float("nan")
        return math.log(fc[0])

    def _PCSAFT_bracketed_pvap_from_eta_roots(self, P_guess, tol_strict, tol_accept, phase_tol):
        """Fallback Pvap solver: scan eta roots at fixed P, then solve lnphiL-lnphiV in log(P)."""
        P_min = 1.0e-300
        P_max = 1.0e10
        P_guess = max(float(P_guess), P_min)

        def eval_at_pressure(P):
            if (not np.isfinite(P)) or P <= 0.0 or P > P_max:
                return None
            self.P = float(P)
            roots = self._PCSAFT_density_roots_by_eta_scan(P)
            if len(roots) < 2:
                return None

            eta_vap = roots[0]
            eta_liq = roots[-1]
            phase_gap = abs(eta_liq - eta_vap)
            if phase_gap <= phase_tol:
                return None

            lnphi_vap = self._PCSAFT_lnphi_at_eta(eta_vap)
            lnphi_liq = self._PCSAFT_lnphi_at_eta(eta_liq)
            if not (np.isfinite(lnphi_liq) and np.isfinite(lnphi_vap)):
                return None

            return {
                "P": float(P),
                "residual": float(lnphi_liq - lnphi_vap),
                "eta_liq": float(eta_liq),
                "eta_vap": float(eta_vap),
                "phase_gap": float(phase_gap),
                "n_roots": int(len(roots)),
            }

        factors = np.unique(
            np.concatenate(
                [
                    np.linspace(0.30, 2.00, 171),
                    np.exp(np.linspace(-3.0, 3.0, 121)),
                ]
            )
        )
        pressures = np.unique(np.clip(P_guess * factors, P_min, P_max))
        pressures.sort()

        states = []
        best_state = None
        for P in pressures:
            state = eval_at_pressure(P)
            if state is None:
                continue
            states.append(state)
            if best_state is None or abs(state["residual"]) < abs(best_state["residual"]):
                best_state = state

        if not states:
            self.highT_pvap_status = "no_two_phase_density_roots"
            return math.nan

        for state in states:
            if abs(state["residual"]) < tol_accept:
                self.eq_liq_density = state["eta_liq"]
                self.eq_vap_density = state["eta_vap"]
                self.pvap_converged_accept = True
                self.pvap_residual = abs(state["residual"])
                self.highT_pvap_status = "accepted_scan_point"
                self.highT_density_root_count = state["n_roots"]
                print(
                    f"Accepted eta-scan Pvap P = {state['P']:.8e} Pa "
                    f"(|lnphiL-lnphiV|={abs(state['residual']):.3e})"
                )
                return state["P"]

        for left, right in zip(states[:-1], states[1:]):
            if left["residual"] == 0.0:
                chosen = left
                self.eq_liq_density = chosen["eta_liq"]
                self.eq_vap_density = chosen["eta_vap"]
                self.pvap_converged_accept = True
                self.pvap_residual = 0.0
                self.highT_pvap_status = "accepted_exact_scan_point"
                return chosen["P"]
            if left["residual"] * right["residual"] > 0.0:
                continue

            lo = left
            hi = right
            best = left if abs(left["residual"]) <= abs(right["residual"]) else right

            for _ in range(80):
                mid_p = math.sqrt(lo["P"] * hi["P"])
                mid = eval_at_pressure(mid_p)
                if mid is None:
                    break
                if abs(mid["residual"]) < abs(best["residual"]):
                    best = mid
                if abs(mid["residual"]) < tol_strict:
                    best = mid
                    break
                if lo["residual"] * mid["residual"] <= 0.0:
                    hi = mid
                else:
                    lo = mid

            if best is not None and abs(best["residual"]) < tol_accept:
                self.eq_liq_density = best["eta_liq"]
                self.eq_vap_density = best["eta_vap"]
                self.pvap_converged_accept = True
                self.pvap_residual = abs(best["residual"])
                self.highT_pvap_status = "accepted_bracketed_eta_scan"
                self.highT_density_root_count = best["n_roots"]
                print(
                    f"Converged eta-scan Pvap P = {best['P']:.8e} Pa "
                    f"(|lnphiL-lnphiV|={abs(best['residual']):.3e})"
                )
                return best["P"]

        if best_state is not None:
            self.best_highT_pvap_pressure = best_state["P"]
            self.best_highT_pvap_residual = abs(best_state["residual"])
            self.best_highT_liq_density = best_state["eta_liq"]
            self.best_highT_vap_density = best_state["eta_vap"]
            self.highT_pvap_status = "two_phase_roots_found_but_no_fugacity_root"

        return math.nan

    def cal_Pvap_PCSAFT(self, P_guess, max_iter=500, tol=1e-8):
        """
        Calculate vapor pressure of pure component using PC-SAFT.
        Robust version with best-iterate fallback for very low vapor pressure.
        """

        P_min = 1.0e-300
        P_max = 1.0e10
        phase_tol = 1.0e-12

        tol_strict = max(float(tol), 1.0e-12)
        tol_accept = max(1.0e-5, 1000.0 * tol_strict)

        P = max(P_guess, P_min)
        lnP = math.log(P)

        self.eq_liq_density = 0.0
        self.eq_vap_density = 0.0

        self.pvap_converged_strict = False
        self.pvap_converged_accept = False
        self.pvap_residual = float("inf")

        self.PCSAFT_parameter()

        best_abs_residual = float("inf")
        best_state = None

        liq_seed = 0.5
        vap_seed = max(self.z3t * P * 1.0e-30 / (self.KBOL * self.T), 1.0e-30)

        for i in range(max_iter):
            P = math.exp(lnP)
            self.P = P

            # -------------------------
            # liquid-like root
            # -------------------------
            fc_liquid = np.zeros(self.ncomp)

            self.densta = max(liq_seed, 1.0e-30)
            self.PCSAFT_density()

            if not getattr(self, "density_converged", True):
                print("Warning: liquid density did not converge at P =", P)

            liq_density_current = self.dense
            if getattr(self, "density_converged", False):
                liq_seed = liq_density_current

            self.cal_fc_PCSAFT(fc_liquid)
            if fc_liquid[0] <= 0.0 or (not np.isfinite(fc_liquid[0])):
                raise ValueError("Non-positive liquid fugacity coefficient encountered.")
            lnphi_liquid = math.log(fc_liquid[0])

            # -------------------------
            # vapor-like root
            # -------------------------
            fc_vapor = np.zeros(self.ncomp)

            eta_ideal = self.z3t * P * 1.0e-30 / (self.KBOL * self.T)
            vap_init = max(eta_ideal, 1.0e-30)
            vap_init = max(vap_init, min(vap_seed, 0.5 * liq_density_current))

            self.densta = vap_init
            self.PCSAFT_density()

            if not getattr(self, "density_converged", True):
                print("Warning: vapor density did not converge at P =", P)

            vap_density_current = self.dense
            if getattr(self, "density_converged", False):
                vap_seed = vap_density_current

            self.cal_fc_PCSAFT(fc_vapor)
            if fc_vapor[0] <= 0.0 or (not np.isfinite(fc_vapor[0])):
                raise ValueError("Non-positive vapor fugacity coefficient encountered.")
            lnphi_vapor = math.log(fc_vapor[0])

            residual = lnphi_liquid - lnphi_vapor
            abs_residual = abs(residual)
            phase_gap = abs(liq_density_current - vap_density_current)

            if np.isfinite(abs_residual) and abs_residual < best_abs_residual:
                best_abs_residual = abs_residual
                best_state = {
                    "P": P,
                    "eta_liq": liq_density_current,
                    "eta_vap": vap_density_current,
                    "residual": residual,
                    "lnphi_liq": lnphi_liquid,
                    "lnphi_vap": lnphi_vapor,
                    "phase_gap": phase_gap,
                    "iteration": i + 1,
                }

            # strict convergence with clear phase separation
            if abs_residual < tol_strict and phase_gap > phase_tol:
                print(f"Converged: Vaporization pressure P = {P:.8e} Pa (iteration: {i+1})")
                print("lnphi_liquid =", lnphi_liquid, "lnphi_vapor =", lnphi_vapor)
                print("eta_liquid =", liq_density_current, "eta_vapor =", vap_density_current)

                self.eq_liq_density = liq_density_current
                self.eq_vap_density = vap_density_current
                self.pvap_converged_strict = True
                self.pvap_converged_accept = True
                self.pvap_residual = abs_residual
                return P

            # damped log-pressure update
            step = residual
            step = max(min(step, 2.0), -2.0)

            lnP_new = lnP + step
            lnP_new = max(min(lnP_new, math.log(P_max)), math.log(P_min))
            lnP = lnP_new

        # No strict convergence: only promote best_state to an equilibrium result
        # when it is close enough to fugacity equality and has distinct roots.
        if best_state is not None:
            self.pvap_residual = abs(best_state["residual"])
            self.best_pvap_pressure = best_state["P"]
            self.best_liq_density = best_state["eta_liq"]
            self.best_vap_density = best_state["eta_vap"]
            self.best_pvap_iteration = best_state["iteration"]
            self.best_pvap_phase_gap = best_state["phase_gap"]

            if best_state["phase_gap"] > phase_tol and abs(best_state["residual"]) < tol_accept:
                self.eq_liq_density = best_state["eta_liq"]
                self.eq_vap_density = best_state["eta_vap"]
                self.pvap_converged_accept = True
                print(
                    f"Accepted near-converged Pvap P = {best_state['P']:.8e} Pa "
                    f"(best iteration: {best_state['iteration']}, |lnphiL-lnphiV|={abs(best_state['residual']):.3e})"
                )
                print("eta_liquid =", best_state["eta_liq"], "eta_vapor =", best_state["eta_vap"])
                return best_state["P"]

            print("Warning: direct Pvap iteration did not converge; trying eta-scan high-temperature fallback.")
            fallback_p = self._PCSAFT_bracketed_pvap_from_eta_roots(P_guess, tol_strict, tol_accept, phase_tol)
            if np.isfinite(fallback_p):
                return fallback_p

            self.eq_liq_density = 0.0
            self.eq_vap_density = 0.0
            self.pvap_converged_accept = False
            print("Warning: Vaporization pressure calculation did not converge to acceptable fugacity residual.")
            print(
                "Best iterate kept for debugging only: P =",
                best_state["P"],
                "Pa, |lnphiL - lnphiV| =",
                abs(best_state["residual"]),
                "phase_gap =",
                best_state["phase_gap"],
            )
            return math.nan

        print("Warning: direct Pvap iteration found no valid iterate; trying eta-scan high-temperature fallback.")
        fallback_p = self._PCSAFT_bracketed_pvap_from_eta_roots(P_guess, tol_strict, tol_accept, phase_tol)
        if np.isfinite(fallback_p):
            return fallback_p

        self.eq_liq_density = 0.0
        self.eq_vap_density = 0.0
        self.pvap_converged_accept = False
        print("Warning: Vaporization pressure calculation did not converge and no valid iterate was found.")
        print("Last P =", math.exp(lnP), "Pa")
        return math.nan

    def calc_binary_molfraction(self, P, z1, tol = 1e-10, max_iter = 500):
        z2 = 1 - z1

        # initial guess of Ki
        K1 = 10     # component1: vapor molefraction/ liquid molefraction
        K2 = 0.1    # component2: vapor molefraction/ liquid molefraction
        for iteration in range(max_iter):
            # solve L(tolerance = 1e-8)
            def equation(L, z1, K1, z2, K2):
                return z1*(1-K1)/(L+K1*(1-L)) + z2*(1-K2)/(L+K2*(1-L))
            initial_guess_L = 0.5

            # Solve the equation numerically
            sol = root(equation, initial_guess_L, args=(z1, K1, z2, K2), tol = 1e-10)
            L = sol.x[0]


            # Calc x1, x2(phase1);y1, y2(phase2) after mass balence
            x1 = z1 / (L + K1 * (1 - L))
            x2 = 1 - x1
            y1 = K1 * x1
            y2 = K2 * x2

            # calc phase1 fugacity (more like liquid)
            eos_phase1 = EOS(ncomp=2, T=self.T, x=[x1,x2], comp=self.comp, kij=self.kij)
            eos_phase1.P = P
            eos_phase1.densta = 0.5  # initial guess of density iteration
            eos_phase1.PCSAFT_parameter()
            eos_phase1.PCSAFT_density()
            fc_phase1 = np.zeros(2)
            eos_phase1.cal_fc_PCSAFT(fc_phase1)
            # print('phase1 xAB = ', eos_phase1.xAB)
            # print('phase1 eAB = ', eos_phase1.eABij)
            # print('phase1 kAB = ', eos_phase1.kABij)


            # calc phase2 fugacity (more like vapor)
            eos_phase2 = EOS(ncomp=2, T=self.T, x=[y1,y2], comp=self.comp, kij=self.kij)
            eos_phase2.P = P
            eos_phase2.densta = 1e-8  # initial guess of density iteration
            eos_phase2.PCSAFT_parameter()
            eos_phase2.PCSAFT_density()
            fc_phase2 = np.zeros(2)
            eos_phase2.cal_fc_PCSAFT(fc_phase2)
            # print('phase2 xAB = ', eos_phase2.xAB)
            # print('phase2 eAB = ', eos_phase2.eABij)
            # print('phase2 kAB = ', eos_phase2.kABij)

            f_I_1 = x1 * fc_phase1[0]
            f_I_2 = x2 * fc_phase1[1]
            f_II_1 = y1 * fc_phase2[0]
            f_II_2 = y2 * fc_phase2[1]

            if abs((f_I_1/f_II_1)-1) <= 1e-10 and abs((f_I_2/f_II_2)-1) <= tol:
                print(f"Flash converged in {iteration+1} iterations at P = {P:.2f} Pa")
                return np.array([x1, y1])
            else:
                K1 = K1*(f_I_1/f_II_1)
                K2 = K2*(f_I_2/f_II_2)

        print("Warning: Flash calculation did not converge within maximum iterations at ", P, "pa")
        return np.array([x1,y1])



    def PCSAFT_Gcav(self):
        self.charging_flag = True
        self.PCSAFT_parameter()
        self.dense = 0.4
        self.PCSAFT_P()
        self.PCSAFT_Ares()
        self.G_cav_pcsaft_RT = ((self.ahs + self.ahc) + (self.Zhs + self.Zhc))
        self.charging_flag = False
        return 0
