"""
htgr/kinetics.py
----------------
Point kinetics model for the HTTR reactor core.

Models neutron population dynamics using the standard 6-group
delayed-neutron precursor formulation:

    dP/dt  = [(ρ(t) − β) / Λ] P(t) + Σ λᵢ Cᵢ(t)
    dCᵢ/dt = (βᵢ / Λ) P(t) − λᵢ Cᵢ(t)

where:
    P(t)   = normalised reactor power (P = 1 → 300 MWth nominal)
    ρ(t)   = reactivity (dimensionless)
    β      = total delayed-neutron fraction
    βᵢ     = fraction for precursor group i
    Λ      = prompt neutron lifetime (s)
    λᵢ     = decay constant for group i (s⁻¹)
    Cᵢ(t)  = precursor concentration for group i

Reference: Imperial College HTGR 5 / HR 5 Report, §2.1
           Standard U-235 delayed neutron data (6-group)
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── U-235 delayed neutron parameters (6-group, IAEA standard) ────────────────
BETA_I = np.array([0.00025, 0.00138, 0.00122, 0.00266, 0.00164, 0.00035])
LAMBDA_I = np.array([0.0766, 0.2825, 0.6154, 1.634, 5.176, 16.72])

BETA_TOTAL = BETA_I.sum()          # 0.0075
LAMBDA_PROMPT = 1.0e-3             # s  (graphite-moderated HTGR prompt lifetime)
Q_NOMINAL = 300e6                  # W  (300 MWth)


class PointKinetics:
    """
    6-group point kinetics model for the HTTR.

    Parameters
    ----------
    Q_nominal : float
        Nominal thermal power (W). Default 300e6.
    Lambda : float
        Prompt neutron lifetime (s). Default 1e-3 (graphite HTGR).
    beta_i : array-like of length 6
        Delayed neutron fractions per group. Defaults to U-235 values.
    lambda_i : array-like of length 6
        Decay constants (s⁻¹). Defaults to U-235 values.

    Examples
    --------
    >>> pk = PointKinetics()
    >>> t, P, Q = pk.simulate(t_end=400, rho_step=0.003, step_time=100)
    """

    def __init__(
        self,
        Q_nominal: float = Q_NOMINAL,
        Lambda: float = LAMBDA_PROMPT,
        beta_i: np.ndarray = BETA_I,
        lambda_i: np.ndarray = LAMBDA_I,
    ):
        self.Q_nominal = Q_nominal
        self.Lambda = Lambda
        self.beta_i = np.asarray(beta_i, dtype=float)
        self.lambda_i = np.asarray(lambda_i, dtype=float)
        self.beta = self.beta_i.sum()

    def _steady_state_precursors(self) -> np.ndarray:
        """
        Initial steady-state precursor concentrations at P = 1.

        At steady state: dCᵢ/dt = 0  →  Cᵢ = (βᵢ / Λ) / λᵢ
        """
        return self.beta_i / (self.Lambda * self.lambda_i)

    def _odes(self, t: float, y: np.ndarray, rho_func) -> np.ndarray:
        """
        Right-hand side of the point kinetics ODE system.

        State vector y = [P, C1, C2, C3, C4, C5, C6]
        """
        P = y[0]
        C = y[1:]
        rho = rho_func(t)

        dP_dt = ((rho - self.beta) / self.Lambda) * P + np.dot(self.lambda_i, C)
        dC_dt = (self.beta_i / self.Lambda) * P - self.lambda_i * C

        return np.concatenate([[dP_dt], dC_dt])

    def simulate(
        self,
        t_end: float = 400.0,
        rho_step: float = 0.003,
        step_time: float = 100.0,
        rho_func=None,
        t_eval: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a reactivity transient.

        Parameters
        ----------
        t_end : float
            End time of simulation (s).
        rho_step : float
            Step reactivity insertion at t = step_time.
            Positive → power increase; negative → power decrease.
        step_time : float
            Time at which reactivity step is applied (s).
        rho_func : callable or None
            Custom reactivity function ρ(t). If None, a step function
            at step_time is used.
        t_eval : array or None
            Times at which to record output.

        Returns
        -------
        t : np.ndarray
            Time array (s).
        P : np.ndarray
            Normalised power P(t).
        Q : np.ndarray
            Thermal power Q(t) in Watts.
        """
        if rho_func is None:
            def rho_func(t):
                return rho_step if t >= step_time else 0.0

        # Initial conditions: steady state at P = 1
        C0 = self._steady_state_precursors()
        y0 = np.concatenate([[1.0], C0])

        if t_eval is None:
            t_eval = np.linspace(0, t_end, 1000)

        sol = solve_ivp(
            self._odes,
            [0, t_end],
            y0,
            args=(rho_func,),
            t_eval=t_eval,
            method="Radau",          # stiff solver — appropriate for kinetics
            rtol=1e-6,
            atol=1e-8,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t = sol.t
        P = sol.y[0]
        Q = P * self.Q_nominal
        return t, P, Q

    def reactivity_worth(self, delta_P_target: float) -> float:
        """
        Estimate the reactivity step needed to achieve a fractional power change.

        Uses the in-hour equation approximation for HTGR timescales.

        Parameters
        ----------
        delta_P_target : float
            Desired fractional power change (e.g. 0.05 → +5%).

        Returns
        -------
        float
            Approximate reactivity ρ required (dimensionless).
        """
        # Prompt jump approximation: ΔP/P ≈ Δρ / (β - Δρ) for Δρ << β
        rho_approx = self.beta * delta_P_target / (1 + delta_P_target)
        return rho_approx
