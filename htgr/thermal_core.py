"""
htgr/thermal_core.py
--------------------
3-node lumped thermal model of the HTTR reactor core.

Three energy balance ODEs (fuel / moderator / coolant):

    Cf dTf/dt = Q_fission − hfm(Tf−Tm) − hfc(Tf−Tc)
    Cm dTm/dt = hfm(Tf−Tm) − hmc(Tm−Tc)
    Cc dTc/dt = hfc(Tf−Tc) + hmc(Tm−Tc) − 2ṁCp(Tc−Tin)

Temperature feedback is modelled via a simple Doppler + moderator
reactivity coefficient, allowing coupling with the point kinetics model.

Reference: Imperial College HTGR 5 / HR 5 Report, §2.2
"""

import numpy as np
from scipy.integrate import solve_ivp


# ── Default thermal parameters (from group report) ────────────────────────────
DEFAULT_PARAMS = {
    # Thermal capacities (J/K)
    "Cf": 5.0e6,       # Fuel
    "Cm": 1.0e8,       # Moderator (graphite — high heat capacity)
    "Cc": 1.0e6,       # Coolant (helium)
    # Heat transfer coefficients (W/K)
    "h_fm": 2.0e5,     # Fuel → Moderator
    "h_fc": 4.0e5,     # Fuel → Coolant
    "h_mc": 1.5e5,     # Moderator → Coolant
    # Helium coolant
    "Cp_He": 5193.0,   # J/(kg·K) — from CoolProp / NIST
    "m_dot": 105.0,    # kg/s  (10× original HTTR, scaled to 300 MWth)
    # Coolant inlet temperature
    "T_in": 400.0,     # K  (≈ 127 °C)
    # Temperature feedback coefficients
    "alpha_f": -3.8e-5,  # K⁻¹ Doppler (fuel) feedback coefficient
    "alpha_m": -1.5e-5,  # K⁻¹ Moderator feedback coefficient
    # Nominal temperatures (K) for feedback reference
    "Tf_0": 900.0,
    "Tm_0": 620.0,
}


class ThermalCore:
    """
    3-node lumped thermal model for the HTTR core.

    Nodes: fuel (Tf), moderator (Tm), coolant (Tc).

    Parameters
    ----------
    params : dict or None
        Override any default thermal parameter (see DEFAULT_PARAMS).
    Q_nominal : float
        Nominal fission power (W). Default 300e6.

    Examples
    --------
    >>> tc = ThermalCore()
    >>> t, Tf, Tm, Tc = tc.simulate(Q_profile=lambda t: 300e6)
    """

    def __init__(self, params: dict | None = None, Q_nominal: float = 300e6):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.Q_nominal = Q_nominal

    def _steady_state(self) -> tuple[float, float, float]:
        """
        Solve for steady-state temperatures at nominal power.

        Returns
        -------
        Tf, Tm, Tc : float
            Steady-state fuel, moderator, coolant temperatures (K).
        """
        p = self.p
        Q = self.Q_nominal
        # Coolant outlet temperature from energy balance
        Tc = p["T_in"] + Q / (2 * p["m_dot"] * p["Cp_He"])
        Tm = Tc + Q / p["h_mc"]
        Tf = Tm + Q / p["h_fm"]
        return Tf, Tm, Tc

    def _odes(
        self,
        t: float,
        y: np.ndarray,
        Q_func,
    ) -> np.ndarray:
        """Right-hand-side of the 3-node thermal ODE system."""
        Tf, Tm, Tc = y
        p = self.p
        Q_fission = Q_func(t)

        dTf = (Q_fission - p["h_fm"] * (Tf - Tm) - p["h_fc"] * (Tf - Tc)) / p["Cf"]
        dTm = (p["h_fm"] * (Tf - Tm) - p["h_mc"] * (Tm - Tc)) / p["Cm"]
        dTc = (
            p["h_fc"] * (Tf - Tc)
            + p["h_mc"] * (Tm - Tc)
            - 2 * p["m_dot"] * p["Cp_He"] * (Tc - p["T_in"])
        ) / p["Cc"]

        return np.array([dTf, dTm, dTc])

    def simulate(
        self,
        Q_profile,
        t_end: float = 400.0,
        t_eval: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate core thermal response to a given power profile.

        Parameters
        ----------
        Q_profile : callable
            Q(t) → fission power in Watts. Can come directly from
            PointKinetics output via interpolation.
        t_end : float
            Simulation end time (s).
        t_eval : array or None
            Times at which to record temperatures.

        Returns
        -------
        t, Tf, Tm, Tc : np.ndarray
            Time (s), fuel, moderator, and coolant temperatures (K).
        """
        Tf0, Tm0, Tc0 = self._steady_state()

        if t_eval is None:
            t_eval = np.linspace(0, t_end, 1000)

        sol = solve_ivp(
            self._odes,
            [0, t_end],
            [Tf0, Tm0, Tc0],
            args=(Q_profile,),
            t_eval=t_eval,
            method="Radau",
            rtol=1e-6,
            atol=1e-6,
        )

        if not sol.success:
            raise RuntimeError(f"Thermal ODE solver failed: {sol.message}")

        return sol.t, sol.y[0], sol.y[1], sol.y[2]

    def temperature_feedback(self, Tf: float, Tm: float) -> float:
        """
        Compute temperature-dependent reactivity feedback.

        ρ_feedback = αf·(Tf − Tf0) + αm·(Tm − Tm0)

        Parameters
        ----------
        Tf, Tm : float
            Current fuel and moderator temperatures (K).

        Returns
        -------
        float
            Reactivity feedback (dimensionless, negative → stabilising).
        """
        p = self.p
        return p["alpha_f"] * (Tf - p["Tf_0"]) + p["alpha_m"] * (Tm - p["Tm_0"])

    def helium_outlet_temperature(self, Q: float) -> float:
        """
        Estimate steady-state helium outlet temperature at power Q.

        Parameters
        ----------
        Q : float
            Thermal power (W).

        Returns
        -------
        float
            Coolant outlet temperature (K).
        """
        p = self.p
        return p["T_in"] + Q / (p["m_dot"] * p["Cp_He"])
