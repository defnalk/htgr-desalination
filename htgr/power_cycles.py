"""
htgr/power_cycles.py
--------------------
Thermodynamic analysis of the combined Helium Brayton + Steam Rankine
power extraction cycles used in the HTTR-desalination complex.

Thermodynamic state points (from group report, Appendix A):
    Primary (He):  1→2 compressor | 2→3 reactor | 3→4 gas turbine | 4→5 HR
    Secondary (H₂O): A feedwater pump | B→C steam turbine | C→D desalination

Reference: Imperial College HTGR 5 / HR 5 Report, §3.2
"""

import numpy as np

# ── Thermodynamic state points from the report ────────────────────────────────
# Primary helium loop
HE_STATES = {
    1: {"T_C": 30,  "P_bar": 43,  "label": "Compressor inlet"},
    2: {"T_C": 350, "P_bar": 70,  "label": "Compressor outlet"},
    3: {"T_C": 950, "P_bar": 70,  "label": "Reactor outlet"},
    4: {"T_C": 650, "P_bar": 45,  "label": "Gas turbine exit"},
    5: {"T_C": 399, "P_bar": 44,  "label": "HR system inlet"},
    6: {"T_C": 30,  "P_bar": 43,  "label": "HR system outlet"},
}

# Secondary steam/water loop
STEAM_STATES = {
    "A": {"T_C": 70,  "P_bar": 70,   "label": "Feedwater pump outlet"},
    "B": {"T_C": 500, "P_bar": 70,   "label": "Steam turbine inlet"},
    "C": {"T_C": 70,  "P_bar": 0.31, "label": "Steam turbine outlet"},
    "D": {"T_C": 70,  "P_bar": 0.31, "label": "Desalination inlet"},
}

# Carnot efficiencies (from report Table 4)
ETA_CARNOT_BRAYTON  = 0.245   # 24.5%  (He Brayton primary)
ETA_CARNOT_RANKINE  = 0.556   # 55.6%  (Steam Rankine secondary)
ETA_CARNOT_COMBINED = 0.720   # 72%    (combined)


class BraytonRankineCycle:
    """
    Combined Helium Brayton + Steam Rankine cycle analyser.

    Uses ideal-gas approximation for helium and saturation/superheated
    steam tables (simplified polynomial fits) for water/steam.

    Parameters
    ----------
    m_dot_He : float
        Helium mass flow rate (kg/s). Default 105 (scaled HTTR).
    m_dot_steam : float
        Steam mass flow rate (kg/s). Default 12.5.
    eta_turbine : float
        Isentropic turbine efficiency (both loops). Default 0.87.
    eta_compressor : float
        Isentropic compressor efficiency. Default 0.85.
    Q_reactor : float
        Total reactor thermal power (W). Default 300e6.

    Examples
    --------
    >>> cyc = BraytonRankineCycle()
    >>> print(cyc.summary())
    """

    # Helium specific heat (real-gas approximation from CoolProp at HTGR conditions)
    Cp_He_avg = 5193.0    # J/(kg·K)

    # Steam enthalpy values (simplified, consistent with report appendix)
    h_steam_B = 3.42e6    # J/kg  steam turbine inlet (500°C, 70 bar)
    h_steam_C = 2.52e6    # J/kg  steam turbine outlet (70°C, 0.31 bar)
    h_water_A = 0.293e6   # J/kg  feedwater pump outlet (70°C, 70 bar)

    def __init__(
        self,
        m_dot_He: float = 105.0,
        m_dot_steam: float = 12.5,
        eta_turbine: float = 0.87,
        eta_compressor: float = 0.85,
        Q_reactor: float = 300e6,
    ):
        self.m_dot_He = m_dot_He
        self.m_dot_steam = m_dot_steam
        self.eta_turbine = eta_turbine
        self.eta_compressor = eta_compressor
        self.Q_reactor = Q_reactor

    @staticmethod
    def _dT_K(state_a, state_b) -> float:
        """Temperature difference (K) between two HE_STATES entries."""
        return float(HE_STATES[state_a]["T_C"] - HE_STATES[state_b]["T_C"])

    def he_turbine_work(self) -> float:
        """
        Net helium turbine work output.

        W_He,turb = ṁ_He · Cp · (T3 − T4) · η_turbine

        Returns
        -------
        float
            Net turbine work (W).
        """
        return (
            self.m_dot_He * self.Cp_He_avg * self._dT_K(3, 4) * self.eta_turbine
        )

    def he_compressor_work(self) -> float:
        """
        Helium compressor work input.

        W_comp = ṁ_He · Cp · (T2 − T1) / η_compressor

        Returns
        -------
        float
            Compressor work (W). Positive = work input to system.
        """
        return (
            self.m_dot_He * self.Cp_He_avg * self._dT_K(2, 1) * self.eta_compressor
        )

    def net_brayton_work(self) -> float:
        """Net electrical output from the Brayton cycle (W)."""
        return self.he_turbine_work() - self.he_compressor_work()

    def steam_turbine_work(self) -> float:
        """
        Steam Rankine cycle turbine work.

        W_steam = ṁ_steam · (h_B − h_C) · η_turbine

        Returns
        -------
        float
            Steam turbine work (W).
        """
        return (
            self.m_dot_steam
            * (self.h_steam_B - self.h_steam_C)
            * self.eta_turbine
        )

    def hrsg_heat_transfer(self) -> float:
        """
        Heat transferred in the Heat Recovery Steam Generator (HRSG).

        Energy balance: ṁ_He · Cp · (T4 − T_HRSG_out) = ṁ_steam · (h_B − h_A)

        Returns
        -------
        float
            HRSG heat duty (W).
        """
        return self.m_dot_steam * (self.h_steam_B - self.h_water_A)

    def thermal_efficiency(self) -> float:
        """
        Overall cycle thermal efficiency.

        η = (W_net_Brayton + W_steam) / Q_reactor

        Returns
        -------
        float
            Combined cycle efficiency (0–1).
        """
        W_net = self.net_brayton_work() + self.steam_turbine_work()
        return W_net / self.Q_reactor

    def energy_utilisation_factor(self, Q_desal: float) -> float:
        """
        Energy Utilisation Factor (EUF) for the cogeneration system.

        EUF = (W_elec + Q_useful) / Q_fuel

        Parameters
        ----------
        Q_desal : float
            Thermal energy delivered to desalination (W).

        Returns
        -------
        float
            EUF (dimensionless; higher → better exergy utilisation).
        """
        W_elec = self.net_brayton_work() + self.steam_turbine_work()
        return (W_elec + Q_desal) / self.Q_reactor

    def carnot_efficiency(self, T_hot_C: float, T_cold_C: float) -> float:
        """
        Carnot efficiency between two temperature limits.

        Parameters
        ----------
        T_hot_C, T_cold_C : float
            Hot and cold reservoir temperatures (°C).

        Returns
        -------
        float
            Carnot efficiency.
        """
        T_hot  = T_hot_C  + 273.15
        T_cold = T_cold_C + 273.15
        return 1 - T_cold / T_hot

    def summary(self) -> str:
        """Print a formatted cycle performance summary."""
        W_brayton = self.net_brayton_work() / 1e6
        W_rankine = self.steam_turbine_work() / 1e6
        W_total   = W_brayton + W_rankine
        eta       = self.thermal_efficiency() * 100
        hrsg      = self.hrsg_heat_transfer() / 1e6

        lines = [
            "=" * 55,
            "  HTGR Combined Cycle Performance Summary",
            "=" * 55,
            f"  Reactor thermal power    : {self.Q_reactor/1e6:.1f} MWth",
            f"  Helium flow rate         : {self.m_dot_He:.1f} kg/s",
            f"  Steam flow rate          : {self.m_dot_steam:.1f} kg/s",
            "-" * 55,
            f"  He Brayton net output    : {W_brayton:.1f} MWe",
            f"  Steam Rankine output     : {W_rankine:.1f} MWe",
            f"  Combined electrical      : {W_total:.1f} MWe",
            f"  HRSG heat duty           : {hrsg:.1f} MWth",
            f"  Cycle thermal efficiency : {eta:.1f}%",
            "-" * 55,
            f"  Brayton Carnot (950→650°C): "
            f"{self.carnot_efficiency(950,650)*100:.1f}%",
            f"  Rankine Carnot (500→70°C) : "
            f"{self.carnot_efficiency(500,70)*100:.1f}%",
            f"  Combined Carnot (950→70°C): "
            f"{self.carnot_efficiency(950,70)*100:.1f}%",
            "=" * 55,
        ]
        return "\n".join(lines)
