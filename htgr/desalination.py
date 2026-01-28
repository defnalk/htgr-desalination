"""
htgr/desalination.py
--------------------
Multi-Effect Distillation (MED) performance model coupled to the
HTTR cogeneration system.

Calculates:
  - Daily freshwater production from steam mass flow and GOR
  - CO₂ emissions offset vs. conventional fossil-fuel desalination
  - Economic metrics (CAPEX recovery, IRR approximation)
  - Working capacity sensitivity across GOR and availability factors

Based on DEEP (Desalination Economic Evaluation Program) outputs
from the HTGR 5 report and standard MED literature.

Reference: Imperial College HTGR 5 / HR 5 Report, §3.3–3.4, §4
"""

import numpy as np


# ── Baseline constants from group report ─────────────────────────────────────
STEAM_FLOW_KGS    = 12.5          # kg/s steam to MED unit
AVAILABILITY      = 0.95          # plant availability factor
CO2_DESAL_KG_M3  = 5.0           # kg CO₂ per m³ (conventional fossil desal)
CO2_GRID_T_MWH   = 0.74          # t CO₂ per MWh (regional grid intensity)
WATER_PRICE_M3   = 1.50          # $/m³ (report value)
CAPEX_USD        = 1.5e9         # $ (1.5 billion CAPEX)
PLANT_LIFE_YR    = 60            # years


class MEDDesalination:
    """
    Multi-Effect Distillation (MED) freshwater production model.

    Coupled to the Steam Rankine cycle exhaust (70°C, low-pressure steam).

    Parameters
    ----------
    m_dot_steam : float
        Steam mass flow rate to MED (kg/s). Default 12.5.
    GOR : float
        Gain Output Ratio (kg freshwater / kg steam). Default 11.1.
    availability : float
        Plant availability factor (0–1). Default 0.95.
    P_elec_MWe : float
        Net electrical power output for CO₂ offset calculation. Default 40.

    Examples
    --------
    >>> med = MEDDesalination(GOR=11.1)
    >>> print(f"Daily production: {med.daily_production():.0f} m³/day")
    """

    def __init__(
        self,
        m_dot_steam: float = STEAM_FLOW_KGS,
        GOR: float = 11.1,
        availability: float = AVAILABILITY,
        P_elec_MWe: float = 40.0,
    ):
        self.m_dot_steam  = m_dot_steam
        self.GOR          = GOR
        self.availability = availability
        self.P_elec_MWe   = P_elec_MWe

    def daily_production(self) -> float:
        """
        Freshwater output per day.

        Calculation:
            steam_kg_day = ṁ_steam × 86400
            theoretical_m3_day = steam_kg_day × GOR / 1000
            actual_m3_day = theoretical × availability

        Returns
        -------
        float
            Freshwater produced (m³/day).
        """
        steam_per_day = self.m_dot_steam * 86_400        # kg/day
        theoretical   = steam_per_day * self.GOR / 1000  # m³/day (density ≈ 1000 kg/m³)
        return theoretical * self.availability

    def annual_production(self) -> float:
        """Annual freshwater production (m³/year)."""
        return self.daily_production() * 365

    def population_served(self, litres_per_capita_day: float = 150.0) -> int:
        """
        Estimate population served at a given per-capita water demand.

        Parameters
        ----------
        litres_per_capita_day : float
            Daily water demand per person (L/day). Default 150 (WHO standard).

        Returns
        -------
        int
            Number of people supplied.
        """
        daily_litres = self.daily_production() * 1000  # m³ → litres
        return int(daily_litres / litres_per_capita_day)

    def co2_saved_annual(self) -> dict[str, float]:
        """
        Annual CO₂ emissions avoided vs. fossil-fuel baseline.

        Compares against:
          - Conventional gas-fired desalination: 5 kg CO₂/m³
          - Regional grid electricity: 0.74 t CO₂/MWh

        Returns
        -------
        dict with keys:
            desal_t_yr  : CO₂ avoided from water production (t/yr)
            elec_t_yr   : CO₂ avoided from clean electricity (t/yr)
            total_t_yr  : Total annual CO₂ avoidance (t/yr)
        """
        V_annual = self.annual_production()                                    # m³/yr
        E_desal  = V_annual * CO2_DESAL_KG_M3 / 1000                          # t CO₂/yr

        MWh_annual = self.P_elec_MWe * self.availability * 8760              # MWh/yr
        E_elec     = MWh_annual * CO2_GRID_T_MWH                              # t CO₂/yr

        return {
            "desal_t_yr":  round(E_desal),
            "elec_t_yr":   round(E_elec),
            "total_t_yr":  round(E_desal + E_elec),
        }

    def co2_saved_lifecycle(self, years: int = 30) -> float:
        """Total CO₂ avoided over the plant lifetime (t)."""
        return self.co2_saved_annual()["total_t_yr"] * years

    def production_surface(
        self,
        GOR_range: np.ndarray | None = None,
        avail_range: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute freshwater production over a GOR × availability grid.

        Used to generate the 3D sensitivity surface shown in the report.

        Parameters
        ----------
        GOR_range : array or None
            GOR values to sweep. Default np.linspace(10, 12, 30).
        avail_range : array or None
            Availability factors to sweep. Default np.linspace(0.80, 0.94, 30).

        Returns
        -------
        GOR_grid, avail_grid, production_grid : np.ndarray
            Meshgrid arrays for 3D plotting.
        """
        if GOR_range is None:
            GOR_range = np.linspace(10.0, 12.0, 30)
        if avail_range is None:
            avail_range = np.linspace(0.80, 0.94, 30)

        GOR_g, avail_g = np.meshgrid(GOR_range, avail_range)
        steam_per_day  = self.m_dot_steam * 86_400
        prod           = (steam_per_day * GOR_g / 1000) * avail_g

        return GOR_g, avail_g, prod

    def economic_summary(self) -> str:
        """Return a formatted economic and environmental performance summary."""
        V_daily  = self.daily_production()
        V_annual = self.annual_production()
        pop      = self.population_served()
        co2      = self.co2_saved_annual()
        co2_30yr = self.co2_saved_lifecycle(30) / 1e6  # million tonnes

        revenue_annual = V_annual * WATER_PRICE_M3          # $/yr
        payback_yr     = CAPEX_USD / revenue_annual

        lines = [
            "=" * 58,
            "  MED Desalination — Performance & Environmental Summary",
            "=" * 58,
            f"  Steam flow rate         : {self.m_dot_steam:.1f} kg/s",
            f"  GOR                     : {self.GOR}",
            f"  Availability factor     : {self.availability*100:.0f}%",
            "-" * 58,
            f"  Daily freshwater output : {V_daily:,.0f} m³/day",
            f"  Annual freshwater output: {V_annual/1e6:.2f} million m³/yr",
            f"  Population served       : ~{pop:,} people (@ 150 L/cap/day)",
            "-" * 58,
            f"  CO₂ avoided (desal)     : {co2['desal_t_yr']:,} t/yr",
            f"  CO₂ avoided (electricity): {co2['elec_t_yr']:,} t/yr",
            f"  Total CO₂ avoided       : {co2['total_t_yr']:,} t/yr",
            f"  30-year lifecycle saving : {co2_30yr:.1f} million t CO₂",
            "-" * 58,
            f"  Water price             : ${WATER_PRICE_M3}/m³",
            f"  Annual revenue          : ${revenue_annual/1e6:.1f}M/yr",
            f"  CAPEX                   : ${CAPEX_USD/1e9:.1f}B",
            f"  Simple payback          : {payback_yr:.1f} years",
            "=" * 58,
        ]
        return "\n".join(lines)
