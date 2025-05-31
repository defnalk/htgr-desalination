"""
htgr-desalination
-----------------
Python simulation of a High-Temperature Gas-Cooled Reactor (HTGR)
coupled to a hybrid Multi-Effect Distillation (MED) desalination complex.

Based on the Imperial College London HTGR 5 / HR 5 group design project (2026).
Reactor architecture: scaled HTTR, 300 MWth, helium-cooled, 950 °C outlet.

Modules
-------
kinetics     : Point kinetics ODE system (6 delayed-neutron precursor groups)
thermal_core : 3-node lumped thermal model (fuel / moderator / coolant)
power_cycles : Helium Brayton + Steam Rankine cycle analysis
desalination : MED performance metrics and water production calculator
heat_rejection: Parasitic load and air-cooler module sizing
"""

from .desalination import MEDDesalination
from .kinetics import PointKinetics
from .power_cycles import BraytonRankineCycle
from .thermal_core import ThermalCore

__all__ = [
    "PointKinetics",
    "ThermalCore",
    "BraytonRankineCycle",
    "MEDDesalination",
]
