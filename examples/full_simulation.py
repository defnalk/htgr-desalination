"""
examples/full_simulation.py
---------------------------
Full HTGR-desalination simulation:
  1. Reactor kinetics transient (reactivity step at t=100s)
  2. Thermal core response (coupled to power profile)
  3. Combined cycle performance summary
  4. Desalination production and CO₂ analysis
  5. 4-panel visualisation

Run from the project root:
    python examples/full_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

from htgr import PointKinetics, ThermalCore, BraytonRankineCycle, MEDDesalination

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

BLUE   = "#1d3557"
RED    = "#e63946"
GREEN  = "#2d6a4f"
ORANGE = "#f4a261"


# ════════════════════════════════════════════════════════════════════════════
# 1. POINT KINETICS — reactivity step transient
# ════════════════════════════════════════════════════════════════════════════
print("Running point kinetics simulation...")

pk = PointKinetics(Q_nominal=300e6)
t_end   = 400.0
rho_ins = 0.003   # +300 pcm pulse

def rho_pulse(t):
    """Reactivity pulse: step insertion then return to zero."""
    if 100 <= t <= 150:
        return rho_ins
    return 0.0

t_kin, P, Q = pk.simulate(t_end=t_end, rho_func=rho_pulse)

print(f"  Peak normalised power: {P.max():.4f}  (at t={t_kin[P.argmax()]:.1f} s)")
print(f"  Final power:          {P[-1]:.4f}")
print(f"  Peak thermal power:   {Q.max()/1e6:.1f} MWth")


# ════════════════════════════════════════════════════════════════════════════
# 2. THERMAL CORE — temperature response
# ════════════════════════════════════════════════════════════════════════════
print("\nRunning thermal core simulation...")

tc = ThermalCore()
Q_interp = interp1d(t_kin, Q, kind="linear", fill_value="extrapolate")

t_th, Tf, Tm, Tc_node = tc.simulate(Q_profile=Q_interp, t_end=t_end)

print(f"  Fuel temperature:      {Tf[-1]:.1f} K  (peak {Tf.max():.1f} K)")
print(f"  Moderator temperature: {Tm[-1]:.1f} K  (peak {Tm.max():.1f} K)")
print(f"  Coolant temperature:   {Tc_node[-1]:.1f} K  (peak {Tc_node.max():.1f} K)")


# ════════════════════════════════════════════════════════════════════════════
# 3. POWER CYCLES
# ════════════════════════════════════════════════════════════════════════════
print("\nCalculating power cycle performance...")

cyc = BraytonRankineCycle()
print(cyc.summary())


# ════════════════════════════════════════════════════════════════════════════
# 4. DESALINATION
# ════════════════════════════════════════════════════════════════════════════
print("\nCalculating desalination performance...")

med = MEDDesalination(GOR=11.1)
print(med.economic_summary())

GOR_g, avail_g, prod_g = med.production_surface()


# ════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

# ── Panel A: Reactor power transient ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t_kin, P, color=RED, lw=2.5, label="Normalised power P(t)")
ax1.axvline(100, ls=":", color="grey", lw=1.5, label="Reactivity step (+300 pcm)")
ax1.axhline(1.0, ls="--", color="black", lw=1, alpha=0.5)
ax1.fill_between(t_kin, 1.0, P, where=(P > 1.0), alpha=0.12, color=RED)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Normalised power P(t)")
ax1.set_title("A  |  Point Kinetics — Power Transient", fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_ylim(0.95, None)

# ── Panel B: Thermal core temperatures ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_th, Tf,       color=RED,    lw=2, label="Fuel (Tf)")
ax2.plot(t_th, Tm,       color=ORANGE, lw=2, label="Moderator (Tm)")
ax2.plot(t_th, Tc_node,  color=BLUE,   lw=2, label="Coolant (Tc)")
ax2.axvline(100, ls=":", color="grey", lw=1.5)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Temperature (K)")
ax2.set_title("B  |  Thermal Core — Temperature Response", fontweight="bold")
ax2.legend(fontsize=8)

# ── Panel C: 3D production surface ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0], projection="3d")
surf = ax3.plot_surface(GOR_g, avail_g, prod_g, cmap="viridis", alpha=0.85, edgecolor="none")
ax3.scatter([11.1], [0.95], [med.daily_production()],
            color=RED, s=80, zorder=10, label=f"Target: {med.daily_production():.0f} m³/day")
ax3.set_xlabel("GOR", labelpad=8)
ax3.set_ylabel("Availability", labelpad=8)
ax3.set_zlabel("m³/day", labelpad=8)
ax3.set_title("C  |  Freshwater Production Surface", fontweight="bold", pad=12)
ax3.legend(fontsize=8)
fig.colorbar(surf, ax=ax3, shrink=0.5, label="m³/day")

# ── Panel D: CO₂ savings bar chart ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
co2 = med.co2_saved_annual()
categories = ["Desalination\noffset", "Electricity\noffset", "Total\navoided"]
values     = [co2["desal_t_yr"], co2["elec_t_yr"], co2["total_t_yr"]]
colours    = [BLUE, ORANGE, GREEN]
bars = ax4.bar(categories, [v / 1000 for v in values], color=colours, width=0.5, alpha=0.85)
ax4.bar_label(bars, labels=[f"{v/1000:.0f}k t" for v in values], padding=4, fontsize=9)
ax4.set_ylabel("CO₂ avoided (thousand tonnes/yr)")
ax4.set_title("D  |  Annual CO₂ Emissions Offset", fontweight="bold")
ax4.set_ylim(0, max(values) / 1000 * 1.25)

fig.suptitle(
    "HTGR-Desalination Complex — Python Simulation\n"
    "Scaled HTTR · 300 MWth · 12,000 m³/day · Zero-emission",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.savefig("examples/htgr_simulation_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✓ Figure saved → examples/htgr_simulation_results.png")
