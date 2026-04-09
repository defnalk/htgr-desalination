"""
htgr.cli — config-driven entry point for reproducible HTGR-desal runs.

Usage:
    python -m htgr.cli --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import interp1d

from htgr import PointKinetics, ThermalCore, BraytonRankineCycle, MEDDesalination


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(config_path: Path) -> dict:
    cfg = load_config(config_path)
    seed = cfg["output"].get("random_seed", 0)
    random.seed(seed)
    np.random.seed(seed)

    rcfg = cfg["reactor"]
    pk = PointKinetics(Q_nominal=float(rcfg["Q_nominal_W"]))
    rho = float(rcfg["reactivity_pcm"]) * 1e-5
    p_start = float(rcfg["pulse_start_s"])
    p_end = float(rcfg["pulse_end_s"])
    t_end = float(rcfg["t_end_s"])

    def rho_pulse(t):
        return rho if p_start <= t <= p_end else 0.0

    t_kin, P, Q = pk.simulate(t_end=t_end, rho_func=rho_pulse)

    tc = ThermalCore()
    Q_interp = interp1d(t_kin, Q, kind="linear", fill_value="extrapolate")
    t_th, Tf, Tm, Tc_node = tc.simulate(Q_profile=Q_interp, t_end=t_end)

    cyc = BraytonRankineCycle()
    med = MEDDesalination(GOR=float(cfg["desalination"]["GOR"]))

    metrics = {
        "config": str(config_path),
        "reactor": {
            "P_peak_norm": float(P.max()),
            "P_peak_t_s": float(t_kin[P.argmax()]),
            "Q_peak_MW": float(Q.max() / 1e6),
        },
        "thermal": {
            "Tf_peak_K": float(Tf.max()),
            "Tm_peak_K": float(Tm.max()),
            "Tc_peak_K": float(Tc_node.max()),
        },
        "cycle_summary": cyc.summary(),
        "desal": {
            "daily_production_m3": float(med.daily_production()),
            "GOR": float(cfg["desalination"]["GOR"]),
        },
        "co2_avoided_t_yr": med.co2_saved_annual(),
    }

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / cfg["output"]["metrics_name"]
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)

    _render_figure(cfg, t_kin, P, t_th, Tf, Tm, Tc_node, med, out_dir / cfg["output"]["figure_name"])

    print(f"✓ metrics → {metrics_path}")
    print(f"✓ figure  → {out_dir / cfg['output']['figure_name']}")
    return metrics


def _render_figure(cfg, t_kin, P, t_th, Tf, Tm, Tc_node, med, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_kin, P, lw=2.5)
    ax1.axvline(cfg["reactor"]["pulse_start_s"], ls=":", color="grey")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("P / P0")
    ax1.set_title("A | Power transient")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_th, Tf, lw=2, label="Fuel")
    ax2.plot(t_th, Tm, lw=2, label="Moderator")
    ax2.plot(t_th, Tc_node, lw=2, label="Coolant")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("T (K)")
    ax2.set_title("B | Thermal core")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.text(0.05, 0.5, f"Daily production:\n  {med.daily_production():.0f} m³/day\nGOR = {cfg['desalination']['GOR']}",
             fontsize=14, va="center")
    ax3.set_axis_off()
    ax3.set_title("C | Desalination")

    ax4 = fig.add_subplot(gs[1, 1])
    co2 = med.co2_saved_annual()
    cats = ["Desal", "Elec", "Total"]
    vals = [co2["desal_t_yr"] / 1000, co2["elec_t_yr"] / 1000, co2["total_t_yr"] / 1000]
    ax4.bar(cats, vals)
    ax4.set_ylabel("kt CO₂/yr")
    ax4.set_title("D | CO₂ avoided")
    ax4.grid(alpha=0.3)

    fig.suptitle("HTGR-desalination — reproducible run", fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m htgr.cli")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
