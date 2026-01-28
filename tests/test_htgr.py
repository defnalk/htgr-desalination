"""
tests/test_htgr.py
Unit tests for the htgr-desalination package.
Run: python -m pytest tests/ -v
"""
import numpy as np
import pytest
from htgr import PointKinetics, ThermalCore, BraytonRankineCycle, MEDDesalination


class TestPointKinetics:
    def test_steady_state_power(self):
        """At zero reactivity, power should stay at P=1."""
        pk = PointKinetics()
        t, P, Q = pk.simulate(t_end=100, rho_step=0.0, step_time=50)
        assert abs(P[-1] - 1.0) < 0.01

    def test_positive_step_increases_power(self):
        pk = PointKinetics()
        _, P, _ = pk.simulate(t_end=400, rho_step=0.003, step_time=50)
        assert P.max() > 1.0

    def test_negative_step_decreases_power(self):
        pk = PointKinetics()
        _, P, _ = pk.simulate(t_end=400, rho_step=-0.003, step_time=50)
        assert P.min() < 1.0

    def test_nominal_power_output(self):
        pk = PointKinetics(Q_nominal=300e6)
        _, _, Q = pk.simulate(t_end=50, rho_step=0.0)
        assert abs(Q[0] - 300e6) < 1e4

    def test_beta_total(self):
        pk = PointKinetics()
        assert abs(pk.beta - 0.0075) < 1e-6

    def test_precursor_initial_conditions(self):
        pk = PointKinetics()
        C0 = pk._steady_state_precursors()
        assert len(C0) == 6
        assert all(c > 0 for c in C0)


class TestThermalCore:
    def test_steady_state_temperatures_reasonable(self):
        tc = ThermalCore()
        Tf, Tm, Tc = tc._steady_state()
        assert Tf > 300 and Tm > 300 and Tc > 300   # Above ambient
        assert Tf < 5000                              # Below melting point

    def test_coolant_outlet_increases_with_power(self):
        tc = ThermalCore()
        T1 = tc.helium_outlet_temperature(300e6)
        T2 = tc.helium_outlet_temperature(150e6)
        assert T1 > T2

    def test_temperature_feedback_is_negative(self):
        tc = ThermalCore()
        p = tc.p
        rho = tc.temperature_feedback(p["Tf_0"] + 100, p["Tm_0"] + 50)
        assert rho < 0   # Negative feedback → safe

    def test_simulate_returns_correct_shape(self):
        tc = ThermalCore()
        t_eval = np.linspace(0, 100, 200)
        t, Tf, Tm, Tc = tc.simulate(Q_profile=lambda t: 300e6, t_eval=t_eval)
        assert len(t) == 200


class TestBraytonRankineCycle:
    def test_turbine_work_positive(self):
        cyc = BraytonRankineCycle()
        assert cyc.he_turbine_work() > 0
        assert cyc.steam_turbine_work() > 0

    def test_combined_output_positive(self):
        """Combined Brayton + Rankine output should be positive."""
        cyc = BraytonRankineCycle()
        W_combined = cyc.net_brayton_work() + cyc.steam_turbine_work()
        assert W_combined > 0

    def test_efficiency_in_range(self):
        cyc = BraytonRankineCycle()
        W_combined = cyc.net_brayton_work() + cyc.steam_turbine_work()
        eta = W_combined / cyc.Q_reactor
        assert 0.0 < eta < 0.5

    def test_carnot_water_at_100C(self):
        cyc = BraytonRankineCycle()
        eta = cyc.carnot_efficiency(100, 0)
        assert abs(eta - (1 - 273.15 / 373.15)) < 1e-4

    def test_euf_greater_than_efficiency(self):
        cyc = BraytonRankineCycle()
        eta = cyc.thermal_efficiency()
        euf = cyc.energy_utilisation_factor(Q_desal=180e6)
        assert euf > eta


class TestMEDDesalination:
    def test_daily_production_target(self):
        """Should produce ~11,400 m³/day as per report."""
        med = MEDDesalination(GOR=11.1)
        assert 10_000 < med.daily_production() < 13_000

    def test_population_served(self):
        med = MEDDesalination(GOR=11.1)
        pop = med.population_served()
        assert 50_000 < pop < 120_000

    def test_co2_savings_positive(self):
        med = MEDDesalination()
        co2 = med.co2_saved_annual()
        assert co2["total_t_yr"] > 0

    def test_annual_greater_than_daily(self):
        med = MEDDesalination()
        assert med.annual_production() > med.daily_production()

    def test_production_surface_shape(self):
        med = MEDDesalination()
        GOR_g, avail_g, prod_g = med.production_surface()
        assert GOR_g.shape == avail_g.shape == prod_g.shape

    def test_higher_gor_more_water(self):
        m1 = MEDDesalination(GOR=10.0).daily_production()
        m2 = MEDDesalination(GOR=12.0).daily_production()
        assert m2 > m1
