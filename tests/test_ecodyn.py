import math

import ecodyn


def test_logistic_initial_value():
    # At t=0 the population equals n0 regardless of r and K.
    assert ecodyn.logistic(0.5, 1000, 10, 0) == 10.0


def test_logistic_approaches_carrying_capacity():
    # For large t the population should be very close to K.
    assert ecodyn.logistic(0.5, 1000, 10, 1000) == 1000.0


def test_logistic_readme_example():
    assert round(ecodyn.logistic(0.5, 1000, 10, 5), 2) == 109.57


def test_logistic_rate_zero_at_bounds():
    # dN/dt = 0 when N is 0 or at carrying capacity.
    assert ecodyn.logistic_rate(0.5, 1000, 0) == 0.0
    assert ecodyn.logistic_rate(0.5, 1000, 1000) == 0.0


def test_logistic_rate_peaks_at_half_capacity():
    r, K = 0.5, 1000
    peak = ecodyn.logistic_rate(r, K, K / 2)
    assert math.isclose(peak, r * K / 4)
    # The peak at K/2 should exceed nearby points.
    assert peak > ecodyn.logistic_rate(r, K, K / 2 - 100)
    assert peak > ecodyn.logistic_rate(r, K, K / 2 + 100)


def test_lotka_volterra_equilibrium_is_fixed():
    # Starting at the non-trivial fixed point (gamma/delta, alpha/beta)
    # the populations should stay put.
    alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
    x_eq = gamma / delta
    y_eq = alpha / beta
    traj = ecodyn.lotka_volterra(alpha, beta, delta, gamma, x_eq, y_eq, 20.0)
    _, x_end, y_end = traj[-1]
    assert math.isclose(x_end, x_eq, rel_tol=1e-9)
    assert math.isclose(y_end, y_eq, rel_tol=1e-9)


def test_lotka_volterra_conserves_invariant():
    # The LV system conserves V = delta*x - gamma*ln(x) + beta*y - alpha*ln(y).
    alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
    traj = ecodyn.lotka_volterra(alpha, beta, delta, gamma, 10.0, 5.0, 15.0)

    def invariant(x, y):
        return delta * x - gamma * math.log(x) + beta * y - alpha * math.log(y)

    _, x0, y0 = traj[0]
    v0 = invariant(x0, y0)
    for _, x, y in traj:
        assert math.isclose(invariant(x, y), v0, rel_tol=1e-6, abs_tol=1e-6)


def test_lotka_volterra_trajectory_shape():
    traj = ecodyn.lotka_volterra(1.1, 0.4, 0.1, 0.4, 10.0, 5.0, 1.0, dt=0.1)
    assert traj[0] == (0.0, 10.0, 5.0)
    # Each entry is (t, prey, predator) and time advances monotonically.
    times = [t for t, _, _ in traj]
    assert times == sorted(times)
    # The fixed-step loop runs until t reaches t_end, so the final recorded
    # time is at or just past t_end (within one step).
    assert times[-1] >= 1.0
    assert times[-1] < 1.0 + 0.1 + 1e-9
