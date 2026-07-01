__version__ = "1.0.3"

def logistic(r, K, n0, t):
    """Closed-form logistic population at time t."""
    import math
    return K / (1 + (K - n0) / n0 * math.exp(-r * t))

def logistic_rate(r, K, n):
    return r * n * (1 - n / K)

def lotka_volterra(alpha, beta, delta, gamma, prey0, pred0, t_end, dt=0.01):
    """Classic predator-prey ODEs integrated with RK4.

    dx/dt = alpha*x - beta*x*y
    dy/dt = delta*x*y - gamma*y
    """
    def f(state):
        x, y = state
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]

    s = [float(prey0), float(pred0)]
    out = [(0.0, *s)]
    t = 0.0
    while t < t_end:
        k1 = f(s)
        k2 = f([a + 0.5 * dt * b for a, b in zip(s, k1)])
        k3 = f([a + 0.5 * dt * b for a, b in zip(s, k2)])
        k4 = f([a + dt * b for a, b in zip(s, k3)])
        s = [a + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
             for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]
        t += dt
        out.append((t, *s))
    return out
