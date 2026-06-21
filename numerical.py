import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
# PDE: f_t + 2 * pi^2 *  f_xx = 0 with boundary conditions f(0, t) = exp(2 * t) and f(1, t) = exp(2 * t)
# Exact solution: f(x, t) = exp(-2 * t) * (cos(2 * pi * x) + 2 * sin(2 * pi * x))
# Discretization: f_t = f(x, t+h) - f(x, t)
# Discretization: f_xx = f(x+h, t) - 2 * f(x, t) + f(x-h, t)
# Discretization: f_t - 1/(2 * pi^2)* f_xx = 0
# Discretization: f(x, t+h) = f(x, t) + dt/dh^2 * 1/(2 * pi^2) * (f(x+h, t) - 2 * f(x, t) + f(x-h, t))

x, t = sp.symbols('x t')

def exact_f():
    return sp.exp(-2*t) * (sp.cos(2*np.pi*x) + 2 * sp.sin(2*np.pi*x))
    

def verify(f):
    f_xx = sp.diff(f,x, 2)
    f_t = sp.diff(f, t)
    s =  "f_t - 1.0/(2 * pi^2)* f_xx"
    PDE = sp.simplify(f_t - 1.0 / (2 * np.pi**2) * f_xx)
    print(f"{s} = {PDE}")
    # Boundary conditions
    BC  = sp.simplify(f.subs(x, 0))
    print(f"Boundary conditions: f(0, t) = {BC}")
    BC  = sp.simplify(f.subs(x, 1))
    print(f"Boundary conditions: f(1, t) = {BC}")

    return

def forward_euler_fdm(f, n, dt, endtime):
    """
    Forward Euler FDM for:
        f_t - 1/(2*pi^2) * f_xx = 0  on x in [0, 1]

    Update rule:
        f(x, t+dt) = f(x, t)
            + dt/h^2 * 1/(2*pi^2) * (f(x+h, t) - 2*f(x, t) + f(x-h, t))
    """
    a, b = 0, 1
    dh = (b - a) / n
    x_grid = np.linspace(a, b, n + 1)

    f_exact = sp.lambdify((x, t), f, modules="numpy")
    coeff = dt/dh**2 * 1 / (2 * np.pi**2)
    n_steps = int(round(endtime / dt))

    # Initial condition f(x, 0)
    u = np.array([float(f_exact(xi, 0.0)) for xi in x_grid])

    for step in range(n_steps):
        t_next = (step + 1) * dt
        u_new = np.zeros(n + 1)

        # Boundary conditions from exact solution at current time
        u_new[0] = float(f_exact(a, t_next))
        u_new[n] = float(f_exact(b, t_next))

        # Interior points (update rule from line 10 in comments)
        for i in range(1, n):
            laplacian = u[i + 1] - 2 * u[i] + u[i - 1]
            u_new[i] = u[i] + coeff * laplacian

        u = u_new

    return x_grid, u


def exact_solution(f, x_vals, t_val):
    f_exact = sp.lambdify((x, t), f, modules="numpy")
    return np.array([float(f_exact(xi, t_val)) for xi in x_vals])


print("Verify the PDE")
verify(exact_f())
# stability condition: dt/dh^2 <= pi^2
n, dt, endtime = 100, 0.001, 0.1
x_grid, u_num = forward_euler_fdm(exact_f(), n, dt, endtime)
u_exact = exact_solution(exact_f(), x_grid, endtime)
err = np.max(np.abs(u_num - u_exact))
print(f"\nForward Euler FDM: n={n}, dt={dt}, t={endtime}")
print(f"Max error vs exact solution: {err:.6e}")

plt.plot(x_grid, u_num, label="FDM")
plt.plot(x_grid, u_exact, "--", label="Exact")
plt.xlabel("x")
plt.ylabel(f"f(x, t={endtime})")
plt.legend()
plt.title("Forward Euler FDM vs Exact Solution")
plt.show()






'''
def exact(x):
    return 3*x**2

def error(x, y, exact):
    return np.abs(y - exact(x))

def plot_error(x, y, exact):
    plt.plot(x, error(x, y, exact))
    plt.show()

x, y = fdm(f, 0, 1, 100)
plot_error(x, y, exact)
'''