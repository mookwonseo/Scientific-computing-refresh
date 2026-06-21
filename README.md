# Scientific Computing Refresh

Numerical study of a 1D parabolic PDE using SymPy (symbolic verification) and a forward Euler finite-difference method (FDM).

## Problem

Heat-type PDE on \(x \in [0, 1]\):

\[
f_t - \frac{1}{2\pi^2} f_{xx} = 0
\]

Equivalent form:

\[
f_t = \frac{1}{2\pi^2} f_{xx}
\]

**Exact solution**

\[
f(x, t) = e^{-2t}\big(\cos(2\pi x) + 2\sin(2\pi x)\big)
\]

**Boundary conditions**

\[
f(0, t) = e^{-2t}, \qquad f(1, t) = e^{-2t}
\]

## Project structure

```
.
├── numerical.py          # Main script: verify PDE, run FDM, plot results
├── requirements.txt      # Python dependencies
├── pyrightconfig.json    # Type checker paths for local packages
└── .vscode/settings.json # Recommended Python interpreter settings
```

## Setup

Use native ARM64 Python on Apple Silicon (`/usr/bin/python3`):

```bash
python3 -m pip install --user -r requirements.txt
```

For IDE import resolution on this external drive, packages can also be installed locally:

```bash
python3 -m pip install --target .python_packages -r requirements.txt
```

In Cursor/VS Code, select interpreter: `/usr/bin/python3`.

## Run

```bash
python3 numerical.py
```

The script will:

1. Symbolically verify the PDE and boundary conditions with SymPy
2. Run forward Euler FDM on a uniform spatial grid
3. Compare the numerical solution to the exact solution
4. Plot FDM vs exact at the final time

## Method

### Symbolic verification (`verify`)

Uses SymPy to check:

- \(f_t - \frac{1}{2\pi^2} f_{xx} = 0\)
- Boundary values at \(x = 0\) and \(x = 1\)

### Forward Euler FDM (`forward_euler_fdm`)

Spatial grid: \(h = 1/n\), nodes \(x_i = ih\).

Central difference for \(f_{xx}\):

\[
f_{xx}(x_i, t_n) \approx \frac{f_{i+1}^n - 2f_i^n + f_{i-1}^n}{h^2}
\]

Forward Euler update:

\[
f_i^{n+1} = f_i^n + \frac{dt}{2\pi^2 h^2}\left(f_{i+1}^n - 2f_i^n + f_{i-1}^n\right)
\]

Boundary values and the initial condition are taken from the exact solution via `sympy.lambdify`.

## Stability (CFL condition)

Let

\[
r = \frac{dt}{2\pi^2 h^2}
\]

For explicit forward Euler on \(f_t = D f_{xx}\) with \(D = \frac{1}{2\pi^2} > 0\), stability requires:

\[
r \le \frac{1}{2}
\quad\Rightarrow\quad
dt \le \pi^2 h^2 = \frac{\pi^2}{n^2}
\]

Example: for `n = 100`, use `dt` smaller than about `9.9e-4`.

## Accuracy

For a stable explicit scheme:

- Time: \(O(dt)\)
- Space: \(O(h^2)\)

Overall error is typically \(O(dt + h^2)\). A common choice is `dt ~ h^2`.

## Key functions

| Function | Purpose |
|---|---|
| `exact_f()` | Returns symbolic exact solution |
| `verify(f)` | Symbolically checks PDE and BCs |
| `forward_euler_fdm(f, n, dt, endtime)` | Runs the FDM solver |
| `exact_solution(f, x_vals, t_val)` | Evaluates exact solution at grid points |

## Notes

- `sympy.lambdify` converts symbolic expressions into fast numeric functions for the FDM loop.
- If imports fail in the editor, ensure the selected Python interpreter matches where packages were installed.
- Creating a `.venv` on an external drive may fail; use system Python or `.python_packages` instead.

## Dependencies

- `numpy` — arrays and numerics
- `matplotlib` — plotting
- `sympy` — symbolic differentiation and verification
- `jax`, `jaxlib`, `optax` — optional, listed for future extensions
