# Upwind scheme for 1D linear advection (CFL, dissipation, error, convergence)

This project implements an **upwind finite difference scheme** to solve the 1D linear transport equation:

\[
\partial_t u + a\,\partial_x u = 0
\]

It includes:
- **CFL stability study** (stable for \(|a|\Delta t/\Delta x \le 1\), blow-up otherwise)
- **Comparison with the exact solution** \(u(x,t)=u_0(x-a t)\) (periodic domain)
- **Numerical dissipation analysis** (smoothing of discontinuities, amplitude decay on a sine wave)
- **Error visualization** and norms \((L^1, L^2, L^\infty)\)
- **Convergence study** on smooth data (order 1)

---

## Mathematical background

### Exact solution (periodic)
For an initial condition \(u_0(x)\), the exact solution is:
\[
u(x,t) = u_0((x-a t)\bmod 1).
\]

### Upwind scheme
Let \(x_j=j\Delta x\), \(t^n=n\Delta t\), and \(u_j^n\approx u(x_j,t^n)\).
For \(a>0\):
\[
u_j^{n+1} = u_j^n - \nu (u_j^n - u_{j-1}^n),\quad \nu=\frac{a\Delta t}{\Delta x}.
\]
For \(a<0\):
\[
u_j^{n+1} = u_j^n - \nu (u_{j+1}^n - u_j^n).
\]

### CFL stability
The scheme is stable if:
\[
\boxed{\frac{|a|\Delta t}{\Delta x}\le 1.}
\]

### Numerical dissipation (intuition)
Upwind is stable but **diffusive**: it smooths gradients and discontinuities.
This can be interpreted as adding a numerical viscosity term (modified equation viewpoint).

---

## Project structure (suggested)
