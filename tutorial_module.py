"""
Minimum Surface Area Problem

This module solves the Minimum Surface Area problem using PETSc's TAO
optimization solver via petsc4py.

Given a 2D rectangular domain with boundary conditions along the edges,
the objective is to find the surface u(x, y) with minimal area:

    min  ∬_Ω √(1 + (∂u/∂x)² + (∂u/∂y)²) dx dy

Domain: Ω = [-0.5, 0.5] × [-0.5, 0.5]
Boundary conditions are determined by a complex mapping (Joukowski-style).
Solved using the L-BFGS (LMVM) algorithm from PETSc TAO.

Reference C code:
    https://petsc.org/main/src/tao/unconstrained/tutorials/minsurf1.c.html
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


class MinSurfSolver:
    """
    Solves the Minimum Surface Area problem using TAO L-BFGS (LMVM).

    The problem seeks u(x, y) over a 2D grid that minimizes the surface area
    integral subject to boundary conditions determined by a complex mapping.

    Parameters
    ----------
    mx : int
        Number of interior grid points in the x-direction.
    my : int
        Number of interior grid points in the y-direction.
    """

    def __init__(self, mx=32, my=32):
        self.mx = mx
        self.my = my
        self.bottom = None
        self.top = None
        self.left = None
        self.right = None

        # Compute boundary conditions (mirrors MSA_BoundaryConditions in C)
        self._setup_boundary_conditions()

    # ------------------------------------------------------------------
    # Boundary Conditions
    # ------------------------------------------------------------------

    def _setup_boundary_conditions(self):
        """
        Compute boundary values using the Joukowski-style complex mapping.

        The C code iterates a Newton method on:
            nf1 = u1 + u1*u2^2 - u1^3/3 - xt = 0
            nf2 = -u2 - u1^2*u2 + u2^3/3 - yt = 0
        and sets boundary[i] = u1^2 - u2^2.

        This mirrors MSA_BoundaryConditions() in minsurf1.c exactly.
        """
        mx, my = self.mx, self.my
        b, t, l, r = -0.5, 0.5, -0.5, 0.5
        tol = 1e-10
        maxits = 5

        hx = (r - l) / (mx + 1)
        hy = (t - b) / (my + 1)

        def _boundary_side(xstart, ystart, n, move_x):
            vals = np.zeros(n)
            xt, yt = xstart, ystart
            for i in range(n):
                u1, u2 = xt, -yt
                for _ in range(maxits):
                    nf1 = u1 + u1 * u2**2 - u1**3 / 3.0 - xt
                    nf2 = -u2 - u1**2 * u2 + u2**3 / 3.0 - yt
                    fnorm = np.sqrt(nf1**2 + nf2**2)
                    if fnorm <= tol:
                        break
                    njac11 = 1.0 + u2**2 - u1**2
                    njac12 = 2.0 * u1 * u2
                    njac21 = -2.0 * u1 * u2
                    njac22 = -1.0 - u1**2 + u2**2
                    det = njac11 * njac22 - njac21 * njac12
                    u1 -= (njac22 * nf1 - njac12 * nf2) / det
                    u2 -= (njac11 * nf2 - njac21 * nf1) / det
                vals[i] = u1**2 - u2**2
                if move_x:
                    xt += hx
                else:
                    yt += hy
            return vals

        # bottom: y=b, x from l, size mx+2
        self.bottom = _boundary_side(l, b, mx + 2, move_x=True)
        # top:    y=t, x from l, size mx+2
        self.top    = _boundary_side(l, t, mx + 2, move_x=True)
        # left:   x=l, y from b, size my+2
        self.left   = _boundary_side(l, b, my + 2, move_x=False)
        # right:  x=r, y from b, size my+2
        self.right  = _boundary_side(r, b, my + 2, move_x=False)

    # ------------------------------------------------------------------
    # Initial Point
    # ------------------------------------------------------------------

    def _initial_point(self):
        """
        Compute the initial guess as an average of the four boundary sides.

        Mirrors MSA_InitialPoint() in minsurf1.c (the non-zero-start branch).

        Returns
        -------
        numpy.ndarray, shape (mx*my,)
            Flattened initial solution vector.
        """
        mx, my = self.mx, self.my
        x = np.zeros(mx * my)
        for j in range(my):
            for i in range(mx):
                row = j * mx + i
                x[row] = (
                    ((j + 1) * self.bottom[i + 1] + (my - j + 1) * self.top[i + 1]) / (my + 2)
                    + ((i + 1) * self.left[j + 1] + (mx - i + 1) * self.right[j + 1]) / (mx + 2)
                ) / 2.0
        return x

    # ------------------------------------------------------------------
    # Objective & Gradient
    # ------------------------------------------------------------------

    def form_function_gradient(self, tao, X, G):
        """
        Evaluate the surface area objective function and its gradient.

        This is the TAO callback registered via tao.setObjectiveGradient().
        Computes the discretized surface area integral and its gradient
        with respect to each interior grid point.

        Parameters
        ----------
        tao : PETSc.TAO
            The TAO solver context (passed automatically by TAO).
        X : PETSc.Vec
            Current solution vector (interior grid values, length mx*my).
        G : PETSc.Vec
            Output gradient vector to populate (same length as X).

        Returns
        -------
        float
            The objective function value (total surface area approximation).
        """
        mx, my = self.mx, self.my
        rhx = mx + 1.0
        rhy = my + 1.0
        hx = 1.0 / rhx
        hy = 1.0 / rhy
        hydhx = hy / hx
        hxdhy = hx / hy
        area = 0.5 * hx * hy

        x = X.getArray(readonly=True)
        g = G.getArray(readonly=False)
        g[:] = 0.0

        ft = 0.0

        for j in range(my):
            for i in range(mx):
                row = j * mx + i
                xc = x[row]
                xl = xr = xb = xt = xlt = xrb = xc

                if i == 0:
                    xl  = self.left[j + 1]
                    xlt = self.left[j + 2]
                else:
                    xl = x[row - 1]

                if j == 0:
                    xb  = self.bottom[i + 1]
                    xrb = self.bottom[i + 2]
                else:
                    xb = x[row - mx]

                if i + 1 == mx:
                    xr  = self.right[j + 1]
                    xrb = self.right[j]
                else:
                    xr = x[row + 1]

                if j + 1 == my:
                    xt  = self.top[i + 1]
                    xlt = self.top[i]
                else:
                    xt = x[row + mx]

                if i > 0 and j + 1 < my:
                    xlt = x[row - 1 + mx]
                if j > 0 and i + 1 < mx:
                    xrb = x[row + 1 - mx]

                d1 = (xc - xl)
                d2 = (xc - xr)
                d3 = (xc - xt)
                d4 = (xc - xb)
                d5 = (xr  - xrb)
                d6 = (xrb - xb)
                d7 = (xlt - xl)
                d8 = (xt  - xlt)

                df1dxc = d1 * hydhx
                df2dxc = d1 * hydhx + d4 * hxdhy
                df3dxc = d3 * hxdhy
                df4dxc = d2 * hydhx + d3 * hxdhy
                df5dxc = d2 * hydhx
                df6dxc = d4 * hxdhy

                d1 *= rhx; d2 *= rhx
                d3 *= rhy; d4 *= rhy
                d5 *= rhy; d6 *= rhx
                d7 *= rhy; d8 *= rhx

                f1 = np.sqrt(1.0 + d1*d1 + d7*d7)
                f2 = np.sqrt(1.0 + d1*d1 + d4*d4)
                f3 = np.sqrt(1.0 + d3*d3 + d8*d8)
                f4 = np.sqrt(1.0 + d3*d3 + d2*d2)
                f5 = np.sqrt(1.0 + d2*d2 + d5*d5)
                f6 = np.sqrt(1.0 + d4*d4 + d6*d6)

                ft += f2 + f4

                g[row] = (df1dxc/f1 + df2dxc/f2 + df3dxc/f3
                          + df4dxc/f4 + df5dxc/f5 + df6dxc/f6) / 2.0

        # Edge contributions (left, bottom, right, top)
        for j in range(my):
            d3 = (self.left[j+1] - self.left[j+2]) * rhy
            d2 = (self.left[j+1] - x[j*mx]) * rhx
            ft += np.sqrt(1.0 + d3*d3 + d2*d2)

        for i in range(mx):
            d2 = (self.bottom[i+1] - self.bottom[i+2]) * rhx
            d3 = (self.bottom[i+1] - x[i]) * rhy
            ft += np.sqrt(1.0 + d3*d3 + d2*d2)

        for j in range(my):
            d1 = (x[(j+1)*mx - 1] - self.right[j+1]) * rhx
            d4 = (self.right[j]   - self.right[j+1]) * rhy
            ft += np.sqrt(1.0 + d1*d1 + d4*d4)

        for i in range(mx):
            d1 = (x[(my-1)*mx + i] - self.top[i+1]) * rhy
            d4 = (self.top[i+1]    - self.top[i])    * rhx
            ft += np.sqrt(1.0 + d1*d1 + d4*d4)

        # Corner contributions
        d1 = (self.left[0]      - self.left[1])      * rhy
        d2 = (self.bottom[0]    - self.bottom[1])     * rhx
        ft += np.sqrt(1.0 + d1*d1 + d2*d2)

        d1 = (self.right[my+1]  - self.right[my])    * rhy
        d2 = (self.top[mx+1]    - self.top[mx])      * rhx
        ft += np.sqrt(1.0 + d1*d1 + d2*d2)

        return ft * area

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self):
        """
        Configure and run the TAO L-BFGS (LMVM) solver.

        Sets up the PETSc TAO context, registers the objective/gradient
        callback, applies tolerances, and solves the minimization problem.

        Returns
        -------
        numpy.ndarray, shape (mx, my)
            The converged solution reshaped as a 2D grid.
        """
        mx, my = self.mx, self.my
        N = mx * my

        print(f"\n---- Minimum Surface Area Problem -----")
        print(f"mx: {mx}     my: {my}\n")

        # Create solution vector and set initial point
        x_vec = PETSc.Vec().createSeq(N)
        x0 = self._initial_point()
        x_vec.setArray(x0)

        # Create gradient vector
        g_vec = PETSc.Vec().createSeq(N)

        # Create TAO solver
        tao = PETSc.TAO().create(PETSc.COMM_SELF)
        tao.setType(PETSc.TAO.Type.LMVM)

        # Register objective + gradient callback
        # Signature: setObjectiveGradient(callback, gradient_vec)
        tao.setObjectiveGradient(self.form_function_gradient, g_vec)

        # Set initial solution
        tao.setSolution(x_vec)

        # Convergence tolerances
        tao.setTolerances(gatol=1e-5, grtol=1e-5, gttol=0.0)

        # Pick up any command-line TAO options (e.g. -tao_monitor)
        tao.setFromOptions()

        # Solve
        tao.solve()

        # Report convergence
        reason = tao.getConvergedReason()
        its    = tao.getIterationNumber()
        fval   = tao.getFunctionValue()

        print(f"Converged Reason : {reason}")
        print(f"Iterations       : {its}")
        print(f"Final Objective  : {fval:.6f}")

        solution = x_vec.getArray().reshape(my, mx)

        tao.destroy()
        x_vec.destroy()
        g_vec.destroy()

        return solution

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_solution(self, solution):
        """
        Visualize the converged surface as 2D contour and 3D surface plots.

        Parameters
        ----------
        solution : numpy.ndarray, shape (my, mx)
            The solution returned by solve().
        """
        mx, my = self.mx, self.my
        x = np.linspace(-0.5, 0.5, mx)
        y = np.linspace(-0.5, 0.5, my)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(14, 5))

        # --- 2D contour plot ---
        ax1 = fig.add_subplot(1, 2, 1)
        cf = ax1.contourf(X, Y, solution, levels=30, cmap='viridis')
        fig.colorbar(cf, ax=ax1, label='u(x, y)')
        ax1.set_title(f'2D Contour Plot\n(mx={mx}, my={my})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # --- 3D surface plot ---
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(X, Y, solution, cmap='viridis', edgecolor='none', alpha=0.9)
        ax2.set_title(f'3D Surface Plot\n(mx={mx}, my={my})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u(x, y)')

        plt.tight_layout()
        plt.show()


# --- Execution Block ---
if __name__ == "__main__":
    solver = MinSurfSolver(mx=32, my=32)
    solution = solver.solve()
    solver.plot_solution(solution)