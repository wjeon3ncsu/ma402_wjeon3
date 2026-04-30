# MA402 Final Project — Minimum Surface Area Problem

> **Course:** MA402 · Department of Mathematics, NC State University · Spring 2026  
> **Solver:** TAO L-BFGS (LMVM) via `petsc4py`  
> **Reference C code:** [`src/tao/unconstrained/tutorials/minsurf1.c`](https://petsc.org/main/src/tao/unconstrained/tutorials/minsurf1.c.html)

---

## Problem Description

We find a function $u(x, y)$ over the 2D domain $\Omega = [-0.5, 0.5]^2$ that minimizes the surface area of the graph $z = u(x, y)$:

$$
\min_{u} \; f(u) = \iint_{\Omega} \sqrt{1 + \left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial u}{\partial y}\right)^2} \, dx \, dy
$$

subject to Dirichlet boundary conditions on all four sides, determined by a Joukowski-style complex mapping.

This is the classical **Plateau's problem** — the shape of a soap film stretched across a wire frame. The solver discretizes the domain on an $m_x \times m_y$ uniform grid and minimizes the resulting nonlinear objective using the **L-BFGS** (Limited-memory BFGS) algorithm from PETSc's TAO library.

---

## Repository Structure

```
my-petsc4py-project/
├── README.md
├── tutorial_module.py            # MinSurfSolver class — TAO L-BFGS solver
├── tutorial_presentation.ipynb  # Jupyter Notebook — visualization & analysis
└── docs/
    ├── setObjectiveGradient.md   # NumPy docstring + source mapping
    ├── setTolerances.md          # NumPy docstring + source mapping
    └── getConvergedReason.md     # NumPy docstring + source mapping
```

---

## Installation

```bash
# Install PETSc and petsc4py (pip wheels bundle PETSc automatically)
pip install petsc petsc4py

# Other dependencies
pip install numpy matplotlib
```

> **Note:** `petsc4py` requires a compatible MPI installation on some systems.  
> On Google Colab: `!pip install petsc petsc4py` is sufficient.

---

## Usage

### Run the solver directly

```bash
python tutorial_module.py
```

Expected output:

```
---- Minimum Surface Area Problem -----
mx: 32     my: 32

Converged Reason : 3
Iterations       : 61
Final Objective  : 1.421041
```

(Run on a single process. Iteration count and final objective will vary slightly with grid size — see the notebook's resolution comparison: 16×16 → ~1.4202, 32×32 → ~1.4210, 64×64 → ~1.4213.)

### Use as a module

```python
from tutorial_module import MinSurfSolver

solver = MinSurfSolver(mx=32, my=32)
solution = solver.solve()        # returns numpy.ndarray shape (my, mx)
solver.plot_solution(solution)   # 2D contour + 3D surface plots
```

### Run the Jupyter Notebook

Open `tutorial_presentation.ipynb` and run all cells (`Restart & Run All`).  
The notebook covers:
- Problem formulation with LaTeX equations
- Convergence history plot (iteration vs. objective value)
- 2D contour and 3D surface visualizations
- Grid resolution comparison (16×16, 32×32, 64×64)

---

## AI Translation Experience

### What we did

The original problem is implemented in C (`minsurf1.c`). We used an AI assistant (Claude by Anthropic) to translate the C code into a `petsc4py` Python script, then debugged and verified the result.

### Errors encountered and how we fixed them

| Error | Cause | Fix |
|---|---|---|
| `AttributeError: TAO has no attribute 'setObjectiveAndGradient'` | AI used the C function name directly | Changed to correct Python API: `tao.setObjectiveGradient()` |
| `AttributeError: 'MinSurfSolver' object has no attribute 'tao'` | `tao` object was local to `solve()` and destroyed before return | Accessed convergence info inside `solve()` before `tao.destroy()` |
| Could not capture per-iteration objective values via `tao.setMonitor` | The TAO monitor callback signature in petsc4py did not match what we wired up, and we wanted iteration-level history without fighting the monitor API | Skipped `setMonitor` entirely. In the notebook, subclassed `MinSurfSolver` as `MinSurfSolverWithHistory` and overrode `form_function_gradient` to append `(iteration, f)` to a module-level `history` list before delegating to `super().form_function_gradient(...)`. TAO calls the obj+grad callback at every evaluation, so this gives a full convergence trace. |
| Boundary condition mismatch | AI's Newton iteration for the Joukowski mapping had sign errors | Traced against the original C `MSA_BoundaryConditions()` and corrected `nf1`, `nf2` signs |

### Key takeaway

AI translation is a useful starting point but requires careful verification against the actual petsc4py API. The Cython source (`TAO.pyx`) and PETSc manual pages were essential references for debugging hallucinated API calls.

---

## Source Mapping (M2)

Three petsc4py functions were traced from Python → Cython → C:

| Python method | Cython file | C function |
|---|---|---|
| `tao.setObjectiveGradient()` | `TAO.pyx` L309 | `TaoSetObjectiveAndGradient` |
| `tao.setTolerances()` | `TAO.pyx` ~L1630 | `TaoSetTolerances` |
| `tao.getConvergedReason()` | `TAO.pyx` L1244 | `TaoGetConvergedReason` |

Full documentation with GitLab links and NumPy-style docstrings is in [`docs/`](docs/).

---

## Git Commit History

```
git log --oneline

f9061b3 (HEAD -> main) docs: add README with problem description, usage, and AI translation experience
556d4b3 docs: add M4 NumPy docstrings for setObjectiveGradient, setTolerances, getConvergedReason
054c526 docs: add M2 source mapping for setObjectiveGradient, setTolerances, getConvergedReason
806e27e feat: add tutorial_module.py
63bfe96 Initial commit
```

> **Note:** Duplicate commits (`89c097e`, `01e9293`) were squashed for clarity.  
> To clean up locally: `git rebase -i 63bfe96`
