# `tao.setTolerances()`

## Source Mapping

| Layer | Location |
|---|---|
| **Python call** | `tao.setTolerances(gatol=1e-5, grtol=1e-5, gttol=0.0)` |
| **Cython wrapper** | [`src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx`, line ~1630](https://gitlab.com/petsc/petsc/-/blob/release/src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx) (between `setMaximumIterations` L1191 and `getTolerances` L1665) |
| **C function** | `TaoSetTolerances` |
| **C header** | [`include/petsctao.h`](https://gitlab.com/petsc/petsc/-/blob/release/include/petsctao.h) |
| **C source** | [`src/tao/interface/taosolver.c`](https://gitlab.com/petsc/petsc/-/blob/release/src/tao/interface/taosolver.c) |

---

## Mathematical Description

TAO declares convergence when **any one** of the following three gradient-norm conditions is satisfied at the current iterate $X$ (with initial guess $X_0$):

$$
\begin{array}{lcll}
\|\nabla f(X)\| & \leq & \varepsilon_{\text{gatol}} & \text{(absolute gradient tolerance)} \\[4pt]
\dfrac{\|\nabla f(X)\|}{|f(X)|} & \leq & \varepsilon_{\text{grtol}} & \text{(relative gradient / objective tolerance)} \\[6pt]
\dfrac{\|\nabla f(X)\|}{\|\nabla f(X_0)\|} & \leq & \varepsilon_{\text{gttol}} & \text{(gradient reduction tolerance)}
\end{array}
$$

- **`gatol`** stops the solver once the gradient norm is small in an absolute sense — useful when the scale of $f$ is known.
- **`grtol`** stops the solver once the gradient is small relative to the current objective value — scale-invariant stopping criterion.
- **`gttol`** stops the solver once the gradient has been reduced by a factor from the initial gradient — useful to specify "reduce gradient by 5 orders of magnitude."

Setting a tolerance to `0.0` disables that particular criterion.

---

## C Function Signature

```c
#include "petsctao.h"

PetscErrorCode TaoSetTolerances(
    Tao       tao,    /* TAO solver context */
    PetscReal gatol,  /* absolute gradient norm tolerance */
    PetscReal grtol,  /* gradient norm / |f| tolerance */
    PetscReal gttol   /* gradient norm reduction factor */
)
```

### C → Python Type Mapping

| C type | Python type | Notes |
|---|---|---|
| `Tao` | `PETSc.TAO` | solver object |
| `PetscReal` | `float` | double precision floating point |
| `PETSC_DEFAULT` | `None` or omit | leaves the tolerance at its default |
| `PETSC_CURRENT` | `PETSC_CURRENT` sentinel | leaves tolerance unchanged |

---

## NumPy-Style Docstring

```python
def setTolerances(self, gatol=None, grtol=None, gttol=None):
    """
    Set convergence tolerances for the TAO gradient-norm stopping criteria.

    Wraps the C function ``TaoSetTolerances``.
    See: https://gitlab.com/petsc/petsc/-/blob/release/src/tao/interface/taosolver.c

    Logically collective.

    The solver declares convergence when **any** of the following holds:

    .. math::

        \\|\\nabla f(X)\\| \\leq \\text{gatol}

        \\|\\nabla f(X)\\| / |f(X)| \\leq \\text{grtol}

        \\|\\nabla f(X)\\| / \\|\\nabla f(X_0)\\| \\leq \\text{gttol}

    Setting a tolerance to ``0.0`` disables that criterion entirely.
    Pass ``None`` (``PETSC_DEFAULT``) to keep the solver's built-in default.

    Parameters
    ----------
    gatol : float or None, optional
        Absolute convergence tolerance on the gradient norm.
        Solver stops when ||grad f(x)|| < gatol.
        Default: 1e-8 (PETSC_DEFAULT).

    grtol : float or None, optional
        Relative convergence tolerance: gradient norm divided by |f(x)|.
        Solver stops when ||grad f(x)|| / |f(x)| < grtol.
        Default: 1e-8 (PETSC_DEFAULT).

    gttol : float or None, optional
        Gradient reduction tolerance: ratio of current gradient norm
        to the initial gradient norm at x0.
        Solver stops when ||grad f(x)|| / ||grad f(x0)|| < gttol.
        Default: 0.0 (disabled by default).

    Returns
    -------
    None

    See Also
    --------
    getTolerances, TaoSetTolerances

    Examples
    --------
    >>> tao = PETSc.TAO().create(PETSc.COMM_WORLD)
    >>> tao.setTolerances(gatol=1e-6, grtol=1e-6, gttol=0.0)
    # Only gatol and grtol active; gttol criterion is disabled.
    """
```

---

## Options Database Equivalent

The same tolerances can be set at runtime from the command line:

```bash
python tutorial_module.py -tao_gatol 1e-6 -tao_grtol 1e-6 -tao_gttol 0.0
```

---

## Usage in `tutorial_module.py`

```python
# Set convergence tolerances.
# Both gatol and grtol at 1e-5 provide a tight convergence test.
# gttol=0.0 disables the gradient-reduction criterion (not needed here).
tao.setTolerances(gatol=1e-5, grtol=1e-5, gttol=0.0)
```
