# `tao.getConvergedReason()`

## Source Mapping

| Layer | Location |
|---|---|
| **Python call** | `reason = tao.getConvergedReason()` |
| **Cython wrapper** | [`src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx`, line 1244](https://gitlab.com/petsc/petsc/-/blob/release/src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx#L1244) |
| **C function** | `TaoGetConvergedReason` |
| **C header** | [`include/petsctao.h`](https://gitlab.com/petsc/petsc/-/blob/release/include/petsctao.h) |
| **C source** | [`src/tao/interface/taosolver.c`](https://gitlab.com/petsc/petsc/-/blob/release/src/tao/interface/taosolver.c) |

---

## Mathematical / Algorithmic Description

After `tao.solve()` returns, `getConvergedReason()` reports **why** the TAO optimizer terminated. The return value is a `TaoConvergedReason` enum cast to a Python `int`.

**Sign convention** (same pattern as PETSc's KSP and SNES):

| Sign | Meaning |
|---|---|
| **positive** | Converged — a solution satisfying at least one stopping criterion was found |
| **zero** | Still iterating (should not appear after `solve()` returns) |
| **negative** | Diverged — the solver failed to find a solution |

---

## `TaoConvergedReason` Enum Table

```c
typedef enum {
    /* --- converged (positive) --- */
    TAO_CONVERGED_GATOL    =  3,  /* ||grad f(x)||        < gatol          */
    TAO_CONVERGED_GRTOL    =  4,  /* ||grad f(x)|| / f(x) < grtol         */
    TAO_CONVERGED_GTTOL    =  5,  /* ||grad f(x)|| / ||grad f(x0)|| < gttol */
    TAO_CONVERGED_STEPTOL  =  6,  /* step size became smaller than steptol  */
    TAO_CONVERGED_MINF     =  7,  /* f(x) < fmin (minimum function value)   */
    TAO_CONVERGED_USER     =  8,  /* user-defined convergence               */

    /* --- diverged (negative) --- */
    TAO_DIVERGED_MAXITS      = -2, /* maximum iterations reached            */
    TAO_DIVERGED_NAN         = -4, /* function/gradient returned NaN        */
    TAO_DIVERGED_MAXFCN      = -5, /* maximum function evaluations reached  */
    TAO_DIVERGED_LS_FAILURE  = -6, /* line search failed                    */
    TAO_DIVERGED_TR_REDUCTION= -7, /* trust region radius reduced to zero   */
    TAO_DIVERGED_USER        = -8, /* user signaled divergence              */

    /* --- still iterating --- */
    TAO_CONTINUE_ITERATING =  0
} TaoConvergedReason;
```

### Most Common Outcomes for L-BFGS (LMVM)

| Value | Name | When it happens |
|---|---|---|
| `3` | `TAO_CONVERGED_GATOL` | Gradient norm below absolute threshold — **typical success** |
| `4` | `TAO_CONVERGED_GRTOL` | Gradient small relative to objective — alternative success |
| `-2` | `TAO_DIVERGED_MAXITS` | Hit max iterations without converging — increase `max_it` |
| `-6` | `TAO_DIVERGED_LS_FAILURE` | Line search failed — often caused by incorrect gradient |

---

## C Function Signature

```c
#include "petsctao.h"

PetscErrorCode TaoGetConvergedReason(
    Tao                 tao,     /* TAO solver context (in) */
    TaoConvergedReason *reason   /* termination code (out)  */
)
```

### C → Python Type Mapping

| C type | Python type | Notes |
|---|---|---|
| `Tao` | `PETSc.TAO` | solver object |
| `TaoConvergedReason` (enum) | `int` | positive = converged, negative = diverged |

---

## NumPy-Style Docstring

```python
def getConvergedReason(self):
    """
    Return the reason the TAO solver terminated.

    Wraps the C function ``TaoGetConvergedReason``.
    See: https://gitlab.com/petsc/petsc/-/blob/release/src/tao/interface/taosolver.c

    Not collective. Must be called after ``tao.solve()``.

    The ``TaoConvergedReason`` enum is returned as a Python ``int``.
    The sign of the value encodes the outcome:

    - **Positive** → the solver converged successfully.
    - **Zero**     → still iterating (should not occur after ``solve()``).
    - **Negative** → the solver diverged or failed.

    Returns
    -------
    reason : int
        An integer code from the ``TaoConvergedReason`` enum.

        Converged:
            3 (GATOL)    ||grad f(x)||        < gatol
            4 (GRTOL)    ||grad f(x)|| / f(x) < grtol
            5 (GTTOL)    ||grad f(x)|| / ||grad f(x0)|| < gttol
            6 (STEPTOL)  step size dropped below steptol
            7 (MINF)     f(x) dropped below fmin

        Diverged:
            -2 (MAXITS)       maximum iterations exceeded
            -4 (NAN)          objective or gradient returned NaN
            -5 (MAXFCN)       maximum function evaluations exceeded
            -6 (LS_FAILURE)   line search failed to find a valid step
            -7 (TR_REDUCTION) trust region collapsed to zero radius

    See Also
    --------
    setTolerances, TaoGetConvergedReason, TaoConvergedReason

    Examples
    --------
    >>> tao = PETSc.TAO().create(PETSc.COMM_WORLD)
    >>> # ... configure and solve ...
    >>> tao.solve(x)
    >>> reason = tao.getConvergedReason()
    >>> if reason > 0:
    ...     print(f"Converged with reason code {reason}")
    ... else:
    ...     print(f"Diverged with reason code {reason}")
    """
```

---

## Usage in `tutorial_module.py`

```python
# After tao.solve(), check and print the convergence status.
# A positive value confirms that one of the gradient tolerances was met.
reason = tao.getConvergedReason()
print(f"TAO converged reason: {reason}")
if reason < 0:
    print("WARNING: TAO solver diverged. Check gradient implementation.")
```
