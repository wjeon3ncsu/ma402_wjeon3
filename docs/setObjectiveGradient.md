# `tao.setObjectiveGradient()`

## Source Mapping

| Layer | Location |
|---|---|
| **Python call** | `tao.setObjectiveGradient(form_function_gradient, None, user)` |
| **Cython wrapper** | [`src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx`, line 309](https://gitlab.com/petsc/petsc/-/blob/release/src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx#L309) |
| **C function** | `TaoSetObjectiveAndGradient` |
| **C header** | [`include/petsctao.h`](https://gitlab.com/petsc/petsc/-/blob/release/include/petsctao.h) |
| **C source** | [`src/tao/interface/tao.c`](https://gitlab.com/petsc/petsc/-/blob/release/src/tao/interface/tao.c) |

---

## Mathematical Description

This function registers the **combined objective + gradient callback** for the optimization problem:

$$\min_{u} f(u)$$

At each TAO iteration, the solver calls the user-provided routine to compute both:

- The scalar objective value $f(u)$
- The gradient vector $\nabla f(u)$

Using a **combined** callback (instead of separate `setObjective` and `setGradient`) is preferred for efficiency — many algorithms need both quantities at the same point $u$, so computing them together avoids redundant function evaluations.

In our minimum surface problem:

$$f(u) = \iint_\Omega \sqrt{1 + \left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial u}{\partial y}\right)^2} \, dx\, dy$$

The gradient component at interior node $(i,j)$ is computed via finite differences of the surface area integrand.

---

## C Function Signature

```c
#include "petsctao.h"

PetscErrorCode TaoSetObjectiveAndGradient(
    Tao    tao,   /* TAO solver context */
    Vec    g,     /* [optional] pre-allocated vector for the gradient output */
    PetscErrorCode (*func)(Tao, Vec, PetscReal *, Vec, void *),  /* callback */
    void  *ctx    /* [optional] user-defined application context */
)
```

The callback `func` must have the signature:

```c
PetscErrorCode FormFunctionGradient(
    Tao       tao,  /* solver context (in) */
    Vec       x,    /* current iterate (in) */
    PetscReal *f,   /* objective value (out) */
    Vec       g,    /* gradient vector (out) */
    void      *ctx  /* user context (in) */
);
```

### C → Python Type Mapping

| C type | Python type | Notes |
|---|---|---|
| `Tao` | `PETSc.TAO` | solver object |
| `Vec` | `PETSc.Vec` | PETSc vector |
| `PetscReal *` | `float` (return via tuple) | scalar objective value |
| `void *` | any Python object | user context, passed through |
| `PetscErrorCode` | raises exception on error | non-zero = error |

---

## NumPy-Style Docstring

```python
def setObjectiveGradient(self, objgrad, g=None, args=None, kargs=None):
    """
    Set the combined objective function and gradient evaluation callback.

    Wraps the C function ``TaoSetObjectiveAndGradient``.
    See: https://gitlab.com/petsc/petsc/-/blob/release/src/tao/interface/tao.c

    Logically collective.

    Using a combined callback is more efficient than setting the objective
    and gradient separately, because optimization algorithms typically
    require both f(x) and grad f(x) at the same point x per iteration.

    Parameters
    ----------
    objgrad : callable
        A Python callable with the signature::

            objgrad(tao, x, g, *args, **kargs) -> float

        where:
        - ``tao`` (PETSc.TAO)  -- the solver context
        - ``x``   (PETSc.Vec)  -- the current iterate
        - ``g``   (PETSc.Vec)  -- output gradient vector (modified in-place)
        - Return value (float) -- the scalar objective value f(x)

    g : PETSc.Vec or None, optional
        Pre-allocated vector to store the gradient.
        If None, TAO allocates one internally.

    args : tuple or None, optional
        Extra positional arguments forwarded to ``objgrad``.

    kargs : dict or None, optional
        Extra keyword arguments forwarded to ``objgrad``.

    Returns
    -------
    None

    See Also
    --------
    setObjective, setGradient, getObjectiveAndGradient,
    TaoSetObjectiveAndGradient

    Examples
    --------
    >>> from petsc4py import PETSc
    >>> def form_fg(tao, x, g):
    ...     # compute f and fill g with gradient
    ...     f = 0.0
    ...     # ... finite-difference stencil ...
    ...     return f
    >>> tao = PETSc.TAO().create(PETSc.COMM_WORLD)
    >>> tao.setType(PETSc.TAO.Type.LMVM)
    >>> tao.setObjectiveGradient(form_fg, None)
    """
```

---

## Usage in `tutorial_module.py`

```python
# Register the combined objective + gradient callback.
# g_vec is a pre-allocated PETSc Vec used to store the gradient output.
# The callback (self.form_function_gradient) is a bound method of MinSurfSolver,
# so it already has access to grid parameters (mx, my) via self.
tao.setObjectiveGradient(self.form_function_gradient, g_vec)
```
