import jax, jax.numpy as jnp
from functools import partial
from jax import Array
from typing import Optional, Callable, Any
from jax import lax


# Public wrapper: trim histories to length k after jitting
def pcg(
    A,
    b,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    wce_tol: float = 0.0,
    maxiter: int = 1_000,
    M_inv: Optional[Array] = None,
    x0: Optional[Array] = None,
    return_history: bool = True,
    f_evals: Optional[Array] = None,
) -> tuple[Any, Any, Any, Any] | tuple[Any, Any, Any, Any, Any]:
    r"""
    Solve the symmetric system :math:`Ax = b` using Preconditioned Conjugate Gradient (PCG),
    optionally tracking residuals, worst-case RKHS errors, and Stein-kernel quadrature.

    This function dispatches to a JIT-compiled core:

    - If both ``return_history=True`` and ``f_vals`` is supplied, uses
      a PCG core that also computes the quadrature estimates at each iteration.
    - Otherwise, uses the history-only core or the plain solver.

    Args:
        A:
            Linear operator or matrix representing :math:`A` (assumed symmetric positive semi-definite).
        b (Array):
            Right-hand side vector.
        rtol (float, optional):
            Relative convergence tolerance.
        atol (float, optional):
            Absolute convergence tolerance.
        wce_tol (float, optional):
            Tolerance for worst-case error early stopping. If WCE drops below this, iteration stops.
            Defaults to``0.0`` (disabled).
        maxiter (int, optional):
            Maximum number of iterations.
        M_inv (Array, optional):
            Preconditioner (approximate inverse of :math:`A`). Defaults to ``None`` (no preconditioning).
        x0 (Array, optional):
            Initial guess for ``x``, shape ``(n,)``. If ``None``, the zero vector is used.
        return_history (bool):
            If ``True``, return per-iteration histories for residual norm and WCE.
            If ``False``, only the final values are returned.
        f_vals (Array, optional):
            Function evaluations :math:`[f(x_i)]_{i=1}^n`, evaluations of shape ``(n,)`` or ``(n, d)`` for
            on-the-fly Stein kernel quadrature estimates. If multi-dimensional, each column yields its own quadrature sequence.
            Defaults to ``None`` meaning no quadrature estimates are computed. If supplied, calls `_pcg_core_with_quad`.
    Returns:
        tuple:
            A tuple whose length and contents depend on the flags:

            - **x** (``Array``):
            Final solution :math:`x_k`. Shape: ``(n,)``.
            - **k** (``int``):
            Number of iterations performed.
            - **res_hist** (:math:`(k,)`, optional):
            Residual norms :math:`\|r_i\|` at each iteration ; only if ``return_history=True``.
            - **wce_hist** (:math:`(k,)`, optional):
            Worst-case RKHS errors :math:`\sigma(x_i)` at each iteration ; only if ``return_history=True``.
            - **c_hist** (:math:`(k,)` or :math:`(k,d)`, optional):
            Quadrature estimates :math:`c_i` at each iteration ; only if ``f_vals`` is provided.

    Notes:
        The worst-case error is defined by:

        .. math::

            \sigma(\mathbf{w}_k) = \sup_{\|v\|_{\mathcal{H}(k_p)} \leq 1}
            \left| c_{N,k}(v) - \int v(\mathbf{x}) p(\mathbf{x}) \, \mathrm{d}\mathbf{x} \right|
            = \frac{ (\mathbf{w}_k^\top \mathbf{K}_p \mathbf{w}_k )^{1/2} }{ \mathbf{1}^\top \mathbf{w}_k },

        where :math:`\mathbf{w}_k` is the solution at iteration :math:`k`, and :math:`\mathbf{K}_p` is
        the Stein kernel matrix. This provides a proxy for integration error
        when the RKHS norm :math:`\|v\|_{\mathcal{H}(k_p)}` is unknown.
    """

    n = b.shape[0]

    # if f evaluations are provided compute Stein quadrature estimates
    if f_evals is not None:
        x, k, res_all, wce_all, c_all = _pcg_core_with_quad(
            A,
            b,
            f_evals,
            rtol=rtol,
            atol=atol,
            wce_tol=wce_tol,
            maxiter=maxiter,
            M_inv=M_inv,
            x0=x0,
        )
        return x, k, res_all[:k], wce_all[:k], c_all[:k]

    if return_history is True:
        x, k, res_all, wce_all = _pcg_core_history(
            A,
            b,
            rtol=rtol,
            atol=atol,
            wce_tol=wce_tol,
            maxiter=maxiter,
            M_inv=M_inv,
            x0=x0,
        )
        return x, k, res_all[:k], wce_all[:k]
    else:
        if M_inv is None:
            M_inv = jnp.eye(n)
        x, k, r, w = _pcg_core(
            A,
            b,
            M_inv,
            rtol=rtol,
            atol=atol,
            wce_tol=wce_tol,
            maxiter=maxiter,
            x0=x0,
        )
        return x, k, r, w


# PCG without history JIT-compiled core function
@partial(jax.jit, static_argnames=("rtol", "atol", "wce_tol", "maxiter"))
def _pcg_core(
    A: Array,
    b: Array,
    M_inv: Array,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    wce_tol: float = 0.0,
    maxiter: int = 1000,
    x0: Optional[Array] = None,
) -> tuple[Any, Any, Any, Any]:
    r"""
    Core implementation of the Preconditioned Conjugate Gradient (PCG) method
    for solving the linear system :math:`\mathbf{A}x = b`.

    This internal function is JIT-compiled for performance and supports optional
    left preconditioning using a symmetric positive definite matrix :math:`\mathbf{M}^{-1}`.
    It also tracks the worst-case error (WCE), which provides a computable
    upper bound on the error of kernel-based integration estimates.

    Returns the values of the residual norms and worst-case RKHS quadrature errors at each
    iteration.

    Args:
        A:
            Linear operator or matrix representing :math:`A`. Must be symmetric
            positive semi-definite and support `A @ x`.
        b (Array):
            Right-hand side vector.
        rtol (float, optional):
            Relative tolerance for convergence.
        atol (float, optional):
            Absolute tolerance for convergence.
        wce_tol (float, optional):
            Early stopping criterion based on the worst-case error. Iteration terminates
            when WCE falls below this threshold. Defaults to 0.0 (disabled).
        maxiter (int, optional):
            Maximum number of iterations.
        M_inv (Array, optional):
            Preconditioner matrix :math:`\mathbf{M}^{-1}`. Must be symmetric positive definite.
        x0 (Array, optional):
            Initial guess for the solution. If not provided, a zero vector is used.

    Returns:
        tuple:
            A 4-tuple containing:

            - **x** (Array): Final solution :math:`x_k` to the system.
            - **k** (int): Number of iterations performed.
            - **residuals** (Array): Norms of the residual at termination :math:`\|r_k\|`. Shape: (1,).
            - **wce** (Array): Worst-case error estimate at termination :math:`\sigma(w_k)`. Shape: (1,).
    """

    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - A @ x
    z = M_inv @ r
    rho = r @ z
    res0 = jnp.linalg.norm(r)
    eps = jnp.finfo(b.dtype).tiny

    def wce(w):  # √(wᵀAw) / (1ᵀw)
        return jnp.sqrt(w @ (A @ w)) / (
            jnp.sum(w) + 1e-30
        )  # protection against zero denominator

    # state = (k, x, r, z, rho, res, w, done)
    state = (0, x, r, z, rho, res0, jnp.inf, False)

    def cond(state):
        k, _, r, _, _, res, w, _ = state
        return ~((res <= atol) | (res <= rtol * res0) | (w <= wce_tol) | (k >= maxiter))

    def body(state):
        k, x, r, z, rho, res, _, _ = state

        Az = A @ z
        alpha = rho / (z @ Az + eps)
        x = x + alpha * z
        r = r - alpha * Az
        rho_new = r @ (M_inv @ r)
        beta = rho_new / (rho + eps)
        z = M_inv @ r + beta * z
        res = jnp.linalg.norm(r)
        w = wce(x)

        return (k + 1, x, r, z, rho_new, res, w, False)

    k_final, x_final, r_final, z_final, rho_final, res_final, wce_final, _ = (
        lax.while_loop(cond, body, state)
    )

    return x_final, k_final, res_final, wce_final


# PCG with history JIT-compiled core function
@partial(jax.jit, static_argnames=("rtol", "atol", "wce_tol", "maxiter"))
def _pcg_core_history(
    A,
    b,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    wce_tol: float = 0.0,
    maxiter: int = 1_000,
    M_inv: Optional[Array] = None,
    x0: Optional[Array] = None,
) -> tuple:
    r"""
    Core implementation of the Preconditioned Conjugate Gradient (PCG) method
    for solving the linear system :math:`Ax = b`.

    This internal function is JIT-compiled for performance and supports optional
    left preconditioning using a symmetric positive definite matrix :math:`M^{-1}`.
    It also tracks the worst-case error (WCE), which provides a computable
    upper bound on the error of kernel-based integration estimates.

    Args:
        A:
            Linear operator or matrix representing :math:`A`. Must be symmetric
            positive semi-definite and support ``A @ x``.
        b:
            Right-hand side vector.
        rtol:
            Relative tolerance for convergence.
        atol:
            Absolute tolerance for convergence.
        wce_tol:
            Early stopping criterion based on the worst-case error. Iteration terminates
            when WCE falls below this threshold. Defaults to ``0.0`` (disabled).
        maxiter:
            Maximum number of iterations.
        M_inv:
            Preconditioner matrix :math:`M^{-1}`. Must be symmetric positive definite.
        x0:
            Initial guess for the solution. If not provided, a zero vector is used.

    Returns:
        tuple:
            A 4-tuple containing:

            - **x** (``Array``): Final solution :math:`x_k` to the system.
            - **k** (``int``): Number of iterations performed.
            - **residuals** (``Array``): Norms of residuals :math:`\|r_k\|` at each iteration. Shape: ``(maxiter,)``.
            - **wce** (``Array``): Worst-case error estimates :math:`\sigma(w_k)` at each iteration. Shape: ``(maxiter,)``.
    """

    n = b.shape[0]
    if M_inv is None:
        M_inv = jnp.eye(n, dtype=b.dtype)
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # worst-case error estimate – stays inside the compiled graph
    def wce(w):  # √(wᵀAw) / (1ᵀw)
        return jnp.sqrt(w @ (A @ w)) / (
            jnp.sum(w) + 1e-30
        )  # protection against zero denominator

    r0 = b - A @ x0
    d0 = M_inv @ r0
    qdr0 = r0 @ (M_inv @ r0)
    res0 = jnp.linalg.norm(r0)
    eps = jnp.finfo(b.dtype).tiny  # protection against zero denominators in cg loop

    # carry: (k, done, x, r, d, qdr, res, res0)
    carry0 = (0, False, x0, r0, d0, qdr0, res0, res0)

    def body(carry, _):
        k, done, x, r, d, qdr, res, res0 = carry

        # branch when iterating (tolerances not met)
        def step(c):
            k, _, x, r, d, qdr, res, res0 = c
            den = d @ (A @ d)
            alpha = qdr / (den + eps)

            x_n = x + alpha * d
            r_n = r - alpha * (A @ d)
            qdr_n = r_n @ (M_inv @ r_n)
            beta = qdr_n / (qdr + eps)
            d_n = M_inv @ r_n + beta * d
            res_n = jnp.linalg.norm(r_n)

            k_n = k + 1
            wce_n = wce(x_n)  # worst-case error
            done_n = (
                (res_n <= atol)
                | (res_n <= rtol * res0)
                | (k_n >= maxiter)
                | (wce_n <= wce_tol)
            )

            carry_next = (k_n, done_n, x_n, r_n, d_n, qdr_n, res_n, res0)
            out_pair = (res, wce_n)  # residual before, worst-case error after
            return carry_next, out_pair

        # branch when converged (tolerances met): output dummy (will be sliced in public wrapper)
        def nop(c):
            k, *_ = c
            out_pair = (jnp.nan, jnp.nan)
            return c, out_pair

        return jax.lax.cond(done, nop, step, carry)

    carry_fin, (res_all, wce_all) = jax.lax.scan(body, carry0, xs=None, length=maxiter)

    x_final = carry_fin[2]
    k_final = carry_fin[0]  # iterations actually executed
    return x_final, k_final, res_all, wce_all


# PCG with full history including history of quadrature estimates with given f_evals - intended for Stein based solves
@partial(jax.jit, static_argnames=("rtol", "atol", "wce_tol", "maxiter"))
def _pcg_core_with_quad(
    A: Array,
    b: Array,
    f_evals: Array,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    wce_tol: float = 0.0,
    maxiter: int = 1_000,
    M_inv: Optional[Array] = None,
    x0: Optional[Array] = None,
) -> tuple[Array, int, Array, Array, Array]:
    r"""
    JIT-compiled Preconditioned Conjugate Gradient (PCG) solver with on-the-fly Stein-kernel quadrature.

    Solves the linear system :math:`\mathbf{K}_p\,x = \mathbf{1}` with optional left preconditioning :math:`\mathbf{M}^{-1}`.
    At each iteration :math:`k`, where :math:`x_k\approx \mathbf{K}_p^{-1}\mathbf{1}` is the current PCG solution, it records:

    - **Residual norm:**
        :math:`\|r_k\|`, where :math:`r_k = \mathbf{1} - \mathbf{K}_p\,x_k`.

    - **Worst-case RKHS error:**
        :math:`\sigma(x_k)=\frac{\sqrt{x_k^\top\,\mathbf{K}_p\,x_k}}{\sum_i x_{k,i}}`.

    - **Quadrature estimate:**
        :math:`c_k = \frac{\mathbf{f}^\top\,x_k}{\mathbf{1}^\top x}`, which yields a sequence :math:`\{c_k\}` approximating :math:`\mathbb{E}_p[f] = \int f(x)\,p(x)\,\mathrm{d}x`.

    Args:
        A:
            Linear operator or matrix representing :math:`A`. Must be symmetric
            positive semi-definite and support ``A @ x``.
        b:
            Right-hand side vector of shape ``(n, )``.
        f_vals:
            Function evaluations :math:`[f(x_i)]_{i=1}^n`, evaluations of shape ``(n,)`` or ``(n, d)``.
            If multi-dimensional, each column yields its own quadrature sequence.
        rtol:
            Relative tolerance for convergence.
        atol:
            Absolute tolerance for convergence.
        wce_tol:
            Early stopping criterion based on the worst-case error. Iteration terminates
            when WCE falls below this threshold. Defaults to ``0.0`` (disabled).
        maxiter:
            Maximum number of iterations.
        M_inv:
            Preconditioner matrix :math:`\mathbf{M}^{-1}`. Must be symmetric positive definite. If not provided, the identity matrix (no preconditioner) is used.
        x0:
            Initial guess for the solution. If not provided, a zero vector is used.

    Returns:
        tuple:
            A 5-tuple containing:

            - **x** (``Array``): Final solution :math:`x_k` to the system.
            - **k** (``int``): Number of iterations performed.
            - **residuals** (``Array``): Norms of residuals :math:`\|r_k\|` at each iteration. Shape: ``(maxiter,)``.
            - **wce** (``Array``): Worst-case error estimates :math:`\sigma(w_k)` at each iteration. Shape: ``(maxiter,)``.
            - **c** (``Array``): Quadrature error estimates :math:`c_{k}` at each iteration. Shape: ``(maxiter, d)``.
    """

    n = b.shape[0]
    if M_inv is None:
        M_inv = jnp.eye(n, dtype=b.dtype)
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # worst-case error estimate – stays inside the compiled graph
    def wce(w):  # √(wᵀAw) / (1ᵀw)
        return jnp.sqrt(w @ (A @ w)) / (
            jnp.sum(w) + 1e-30
        )  # protection against zero denominator

    r0 = b - A @ x0
    d0 = M_inv @ r0
    qdr0 = r0 @ (M_inv @ r0)
    res0 = jnp.linalg.norm(r0)
    eps = jnp.finfo(b.dtype).tiny  # protection against zero denominators in cg loop

    # carry: (k, done, x, r, d, qdr, res, res0)
    carry0 = (0, False, x0, r0, d0, qdr0, res0, res0)

    def body(carry, _):
        k, done, x, r, d, qdr, res, res0 = carry

        # branch when iterating (tolerances not met)
        def step(c):
            k, _, x, r, d, qdr, res, res0 = c
            den = d @ (A @ d)
            alpha = qdr / (den + eps)

            x_n = x + alpha * d
            r_n = r - alpha * (A @ d)
            qdr_n = r_n @ (M_inv @ r_n)
            beta = qdr_n / (qdr + eps)
            d_n = M_inv @ r_n + beta * d
            res_n = jnp.linalg.norm(r_n)

            k_n = k + 1
            wce_n = wce(x_n)  # worst-case error
            c_n = x_n @ f_evals / jnp.sum(x_n + 1e-30)

            done_n = (
                (res_n <= atol)
                | (res_n <= rtol * res0)
                | (k_n >= maxiter)
                | (wce_n <= wce_tol)
            )

            carry_next = (k_n, done_n, x_n, r_n, d_n, qdr_n, res_n, res0)
            out = (res, wce_n, c_n)  # residual before, worst-case error after
            return carry_next, out

        # branch when converged (tolerances met): output dummy (will be sliced in public wrapper)
        def nop(c):
            d = f_evals.shape[1]
            dummy = jnp.full((d,), jnp.nan)
            out = (jnp.nan, jnp.nan, dummy)
            return c, out

        return jax.lax.cond(done, nop, step, carry)

    carry_fin, (res_all, wce_all, c_all) = jax.lax.scan(
        body, carry0, xs=None, length=maxiter
    )

    x_final = carry_fin[2]
    k_final = carry_fin[0]  # iterations actually executed
    return x_final, k_final, res_all, wce_all, c_all


def pcg_batch(
    A: Array,
    b: Array,
    M_inv_batch: Array,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    wce_tol: float = 0.0,
    maxiter: int = 1000,
    x0: Optional[Array] = None,
) -> tuple[Array, Array, Array, Array]:
    r"""
    Batched Preconditioned Conjugate Gradient (PCG) solver over a batch of preconditioners.

    Wraps the core :func:`_pcg_core` function to solve
    :math:`A x = b` for multiple preconditioners :math:`M^{-1}` in parallel.
    JAX's ``vmap`` is used to vectorise over the first axis of `M_inv_batch`.

    Args:
        A:
            Linear operator or matrix representing :math:`A`. Must be symmetric
            positive semi-definite and support ``A @ x``.
        b:
            Right-hand side vector.
        M_inv_batch:
            Batch of preconditioner matrices of shape ``(B, N, N)``, where each
            slice ``M_inv_batch[i]`` is a symmetric positive definite left preconditioner.
        rtol:
            Relative tolerance for convergence.
        atol:
            Absolute tolerance for convergence.
        wce_tol:
            Early stopping threshold on the worst-case error (WCE). Iteration
            terminates when WCE falls below this. Defaults to ``0.0`` (disabled).
        maxiter:
            Maximum number of iterations per solve.
        x0:
            Initial guess for each solve. If provided, must be broadcastable
            to shape ``(N,)``; otherwise a zero vector is used.

    Returns:
        tuple:
            A 4-tuple of batched results, each of shape ``(B, ...)``:

            - **x_batch** (Array, shape ``(B, N)``):
              Final solutions :math:`x_k` for each preconditioner in the batch.
            - **k_batch** (Array, shape ``(B,)``):
              Number of iterations performed for each solve.
            - **res_batch** (Array, shape ``(B, maxiter)``):
              Norms of the residuals :math:`\|r_k\|` at each iteration, for each batch.
            - **wce_batch** (Array, shape ``(B, maxiter)``):
              Worst-case error estimates :math:`\sigma(w_k)` at each iteration, for each batch.
    """
    # 1) bind the scalar/defaults into _pcg_core
    pcg_fixed = partial(
        _pcg_core,
        rtol=rtol,
        atol=atol,
        wce_tol=wce_tol,
        maxiter=maxiter,
        x0=x0,
    )

    # 2) vectorise over the 0th axis of M_inv_batch
    vmapped = jax.vmap(
        pcg_fixed,
        in_axes=(
            None,
            None,
            0,
        ),  # A: broadcast, b: broadcast, M_inv_batch: map over axis 0
        out_axes=(0, 0, 0, 0),  # each output gets a leading batch dim
    )

    # 3) call it
    x_batch, k_batch, res_batch, wce_batch = vmapped(A, b, M_inv_batch)
    return x_batch, k_batch, res_batch, wce_batch


# PCG with history JIT-compiled core function
@partial(jax.jit, static_argnames=("rtol", "atol", "wce_tol", "maxiter"), static_argnums=(0,))
def _pcg_core_history_static_mats(
    A,
    b,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    wce_tol: float = 0.0,
    maxiter: int = 1_000,
    M_inv: Optional[Array] = None,
    x0: Optional[Array] = None,
) -> tuple:
    r"""
    Core implementation of the Preconditioned Conjugate Gradient (PCG) method
    for solving the linear system :math:`Ax = b`.

    This internal function is JIT-compiled for performance and supports optional
    left preconditioning using a symmetric positive definite matrix :math:`M^{-1}`.
    It also tracks the worst-case error (WCE), which provides a computable
    upper bound on the error of kernel-based integration estimates.

    Args:
        A:
            Linear operator or matrix representing :math:`A`. Must be symmetric
            positive semi-definite and support ``A @ x``.
        b:
            Right-hand side vector.
        rtol:
            Relative tolerance for convergence.
        atol:
            Absolute tolerance for convergence.
        wce_tol:
            Early stopping criterion based on the worst-case error. Iteration terminates
            when WCE falls below this threshold. Defaults to ``0.0`` (disabled).
        maxiter:
            Maximum number of iterations.
        M_inv:
            Preconditioner matrix :math:`M^{-1}`. Must be symmetric positive definite.
        x0:
            Initial guess for the solution. If not provided, a zero vector is used.

    Returns:
        tuple:
            A 4-tuple containing:

            - **x** (``Array``): Final solution :math:`x_k` to the system.
            - **k** (``int``): Number of iterations performed.
            - **residuals** (``Array``): Norms of residuals :math:`\|r_k\|` at each iteration. Shape: ``(maxiter,)``.
            - **wce** (``Array``): Worst-case error estimates :math:`\sigma(w_k)` at each iteration. Shape: ``(maxiter,)``.
    """

    n = b.shape[0]
    if M_inv is None:
        M_inv = jnp.eye(n, dtype=b.dtype)
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # worst-case error estimate – stays inside the compiled graph
    def wce(w):  # √(wᵀAw) / (1ᵀw)
        return jnp.sqrt(w @ (A @ w)) / (
            jnp.sum(w) + 1e-30
        )  # protection against zero denominator

    r0 = b - A @ x0
    d0 = M_inv @ r0
    qdr0 = r0 @ (M_inv @ r0)
    res0 = jnp.linalg.norm(r0)
    eps = jnp.finfo(b.dtype).tiny  # protection against zero denominators in cg loop

    # carry: (k, done, x, r, d, qdr, res, res0)
    carry0 = (0, False, x0, r0, d0, qdr0, res0, res0)

    def body(carry, _):
        k, done, x, r, d, qdr, res, res0 = carry

        # branch when iterating (tolerances not met)
        def step(c):
            k, _, x, r, d, qdr, res, res0 = c
            den = d @ (A @ d)
            alpha = qdr / (den + eps)

            x_n = x + alpha * d
            r_n = r - alpha * (A @ d)
            qdr_n = r_n @ (M_inv @ r_n)
            beta = qdr_n / (qdr + eps)
            d_n = M_inv @ r_n + beta * d
            res_n = jnp.linalg.norm(r_n)

            k_n = k + 1
            wce_n = wce(x_n)  # worst-case error
            done_n = (
                (res_n <= atol)
                | (res_n <= rtol * res0)
                | (k_n >= maxiter)
                | (wce_n <= wce_tol)
            )

            carry_next = (k_n, done_n, x_n, r_n, d_n, qdr_n, res_n, res0)
            out_pair = (res, wce_n)  # residual before, worst-case error after
            return carry_next, out_pair

        # branch when converged (tolerances met): output dummy (will be sliced in public wrapper)
        def nop(c):
            k, *_ = c
            out_pair = (jnp.nan, jnp.nan)
            return c, out_pair

        return jax.lax.cond(done, nop, step, carry)

    carry_fin, (res_all, wce_all) = jax.lax.scan(body, carry0, xs=None, length=maxiter)

    x_final = carry_fin[2]
    k_final = carry_fin[0]  # iterations actually executed
    return x_final, k_final, res_all, wce_all
