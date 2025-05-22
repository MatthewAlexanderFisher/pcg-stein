import jax, jax.numpy as jnp
from functools import partial
from jax import Array
from typing import Optional

# 1.  PCG JIT-compiled core function


@partial(jax.jit, static_argnames=("rtol", "atol", "maxiter"))
def _pcg_core(
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
            positive semi-definite and support `A @ x`.
        b (Array):
            Right-hand side vector.
        rtol (float, optional):
            Relative tolerance for convergence. Defaults to 1e-6.
        atol (float, optional):
            Absolute tolerance for convergence. Defaults to 1e-6.
        wce_tol (float, optional):
            Early stopping criterion based on the worst-case error. Iteration terminates
            when WCE falls below this threshold. Defaults to 0.0 (disabled).
        maxiter (int, optional):
            Maximum number of iterations. Defaults to 1000.
        M_inv (Array, optional):
            Preconditioner matrix :math:`M^{-1}`. Must be symmetric positive definite.
        x0 (Array, optional):
            Initial guess for the solution. If not provided, a zero vector is used.

    Returns:
        tuple:
            A 4-tuple containing:

            - **x** (Array): Final solution :math:`x_k` to the system.
            - **k** (int): Number of iterations performed.
            - **residuals** (Array): Norms of residuals :math:`\|r_k\|` at each iteration. Shape: (maxiter,).
            - **wce** (Array): Worst-case error estimates :math:`\sigma(w_k)` at each iteration. Shape: (maxiter,).
    """

    n = A.shape[0]
    if M_inv is None:
        M_inv = jnp.eye(n, dtype=A.dtype)
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # worst-case error estimate – stays inside the compiled graph
    def wce_estimate(w):  # √(wᵀAw) / (1ᵀw)
        return jnp.sqrt(w @ (A @ w)) / (
            jnp.sum(w) + 1e-30
        )  # protection against zero denominator

    r0 = b - A @ x0
    d0 = M_inv @ r0
    qdr0 = r0 @ (M_inv @ r0)
    res0 = jnp.linalg.norm(r0)
    eps = jnp.finfo(A.dtype).tiny  # protection against zero denominators in cg loop

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
            wce_n = wce_estimate(x_n)  # worst-case error
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
) -> tuple[Array, int, Array, Array]:
    r"""
    Solves the linear system :math:`Ax = b` using the Preconditioned Conjugate Gradient (PCG) method,
    and returns diagnostic information including the worst-case error in an RKHS.

    This wrapper calls a JIT-compiled internal implementation and trims residual and error histories
    to the final iteration count. The worst-case error (WCE) provides a computable bound on the
    integration error when estimating expectations under a distribution using a weighted kernel
    quadrature rule.

    Args:
        A:
            Linear operator or matrix representing :math:`A` (assumed symmetric positive semi-definite).
        b (Array):
            Right-hand side vector.
        rtol (float, optional):
            Relative convergence tolerance. Defaults to 1e-6.
        atol (float, optional):
            Absolute convergence tolerance. Defaults to 1e-6.
        wce_tol (float, optional):
            Tolerance for worst-case error early stopping. If WCE drops below this, iteration stops.
            Defaults to 0.0 (disabled).
        maxiter (int, optional):
            Maximum number of iterations. Defaults to 1000.
        M_inv (Array, optional):
            Preconditioner (approximate inverse of :math:`A`). Defaults to `None` (no preconditioning).
        x0 (Array, optional):
            Initial guess for the solution. Defaults to zero if not provided.

    Returns:
        tuple:
            A 4-tuple containing:

            - **x** (Array): Final solution :math:`x_k` to the system :math:`Ax = b`.
            - **k** (int): Number of iterations performed.
            - **res_hist** (Array): Residual norms :math:`\|r_k\|` at each iteration. Shape: (k,).
            - **wce_hist** (Array): Worst-case error :math:`\sigma(\mathbf{w}_k)` at each iteration. Shape: (k,).

    Notes:
        The worst-case error is defined by:

        .. math::

            \sigma(\mathbf{w}_k) = \sup_{\|v\|_{\mathcal{H}(k_p)} \leq 1}
            \left| c_{N,k}(v) - \int v(\mathbf{x}) p(\mathbf{x}) \, \mathrm{d}\mathbf{x} \right|
            = \frac{ (\mathbf{w}_k^\top \mathbf{K}_p \mathbf{w}_k )^{1/2} }{ \mathbf{1}^\top \mathbf{w}_k },

        where :math:`\mathbf{w}_k` is the solution at iteration :math:`k`, and :math:`\mathbf{K}_p` is
        the kernel matrix under a preconditioned kernel. This provides a proxy for integration error
        when the RKHS norm :math:`\|v\|_{\mathcal{H}(k_p)}` is unknown.
    """

    x, k, res_all, wce_all = _pcg_core(
        A, b, rtol=rtol, atol=atol, wce_tol=wce_tol, maxiter=maxiter, M_inv=M_inv, x0=x0
    )
    return x, k, res_all[:k], wce_all[:k]
