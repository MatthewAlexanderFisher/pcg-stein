import jax, jax.numpy as jnp
from jax import Array, lax, tree_util
from typing import Optional, Tuple, Any
from functools import partial
from dataclasses import dataclass

from pcg_stein.linear import LinearOperator

class Preconditioner:
    """
    Base class for preconditioners.

    ``(n, n)`` is the shape of the original (square) matrix to be preconditioned.
    """

    name: str = "base_precon"
    display_name: str = "base_precon"

    def __call__(
        self, key: Array, matrix: Array, return_mat_and_precon: bool = False, **kwargs
    ) -> Array:
        r"""
        Apply the preconditioner to the matrix.

        Should be implemented in subclasses.

        Args:
            key:
                JAX random key, for any randomness required in the preconditioner.
            matrix:
                The matrix to precondition.
            return_mat_and_precon:
                Whether to return both the approximated matrix and its preconditioner (approximate inverse).
                If ``True``, returns the tuple ``(preconditioner, matrix_approx)``.
                If ``False``, returns only the preconditioner.
            **kwargs: Additional keyword arguments.

        Returns:
            Depends on subclass implementation.
        """
        raise NotImplementedError("Implemented in subclasses")

    def sub_matrix(
        self, key: Array, matrix: Array, m: int, indices: Optional[Array] = None
    ) -> tuple[Array, Array, Array]:
        r"""
        Extracts submatrices for the  Nyström (or other) preconditioner.

        Args:
            matrix:
                Input matrix. Shape: ``(n,n)``.
            key:
                JAX random key for sampling indices. Ignored if ``indices`` are provided.
            m:
                Number of columns/rows to select (size of the submatrix).
            indices:
                If given, use these indices directly. Shape: ``(m,)``.
                If not provided, sample indices randomly.

        Returns:
            tuple[Array, Array, Array]:

                - ``K_mm``: Submatrix for chosen indices. Shape: ``(m,m)``.
                - ``K_nm``: All rows, selected columns. Shape: ``(n,m)``.
                - ``K_mn``: All columns, selected rows. Shape: ``(m,n)``.

        """
        n = matrix.shape[0]  # size of matrix

        if indices is None:
            indices = jax.random.choice(key, jnp.arange(n), shape=(m,), replace=False)
            indices = jnp.sort(indices)

        K_mm = matrix[jnp.ix_(indices, indices)]
        K_nm = matrix[:, indices]
        K_mn = K_nm.T

        return K_mm, K_nm, K_mn

    def spectral_pinv(self, matrix: Array, max_cond_number: float = 1e14) -> Array:
        r"""
        Computes the Moore-Penrose pseudoinverse of a matrix, with optional spectral clipping.

        Args:
            matrix:
                The matrix to invert. Shape: ``(m,m)``.
            max_cond_number:
                Maximum allowed condition number (largest/smallest singular value ratio).
                If provided, singular values smaller than ``s_max / max_cond_number`` are clipped
                for numerical stability.

        Returns:
            Array:
                The pseudoinverse of the input matrix. Shape: ``(m,m)``.
        """

        U, s, Vh = jnp.linalg.svd(matrix, full_matrices=False)
        if max_cond_number is not None:
            s_max = s[0]
            clip_threshold = s_max / max_cond_number
            s = jnp.clip(s, a_min=clip_threshold, a_max=None)
        matrix_inv = (Vh.T * (1.0 / s)) @ U.T
        return matrix_inv

@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LowRankWoodbury:
    r"""
    Implements :math:`P = (I - K_{nm}\,W \,K_{mn}) / \mathrm{nugget}`.
    On vectors ``v`` this is :math:`P \, v`.
    """
    K_nm: Array
    W: Array
    nugget: float
    __array_priority__ = 10.0
    def __matmul__(self, v):   
        ktv = self.K_nm.T @ v
        b = self.W @ ktv
        kb = self.K_nm @ b
        return (v - kb) / self.nugget

    # pytree methods:
    def tree_flatten(self):
        # the arrays are leaves, nugget goes in aux
        return ((self.K_nm, self.W), self.nugget)

    @classmethod
    def tree_unflatten(cls, nugget, children):
        K_nm, W = children
        return cls(K_nm=K_nm, W=W, nugget=nugget)

class Nystrom(Preconditioner):
    r"""
    Random Index  Nyström preconditioner using the Woodbury formula.

    Approximates :math:`(K + \eta  I)^{-1}` using random column sampling and the Woodbury identity.
    """

    name: str = "Nystrom"
    display_name: str = "Nyström"

    def __call__(self, key: Array, matrix: Array, full_mat: bool = True, **kwargs) -> Array | LowRankWoodbury:
        r"""
        Construct the Nyström preconditioner using the Woodbury Inverse.

        Args:
            key:
                JAX random key for sampling indices. Ignored if ``indices`` are provided.
            matrix:
                Matrix to be preconditioned. Shape: ``(n,n)``.
            full_mat:
                If ``True`` returns the full shape ``(n,n)`` approximate inverse.
                If ``False``, return only the Woodbury‐form low‐rank components
                needed to apply the inverse in :math:`O(n m + m^2)` time and :math:`O(n m)` memory.
            m (``int``):
                Required in ``kwargs``. Number of columns/rows for Nyström approximation.
            max_cond_number (``float``):
                Maximum allowed condition number for pseudoinverse. Defaults to ``1e14``.
            indices (``Optional[Array]``):
                An optional array of indices to to compute the submatrix. If not provided, defaults
                to random sampling using ``key``.
            nugget (``float``):
                Required in ``kwargs``. Diagonal regulariser in Woodbury inverse.

        Returns:
            Array:
                Approximate inverse :math:`(\mathbf{K} + \eta \mathbf{I})^{-1}` constructed
                via random index sampling and the Woodbury identity. Shape: ``(n,n)``.
        """
        # ---------- kwargs & sanity checks -----------------------------
        m = kwargs.get("m")
        nugget = kwargs.get("nugget")
        max_cond_number = kwargs.get("max_cond_number", 1e14)
        indices = kwargs.get("indices", None)

        if m is None:
            raise ValueError("keyword `m` (number of inducing points) is required")
        if nugget is None:
            raise ValueError("keyword `nugget` (jitter) is required")

        # ---------- ensure symmetry ------------------------------------
        K = (matrix + matrix.T) * 0.5
        n = matrix.shape[0]

        #  Nyström submatrices
        K_mm, K_nm, K_mn = self.sub_matrix(key, K, m, indices=indices)

        # Woodbury: (nugget * K_mm + K_mn @ K_nm) is the low-rank correction
        matrix_woodbury = nugget * K_mm + K_mn @ K_nm  # shape (m, m)
        matrix_woodbury_inv = self.spectral_pinv(
            matrix_woodbury, max_cond_number=max_cond_number
        )

        # Final preconditioner approximation
        # Precon ≈ (K + nugget * I)^{-1} = (I - K_nm @ matrix_woodbury_inv @ K_mn) / nugget
        if full_mat is True:
            eye = jnp.eye(n)
            precon = (eye - K_nm @ matrix_woodbury_inv @ K_mn) / nugget

            return precon
        else:
            return LowRankWoodbury(K_nm, matrix_woodbury_inv, nugget)

class NystromRandom(Preconditioner):
    r"""
    Random Projection  Nyström preconditioner using the Woodbury formula.

    Approximates :math:`(\mathbf{K} + \eta \mathbf{I})^{-1}` where :math:`\mathbf{K}` is the input `matrix` and :math:`\eta`
    corresponds to the ``nugget`` parameter in the Woodbury inverse. Projects :math:`\mathbf{K}` into an m-dimensional subspace,
    using a random Gaussian matrix :math:`\mathbf{\Omega}` and applies the Woodbury identity to the resulting low-rank matrix.

    This method differs from standard Nyström by using a random Gaussian projection :math:`\mathbf{\Omega}` (not a random sampling of columns).
    """

    name: str = "NystromRandom"
    display_name: str = " Nyström (random projection)"

    def __call__(self, key: Array, matrix: Array, full_mat: bool = True, **kwargs) -> Array | LowRankWoodbury:
        r"""
        Construct the random projection Nyström preconditioner using the Woodbury formula.

        Args:
            key:
                JAX random key used in random projection.
            matrix:
                Matrix to be preconditioned. Shape: ``(n,n)``.
            full_mat:
                If ``True`` returns the full shape ``(n,n)`` approximate inverse.
                If ``False``, return only the Woodbury‐form low‐rank components
                needed to apply the inverse in :math:`O(n m + m^2)` time and :math:`O(n m)` memory.
            m (``int``):
                Required in ``kwargs``. Target dimension for the projection (rank of the approximation).
            max_cond_number (``float``):
                Maximum allowed condition number for pseudoinverse. Defaults to ``1e14``.
            nugget (``float``):
                Required in ``kwargs``. Diagonal regulariser in Woodbury inverse.

        Returns:
            Array:
                Approximate inverse :math:`(\mathbf{K} + \eta \mathbf{I})^{-1}` constructed via
                random projection and the Woodbury identity. Shape: ``(n,n)``.
        """
        # ---------- kwargs & sanity checks -----------------------------
        m = kwargs.get("m")
        nugget = kwargs.get("nugget")
        max_cond_number = kwargs.get("max_cond_number", 1e14)

        if m is None:
            raise ValueError("keyword `m` (number of inducing points) is required")
        if nugget is None:
            raise ValueError("keyword `nugget` (jitter) is required")

        # ---------- ensure symmetry ------------------------------------
        K = (matrix + matrix.T) / 2  # Ensure symmetry
        n = matrix.shape[0]

        # 1. Random Gaussian projection Ω: shape (m, n)
        omega = jax.random.normal(key, shape=(m, n))  # Standard normal entries

        # 2. Compute projected matrices
        # mat_project: (n, m), C: (m, m)
        mat_project = K @ omega.T  # n x m
        C = omega @ mat_project  # m x m

        # 3. Woodbury-style correction matrix (regularized)
        # matrix_woodbury = nugget * C + mat_project.T @ mat_project
        matrix_woodbury = nugget * C + mat_project.T @ mat_project  # m x m
        matrix_woodbury_inv = self.spectral_pinv(
            matrix_woodbury, max_cond_number=max_cond_number
        )

        # 4. Preconditioner via Woodbury identity
        if full_mat is True:
            eye = jnp.eye(n)
            # precon ≈ (K + nugget * I)^(-1) ≈ (I - mat_project @ matrix_woodbury_inv @ mat_project.T) / nugget
            precon = (eye - mat_project @ matrix_woodbury_inv @ mat_project.T) / nugget
            return precon
        else:
            return LowRankWoodbury(mat_project, matrix_woodbury_inv, nugget)


class NystromDiagonal(Preconditioner):
    r"""
    Diagonally-weighted Random Index Nyström preconditioner using the Woodbury formula.

    Approximates :math:`(K + \eta I)^{-1}` by sampling columns of :math:`K` with probability proportional to their diagonal
    values, and then applies the Woodbury identity for an approximate inverse.

    This method differs from standard Nyström by performing weighted (importance) column sampling.
    """

    name: str = "NystromDiagonal"
    display_name: str = " Nyström (diagonal sampling)"

    def __call__(self, key: Array, matrix: Array, full_mat: bool = True, **kwargs) -> Array | LowRankWoodbury:
        r"""
        Construct the diagonal-weighted  Nyström preconditioner using the Woodbury formula.

        Args:
            key:
                Random key for reproducibility in random projection.
            matrix:
                Symmetric matrix to be preconditioned. Shape ``(n, n)``.
            full_mat:
                If ``True`` returns the full shape ``(n,n)`` approximate inverse.
                If ``False``, return only the Woodbury‐form low‐rank components
                needed to apply the inverse in :math:`O(n m + m^2)` time and :math:`O(n m)` memory.
            m (``int``):
                 Required in ``kwargs``. Target rank - the number of columns/rows to sample for the  Nyström approximation.
            max_cond_number (``float``):
                Optional in ``kwargs``. Maximum allowed condition number for pseudoinverse. Defaults to ``1e14``.
            nugget (``float``):
                Required in ``kwargs``. Diagonal regulariser in Woodbury inverse.

        Returns:
            Array:
                A shape ``(n,n)`` preconditioner - an approximate inverse :math:`(K + \eta I)^{-1}` constructed via weighted Nyström sampling and Woodbury identity.
        """
        # ---------- kwargs & sanity checks -----------------------------
        m = kwargs.get("m")
        nugget = kwargs.get("nugget")
        max_cond_number = kwargs.get("max_cond_number", 1e14)

        if m is None:
            raise ValueError("keyword `m` (number of inducing points) is required")
        if nugget is None:
            raise ValueError("keyword `nugget` (jitter) is required")

        # ---------- ensure symmetry ------------------------------------
        K = (matrix + matrix.T) / 2  # Ensure symmetry
        n = matrix.shape[0]

        # Weighted sampling: probability proportional to diagonal entries
        diag_weights = jnp.diag(K)
        diag_weights = (
            diag_weights / diag_weights.sum()
        )  # JAX requires probabilities to sum to 1

        # Weighted random sampling of m unique indices
        indices = jax.random.choice(key, n, shape=(m,), replace=False, p=diag_weights)

        #  Nyström submatrices from weighted indices (key argument not used)
        K_mm, K_nm, K_mn = self.sub_matrix(None, K, m, indices=indices)

        # Woodbury: (nugget * K_mm + K_mn @ K_nm) is the low-rank correction
        matrix_woodbury = nugget * K_mm + K_mn @ K_nm  # shape (m, m)
        matrix_woodbury_inv = self.spectral_pinv(
            matrix_woodbury, max_cond_number=max_cond_number
        )

        if full_mat is True:
            # Final preconditioner via Woodbury identity
            # Precon ≈ (K + nugget * I)^{-1} ≈ (I - K_nm @ matrix_woodbury_inv @ K_mn) / nugget
            eye = jnp.eye(n)
            precon = (eye - K_nm @ matrix_woodbury_inv @ K_mn) / nugget

            return precon
        else:
            return LowRankWoodbury(K_nm, matrix_woodbury_inv, nugget)

class RandomisedEVD(Preconditioner):
    r"""
    An implementation of Algorithm 5.5 from Halko et. al. :cite:`halko2011finding`. Computes a low-rank Woodbury inverse preconditioner using Randomised
    Eigenvalue Decomposition (EVD) with power iterations.

    This approximates :math:`(\mathbf{K} + \eta \mathbf{I})^{-1}` via power itrations to get the dominant subspace, uses and eigenvalue
    decomposition to form the low-rank approximation, and then applies the Woodbury formula for efficient approximate inversion.
    Assumes :math:`\mathbf{K}` is symmetric and PSD.
    """

    name: str = "RandomisedEVD"
    display_name: str = "Randomised Nyström EVD"

    def __call__(self, key: Array, matrix: Array, full_mat: bool = True, **kwargs) -> Array | LowRankWoodbury:
        r"""
        Construct a randomised EVD-based Woodbury preconditioner.

        Args:
            key:
                JAX random key used in random projection.
            matrix:
                Symmetric matrix to be preconditioned. Shape ``(n, n)``.
            full_mat:
                If ``True`` returns the full shape ``(n,n)`` approximate inverse.
                If ``False``, return only the Woodbury‐form low‐rank components
                needed to apply the inverse in :math:`O(n m + m^2)` time and :math:`O(n m)` memory.
            m (``int``):
                Required in ``kwargs``. Target rank of the approximation.
            nugget (``float``):
                Required in ``kwargs``. Diagonal regulariser in Woodbury inverse.
            return_mat_and_precon (``bool``):
                Whether to return both the approximated matrix and its preconditioner (approximate inverse).
                If ``True``, returns a tuple ``(preconditioner, matrix_approx)``.
                If ``False`` (default behaviour), returns only ``preconditioner``.
            n_iter (``int``):
                Optional in ``kwargs``. Number of power iterations. Defaults to ``2``.
            tau (``float``):
                Optional in ``kwargs``. Tikhonov regularisation parameter for the inversion of eigenvalues (default ``0.0``).
            max_cond_number (``float``):
                Maximum allowed condition number for pseudoinverse. Defaults to ``1e14``.

        Returns:
            Array:
                Approximate inverse :math:`(\mathbf{K} + \eta \mathbf{I})^{-1}` constructed via randomised EVD and the Woodbury identity.
        """
        # ---------- kwargs & sanity checks -----------------------------
        m = kwargs.get("m")
        nugget = kwargs.get("nugget")
        tau = kwargs.get("tau", 0.0)
        n_iter = kwargs.get("n_iter", 2)
        return_mat_and_precon = kwargs.get("return_mat_and_precon", False)
        max_cond_number = kwargs.get("max_cond_number", 1e14)

        if m is None:
            raise ValueError("keyword `m` (rank of the approximation) is required")
        if nugget is None:
            raise ValueError("keyword `nugget` (jitter) is required")

        K = matrix  
        n = matrix.shape[0]

        # ---------- range find -----------------------------------------
        omega = jax.random.normal(key, (n, m))  # (n, m)

        # 2. Y = K @ omega, then power iterations
        Y = K @ omega
        for _ in range(n_iter):
            Y = K @ (K.T @ Y)
            Q, _ = jnp.linalg.qr(Y)
            Y = Q

        Q, _ = jnp.linalg.qr(Y)  # Final orthonormal basis (n, m)

        # ---------- approximate -----------------------------------------

        # Projections
        B1 = K @ Q  # (n, m)
        B2 = Q.T @ B1  # (m, m)

        # Cholesky
        C = jnp.linalg.cholesky(B2)

        # Triangle solve
        F = jax.scipy.linalg.solve_triangular(C, B1.T, lower=True).T

        # SVD
        U, S, V = jnp.linalg.svd(
            F, full_matrices=False
        )  # shapes:  U (n, m)   S (m,)   Vt (m, m)

        # Compute Λ and Λ⁻¹ (using optional Tikhonov regularisation)
        Lambda = S**2  # (m, m)
        Lambda_inv = jnp.diag(1 / (Lambda + tau))  # Λ⁻¹

        # Woodbury inverse to compute preconditioner
        middle = nugget * Lambda_inv + jnp.eye(U.shape[1])  #  σΛ⁻¹ + I
        middle_inv = self.spectral_pinv(middle, max_cond_number)

        if full_mat is True:
            eye = jnp.eye(n)
            precon = (eye / nugget) - (U @ middle_inv @ U.T) / nugget

            if return_mat_and_precon is True:
                K_approx = U @ Lambda @ U
                return precon, K_approx

            return precon
        else:
            return LowRankWoodbury(U, middle_inv, nugget)

@dataclass(frozen=True)
class LowRankFITC:
    r"""
    Implements :math:`P = \Lambda^{-1} - \Lambda^{-1} K_{nm} W K_{nm}^\top \Lambda^{-1}`.
    On vectors ``v`` this is :math:`P \, v`.
    """
    K_nm: jnp.ndarray
    W: jnp.ndarray
    lambda_inv: jnp.ndarray
    __array_priority__ = 10.0
    def __matmul__(self, v):   
        lv = self.lambda_inv * v
        t  = self.K_nm.T @ lv
        u  = self.W @ t
        z  = self.K_nm   @ u
        return lv - z

class FITC(Preconditioner):
    r"""
    Implementation of Fully–Independent Training Conditional (FITC) preconditioner from :cite:`quinonero2005unifying`.
    """

    name: str = "FITC"
    display_name: str = "FITC"

    def __call__(self, key: Array, matrix: Array, full_mat: bool = True, **kwargs) -> Array | LowRankFITC:
        r"""
        Constructs a preconditioner using the FITC (Fully Independent Training Conditional) approximation.

        Args:
            key:
                Random key used in choice of inducing points. Ignored if ``indices`` are provided.
            matrix:
                Symmetric matrix to be preconditioned. Shape: ``(n,n)``.
            m (``int``):
                Required in ``kwargs``. The number of inducing points to use - the rank of the approximation.
            nugget (``float``):
                Required in ``kwargs``. Diagonal regulariser in Woodbury inverse.
            indices (``Array`` or ``list[int]``):
                Explicit array of inducing indices to use. If not provided, indices are generated randomly using ``key``.
            max_cond_number (``float``):
                Maximum allowed condition number for pseudoinverse. Defaults to ``1e14``.

        Returns:
            Array:
                A matrix representing the FITC-based preconditioner.
        """

        # ---------- kwargs & sanity checks -----------------------------
        m = kwargs.get("m")
        nugget = kwargs.get("nugget")
        max_cond_number = kwargs.get("max_cond_number", 1e14)
        indices = kwargs.get("indices", None)

        if m is None:
            raise ValueError("keyword `m` (number of inducing points) is required")
        if nugget is None:
            raise ValueError("keyword `nugget` (jitter) is required")

        # ---------- ensure symmetry ------------------------------------
        K = (matrix + matrix.T) * 0.5

        # ---------- form sub-matrices ----------------------------------
        K_mm, K_nm, K_mn = self.sub_matrix(
            key, K, m, indices=indices
        )  # (m,m) (n,m) (m,n)

        # ---------- Q = K_nm K_mm^{-1} K_mn ----------------------------
        K_mm_inv = self.spectral_pinv(K_mm, max_cond_number=max_cond_number)
        Q = K_nm @ K_mm_inv @ K_mn

        # ---------- Λ  = diag(K − Q) + σ² I and  Λ^{-1} ---------------
        d = jnp.diag(K - Q) + nugget
        Lambda_inv = jnp.diag(1.0 / d)

        # ---------- small m×m matrix  M = K_mm + K_mn Λ^{-1} K_nm -----
        M = K_mm + K_mn @ Lambda_inv @ K_nm
        M_inv = self.spectral_pinv(M, max_cond_number=max_cond_number)

        # ---------- FITC inverse via Woodbury --------------------------
        if full_mat is True:
            return Lambda_inv - Lambda_inv @ (K_nm @ M_inv @ K_mn) @ Lambda_inv
        else:
            return LowRankFITC(K_nm, M_inv, jnp.diag(Lambda_inv))

class BlockJacobi(Preconditioner):

    name: str = "BlockJacobi"
    display_name: str = "Block Jacobi"

    def __call__(self, key: Array, matrix: Array, **kwargs: Any) -> Array:
        r"""
        Constructs a preconditioner using the Block Jacobi approximation, where the
        matrix is partitioned into diagonal blocks and each is inverted separately.

        Args:
            key:
                JAX PRNG key (ignored, included for API consistency).
            matrix:
                Symmetric PSD matrix of shape ``(n, n)`` to precondition.
            block_size (``int``):
                Required by ``kwargs``. Size of the diagonal blocks used in the Jacobi approximation.
            max_cond_number (``float``):
                Maximum allowed condition number for pseudoinverse. Defaults to ``1e14``.

        Returns:
            Array:
                Preconditioning matrix constructed via block-wise spectral pseudo-inversion.
        """

        B = kwargs.get("block_size")
        max_cond = kwargs.get("max_cond_number", 1e14)
        if B is None:
            raise ValueError("`block_size` required")

        # Ensure symmetry
        K = 0.5 * (matrix + matrix.T)
        return _block_jacobi_jit(K, B, max_cond, self.spectral_pinv)


@partial(jax.jit, static_argnums=(1, 3))
def _block_jacobi_jit(
    K: Array, block_size: Array, max_cond: Array, spectral_pinv: Any
) -> Array:
    n = K.shape[0]
    nb = (n + block_size - 1) // block_size
    padded_n = nb * block_size

    # 1) pad to (padded_n, padded_n)
    pad_amt = padded_n - n
    Kp = jnp.pad(K, ((0, pad_amt), (0, pad_amt)))

    # 2) reshape into blocks
    #    shape -> (nb, block_size, nb, block_size)
    Kb = Kp.reshape(nb, block_size, nb, block_size)

    # 3) extract just the diagonal blocks: shape (nb, block_size, block_size)
    diag_blocks = Kb[jnp.arange(nb), :, jnp.arange(nb), :]

    # 4) invert each block in parallel
    def inv_block(block):
        return spectral_pinv(block, max_cond_number=max_cond)

    inv_blocks = jax.vmap(inv_block)(diag_blocks)

    # 5) scatter them back into a zero-padded precon matrix
    precon_blocks = jnp.zeros_like(Kb)
    precon_blocks = precon_blocks.at[jnp.arange(nb), :, jnp.arange(nb), :].set(
        inv_blocks
    )

    # 6) reshape & unpad to (n, n)
    precon_padded = precon_blocks.reshape(padded_n, padded_n)
    return precon_padded[:n, :n]
