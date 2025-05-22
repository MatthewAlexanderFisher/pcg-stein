import jax, jax.numpy as jnp
from jax import Array
from typing import Optional, Tuple, Any


class Preconditioner:
    """
    Base class for preconditioners.

    n x n is the size of the original (square) matrix to be preconditioned.
    m x m is the size of the preconditioner or the number of columns in the low-rank approximation.
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
            key: Array
                JAX random key, for any stochasticity required in the preconditioner.
            matrix: Array
                The matrix to precondition.
            return_mat_and_precon: bool (default False)
                Whether to return both the approximated matrix and its preconditioner (approximate inverse).
                If True, the function returns a tuple `(preconditioner, matrix_approx)`; otherwise, only the preconditioner.
            **kwargs: Additional keyword arguments for flexibility.

        Returns:
            Depends on subclass implementation.
        """
        raise NotImplementedError("Implemented in subclasses")

    def sub_matrix(
        self, key: Array, matrix: Array, m: int, indices: Optional[Array] = None
    ) -> Tuple[Array, Array, Array]:
        r"""
        Extracts submatrices for the Nystrom (or other) preconditioner.

        Args:
            matrix: Array
                Input matrix (shape: [n, n]).
            key: Array
                JAX random key for sampling indices, if not provided.
            m: int
                Number of columns/rows to select (size of the submatrix).
            indices: Optional[Array], optional
                If given, use these indices directly (shape: [m,]).
                If None, sample indices randomly.

        Returns:
            K_mm: Array
                Submatrix for chosen indices (shape: [m, m]).
            K_nm: Array
                All rows, selected columns (shape: [n, m]).
            K_mn: Array
                All columns, selected rows (shape: [m, n]).
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
            matrix: Array
                The matrix to invert (shape: [m, m]).
            max_cond_number: float, optional
                Maximum allowed condition number (largest/smallest singular value ratio).
                If provided, singular values smaller than s_max / max_cond_number are clipped
                for numerical stability. Default is 1e14.

        Returns:
            matrix_inv: Array
                The pseudoinverse of the input matrix (shape: [m, m]).
        """

        U, s, Vh = jnp.linalg.svd(matrix, full_matrices=False)
        if max_cond_number is not None:
            s_max = s[0]
            clip_threshold = s_max / max_cond_number
            s = jnp.clip(s, a_min=clip_threshold, a_max=None)
        matrix_inv = (Vh.T * (1.0 / s)) @ U.T
        return matrix_inv


class Nystrom(Preconditioner):
    r"""
    Random Index Nystrom preconditioner using the Woodbury formula.

    Approximates :math:`(K + \eta  I)^{-1}` using random column sampling and the Woodbury identity.
    """

    name: str = "Nystrom"  
    display_name: str = "Nystrom"  

    def __call__(self, key: Array, matrix: Array, **kwargs) -> Array:
        r"""
        Construct the Nystrom preconditioner using the Woodbury Inverse.

        Args:
            key: Array
                JAX random key for sampling indices, if not provided.
            matrix: Array
                Input matrix (shape: [n, n]).
            m: int, required.
                Number of columns/rows for Nystrom approximation.
            max_cond_number: float, optional
                Maximum allowed condition number for pseudoinverse (default 1e14).
            indices: array or None, optional.
                Explicit indices for Nystrom subset.
            nugget: float, required.
                Diagonal regulariser in Woodbury inverse.

        Returns:
            precon: (n, n) matrix.
                Approximation to (K + nugget * I)^{-1} using the Nystrom-Woodbury formula.
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

        # Nystrom submatrices
        K_mm, K_nm, K_mn = self.sub_matrix(key, K, m, indices=indices)

        # Woodbury: (nugget * K_mm + K_mn @ K_nm) is the low-rank correction
        matrix_woodbury = nugget * K_mm + K_mn @ K_nm  # shape (m, m)
        matrix_woodbury_inv = self.spectral_pinv(
            matrix_woodbury, max_cond_number=max_cond_number
        )

        # Final preconditioner approximation
        # Precon ≈ (K + nugget * I)^{-1} = (I - K_nm @ matrix_woodbury_inv @ K_mn) / nugget
        eye = jnp.eye(n)
        precon = (eye - K_nm @ matrix_woodbury_inv @ K_mn) / nugget

        return precon


class NystromRandom(Preconditioner):
    """
    Random Projection Nystrom preconditioner using the Woodbury formula.

    Approximates (K + nugget * I)^{-1} by projecting the n x n matrix K into an m-dimensional random subspace,
    and applying the Woodbury identity to the resulting low-rank matrix.

    This method differs from standard Nystrom by using a random Gaussian projection Ω (not random sampling of columns).
    """

    name: str = "NystromRandom"  
    display_name: str = "Nystrom (random projection)"  

    def __call__(self, key: Array, matrix: Array, **kwargs) -> Array:
        """
        Construct the random projection Nystrom preconditioner using the Woodbury formula.

        Args:
            key: Array
                Random key for reproducibility in random projection.
            matrix: Array (n, n)
                Symmetric matrix to be preconditioned.
            m: int, required in kwargs
                Target dimension for the projection (rank of the approximation).
            max_cond_number: float, optional
                Maximum allowed condition number for pseudoinverse (default 1e14).
            nugget: float, required.
                Diagonal regulariser in Woodbury inverse.

        Returns:
            precon: Array (n, n)
                Approximate inverse (K + nugget * I)^{-1} constructed via random projection and the Woodbury identity.
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
        eye = jnp.eye(n)
        # precon ≈ (K + nugget * I)^(-1) ≈ (I - mat_project @ matrix_woodbury_inv @ mat_project.T) / nugget
        precon = (eye - mat_project @ matrix_woodbury_inv @ mat_project.T) / nugget

        return precon


class NystromDiagonal(Preconditioner):
    r"""
    Diagonally-weighted Random Index Nystrom preconditioner using the Woodbury formula.

    Approximates :math:`(K + \eta I)^{-1}` by sampling columns of :math:`K` with probability proportional to their diagonal
    values, and then applies the Woodbury identity for an approximate inverse.

    This method differs from standard Nystrom by performing weighted (importance) column sampling.
    """

    name: str = "NystromDiagonal"  
    display_name: str = "Nystrom (diagonal sampling)"  

    def __call__(self, key: Array, matrix: Array, **kwargs) -> Array:
        r"""
        Construct the diagonal-weighted Nystrom preconditioner using the Woodbury formula.

        Args:
            key: Array
                Random key for reproducibility in random projection.
            matrix: Array (n, n)
                Symmetric matrix to be preconditioned.
            m: int, required in kwargs.
                Target rank - the number of columns/rows to sample for the Nystrom approximation.
            max_cond_number: float, optional in kwargs.
                Maximum allowed condition number for pseudoinverse (default 1e14).
            nugget: float, required in kwargs.
                Diagonal regulariser in Woodbury inverse.

        Returns:
            precon: Array (n, n)
                Approximate inverse :math:`(K + \eta I)^{-1}` constructed via weighted Nystrom sampling and Woodbury identity.
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

        # Nystrom submatrices from weighted indices (key argument not used)
        K_mm, K_nm, K_mn = self.sub_matrix(None, K, m, indices=indices)

        # Woodbury: (nugget * K_mm + K_mn @ K_nm) is the low-rank correction
        matrix_woodbury = nugget * K_mm + K_mn @ K_nm  # shape (m, m)
        matrix_woodbury_inv = self.spectral_pinv(
            matrix_woodbury, max_cond_number=max_cond_number
        )

        # Final preconditioner via Woodbury identity
        # Precon ≈ (K + nugget * I)^{-1} ≈ (I - K_nm @ matrix_woodbury_inv @ K_mn) / nugget
        eye = jnp.eye(n)
        precon = (eye - K_nm @ matrix_woodbury_inv @ K_mn) / nugget

        return precon


class RandomisedSVD(Preconditioner):
    r"""
    Computes a low-rank Woodbury inverse preconditioner using Randomized SVD with power iterations.

    This approximates :math:`(K + \eta I)^{-1}` via a randomized SVD to get the dominant subspace,
    and then applies the Woodbury formula for efficient approximate inversion. Assumes K is symmetric or PSD.
    """

    name: str = "RandomisedSVD" 
    display_name: str = "Randomised SVD"

    def __call__(self, key: Array, matrix: Array, **kwargs) -> Array:
        """
        Construct a randomized SVD-based Woodbury preconditioner.

        Args:
            key: Array
                Random key for reproducibility in random projection.
            matrix: Array (n, n)
                Symmetric matrix to be preconditioned.
            m: int, required in kwargs
                Target rank of the approximation.
            nugget: float, required in kwargs.
                Diagonal regulariser in Woodbury inverse.
            return_mat_and_precon: bool (default False)
                Whether to return both the approximated matrix and its preconditioner (approximate inverse).
                If True, the function returns a tuple `(preconditioner, matrix_approx)`; otherwise, only the preconditioner.
            n_iter: int, optional in kwargs
                Number of power iterations (default 2).
            tau: float, optional in kwargs
                Tikhonov regularisation parameter for the inversion of eigenvalues (default 0.0).
            max_cond_number: float, optional
                Maximum allowed condition number for pseudoinverse (default 1e14).

        Returns:
            Array:
                Approximate inverse (K + nugget * I)^{-1} constructed via randomized SVD and the Woodbury identity.
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

        # ---------- ensure symmetry ------------------------------------
        K = (matrix + matrix.T) / 2  # Ensure symmetry
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

        # projections
        B1 = K @ Q  # (n, m)
        B2 = Q.T @ B1  # (m, m)

        # cholesky
        C = jnp.linalg.cholesky(B2)

        # triangle solve
        F = jax.scipy.linalg.solve_triangular(C, B1.T, lower=True).T

        # svd
        U, S, V = jnp.linalg.svd(
            F, full_matrices=False
        )  # shapes:  U (1000, 50)   S (50,)   Vt (50, 50)

        # Compute Λ and Λ⁻¹ (using optional Tikhonov regularisation)
        Lambda = S**2  # (50, 50)
        Lambda_inv = jnp.diag(1 / (Lambda + tau))  # Λ⁻¹

        # Woodbury inverse to compute preconditioner
        middle = nugget * Lambda_inv + jnp.eye(U.shape[1])  #  σΛ⁻¹ + I
        middle_inv = self.spectral_pinv(middle, max_cond_number)

        eye = jnp.eye(n)
        precon = (eye / nugget) - (U @ middle_inv @ U.T) / nugget

        if return_mat_and_precon is True:
            K_approx = U @ Lambda @ U
            return precon, K_approx

        return precon


class FITC(Preconditioner):
    """Fully–Independent Training Conditional (FITC) pre-conditioner."""

    name: str = "FITC" 
    display_name: str = "FITC"

    def __call__(self, key: Array, matrix: Array, **kwargs) -> Array:
        """
        Constructs a preconditioner using the FITC (Fully Independent Training Conditional) approximation.

        Args:
            key: Array
                Random key for reproducibility in random projection.
            matrix: Array (n, n)
                Symmetric matrix to be preconditioned.
            m (int): required in kwargs.
                Number of inducing points (required).
            nugget (float): required in kwargs.
                Diagonal regulariser in Woodbury inverse.
            indices (Array or list of int, optional):
                Explicit array of inducing indices to use. If not provided, indices are selected using `key`.
            max_cond_number: float, optional
                Maximum allowed condition number for pseudoinverse (default 1e14).

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
        return Lambda_inv - Lambda_inv @ (K_nm @ M_inv @ K_mn) @ Lambda_inv


class BlockJacobi(Preconditioner):

    name: str = "BlockJacobi"
    display_name: str = "Block Jacobi"

    def __call__(self, key: Array, matrix: Array, **kwargs: Any) -> Array:
        """
        Constructs a preconditioner using the Block Jacobi approximation, where the
        matrix is partitioned into diagonal blocks and each is inverted separately.

        Args:
            key (Array):
                JAX PRNG key (ignored, included for API consistency).
            matrix (Array):
                Symmetric positive semi-definite matrix of shape (n, n) to precondition.
            block_size (int, keyword-only):
                Size of the diagonal blocks used in the Jacobi approximation (required).
            max_cond_number: float, optional
                Maximum allowed condition number for pseudoinverse (default 1e14).

        Returns:
            Array:
                Preconditioning matrix constructed via block-wise spectral pseudo-inversion.
        """
        # ---------- kwargs & sanity checks -----------------------------
        block_size = kwargs.get("block_size")
        max_cond_number = kwargs.get("max_cond_number", 1e14)

        if block_size is None:
            raise ValueError("Keyword argument `block_size` is required.")

        # ---------- ensure symmetry ------------------------------------
        K = (matrix + matrix.T) * 0.5
        n = matrix.shape[0]

        # ---------- allocate result ------------------------------------
        precon = jnp.zeros_like(K)

        # ---------- loop over blocks -----------------------------------
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)  # allow ragged last block
            block = K[start:stop, start:stop]

            block_inv = self.spectral_pinv(block, max_cond_number=max_cond_number)
            precon = precon.at[start:stop, start:stop].set(block_inv)

        return precon
