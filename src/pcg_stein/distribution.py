from jax import grad
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array


class Distribution:

    def __init__(self):
        self.set_mh_step_jit()

    def log_prob(self, x: Array):
        raise NotImplementedError

    def score(self, x: Array) -> Array:
        # Vectorise the gradient calculation
        grad_fn = jax.vmap(grad(lambda single_x: self.log_prob(single_x).sum()))
        return grad_fn(x)

    def sample(self, key: Array, num_samples: int):
        """
        Sample `num_samples` from the distribution with a provided JAX key.
        """
        raise NotImplementedError

    def _propose(self, key: Array, beta: Array, step_size: float) -> Array:
        return beta + step_size * jax.random.normal(key, beta.shape)

    def _mh_step(
        self, key: Array, state: tuple[Array, Array], step_size: float
    ) -> tuple[Array, ...]:
        beta, logp = state
        key_prop, key_u = jax.random.split(key)
        beta_prop = self._propose(key_prop, beta, step_size)
        logp_prop = self.log_prob(beta_prop)
        log_alpha = jnp.minimum(0.0, logp_prop - logp)  # log of accept prob
        accept = jnp.log(jax.random.uniform(key_u)) < log_alpha
        beta_new = jnp.where(accept, beta_prop, beta)
        logp_new = jnp.where(accept, logp_prop, logp)
        return (beta_new, logp_new), accept

    # JIT‑compiled single step

    def set_mh_step_jit(self):
        self._mh_step_jit = jax.jit(self._mh_step, static_argnums=(2))


class BayesianLogisticRegression(Distribution):
    """Bayesian logistic regression with Gaussian prior and MH sampler.

    Use :py:meth:`from_synthetic` to build a model instance directly from a
    *true* parameter vector – handy for demos, teaching, or simulation studies.
    """

    @classmethod
    def from_synthetic(
        cls,
        key: Array,
        n_samples: int,
        true_beta: Array,
        *,
        add_intercept: bool = True,
        X: Optional[Array] = None,
        sigma_prior: float | Array = 1.0,
    ) -> "BayesianLogisticRegression":
        r"""
        Generate object with generated synthetic data (X, y) from a known parameter vector.

        Args:
            key (Array):
                JAX PRNG key used for random number generation.
            n_samples (int):
                Number of samples (rows) to generate.
            true_beta (Array):
                1D array of shape (d,). If `add_intercept=True`, the first element is treated
                as the intercept; the remaining elements correspond to slopes. Otherwise, all
                elements are treated as slopes.
            add_intercept (bool, optional):
                If True, prepends a column of ones to the design matrix X so that its number
                of columns matches `len(true_beta)`. Defaults to True.
            X (Array, optional):
                Optional design matrix to reuse. See "Shape rules" below for requirements
                depending on `add_intercept`.
            sigma_prior (float or Array, optional):
                Standard deviation(s) of the Gaussian prior distribution over the true
                coefficient vector :math:`\beta \sim \mathcal{N}(0, \sigma_{\text{prior}}^2 I)`.
                Controls the magnitude of sampled coefficients. Defaults to 1.0.

        Shape rules:

            -If `add_intercept=True`:
                - If `X` is `None`, a random matrix of shape `(n, d - 1)` is drawn and a ones column is prepended.
                - If `X` is provided, it may:
                    - Have shape `(n, d - 1)` → a ones column is prepended.
                    - Have shape `(n, d)`     → it is used as-is, with its own intercept.

            -If `add_intercept=False`:
                - `X` must already have `d` columns (matching `len(true_beta)`).
        """

        d = true_beta.shape[0]

        #   Build design matrix X
        if X is None:
            key, subkey = jax.random.split(key)
            p = d - 1 if add_intercept else d
            X = jax.random.normal(subkey, (n_samples, p))
            if add_intercept:
                X = jnp.concatenate((jnp.ones((n_samples, 1)), X), axis=1)
        else:
            if add_intercept:
                if X.shape == (n_samples, d - 1):
                    X = jnp.concatenate((jnp.ones((n_samples, 1)), X), axis=1)
                elif X.shape != (n_samples, d):
                    raise ValueError(
                        "With add_intercept=True, X must have d‑1 or d columns."
                    )
            else:
                if X.shape != (n_samples, d):
                    raise ValueError(
                        "With add_intercept=False, X must have exactly d columns."
                    )

        #   Simulate y outcomes
        probs = jax.nn.sigmoid(X @ true_beta)
        key, subkey = jax.random.split(key)
        y = jax.random.bernoulli(subkey, probs)
        return cls(X, y, sigma_prior)

    def __init__(self, X: Array, y: Array, sigma_prior: float | Array = 10.0):

        super().__init__()

        if y.ndim != 1:
            raise ValueError("`y` must be a 1‑D array of 0/1 labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        self.X = X
        self.y = y

        # Allow scalar or vector prior scale.
        if jnp.ndim(sigma_prior) == 0:
            self.sigma_prior = jnp.full(X.shape[1], float(sigma_prior))
        else:
            if sigma_prior.shape != (X.shape[1],):
                raise ValueError("`sigma_prior` must be a scalar or length‑d vector.")
            self.sigma_prior = sigma_prior

    ###  Log‑probability functions
    @staticmethod
    def _log_sigmoid(x: Array) -> Array:
        """Numerically stable `log(sigmoid(x))`."""
        return -jax.nn.softplus(-x)

    @partial(jax.jit, static_argnums=0)
    def log_likelihood(self, beta: Array) -> Array:
        t = self.X @ beta  # shape (n,)
        return jnp.sum(
            self.y * self._log_sigmoid(t) + (1 - self.y) * self._log_sigmoid(-t)
        )

    @partial(jax.jit, static_argnums=0)
    def log_prior(self, beta: Array) -> Array:
        # Normal(0, sigma_prior^2) independent components.
        lp = -0.5 * jnp.sum((beta / self.sigma_prior) ** 2)
        lp -= jnp.sum(
            jnp.log(self.sigma_prior * jnp.sqrt(2 * jnp.pi))
        )  # constant helpful for diagnostics
        return lp

    def log_prob(self, beta: Array) -> Array:
        return self.log_likelihood(beta) + self.log_prior(beta)

    ###  Metropolis–Hastings sampler
    def sample(
        self,
        key: Array,
        num_samples: int,
        *,
        step_size: float = 0.1,
        burn_in: int = 1_000,
        thinning: int = 1,
        init_beta: Optional[Array] = None,
        unique: bool = True,
    ) -> Array:
        """Draws `num_samples` posterior samples (after burn‑in & thinning).

        Args:
            key : Array
                PRNG key.
            num_samples : int
                Number of stored samples returned.
            step_size : float, default 0.1
                Std. dev. of random‑walk proposal.
            burn_in : int, default 1000
                Number of warm‑up iterations (discarded).
            thinning : int, default 1
                Keep every `thinning`‑th draw after burn‑in.
            init_beta : Array | None
                Initial parameter vector.  If None, starts at 0.
            unique : bool, default True
                Compile core MH step with `jax.jit` (worthwhile for long chains).
        Returns:
            JAX array of MH samples.
        """
        if thinning < 1:
            raise ValueError("`thinning` must be >= 1")

        if init_beta is None:
            init_beta = jnp.zeros(self.X.shape[1])
        logp = self.log_prob(init_beta)

        mh_step = self._mh_step_jit

        # Burn‑in
        beta = init_beta
        key_i = key
        for _ in range(burn_in):
            key_i, subkey = jax.random.split(key_i)
            (beta, logp), _ = mh_step(subkey, (beta, logp), step_size)

        # Sampling
        samples = []
        accept_count = 0  # counts accepted proposals (for unique branch)
        iter_count = 0  # counts total iterations (for non‑unique branch)

        while len(samples) < num_samples:
            key_i, subkey = jax.random.split(key_i)
            (beta, logp), accept = mh_step(subkey, (beta, logp), step_size)
            if unique:
                if bool(accept):
                    accept_count += 1
                    if accept_count % thinning == 0:
                        samples.append(beta)
            else:
                iter_count += 1
                if iter_count % thinning == 0:
                    samples.append(beta)
        return jnp.stack(samples)
