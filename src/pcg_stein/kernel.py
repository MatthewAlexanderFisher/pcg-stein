import functools
import jax, jax.numpy as jnp
from jax import jit, vmap, Array, lax
from typing import Optional, Any, Callable, Tuple
from functools import partial
from dataclasses import dataclass

from pcg_stein.linear import LinearOperator


class Kernel:
    name: str = "base_kernel"  # class attribute

    # radial profile and its two helpers — must be provided by subclass
    @staticmethod
    def _phi(r: Array, **hyper):
        raise NotImplementedError  # ϕ(r)

    @staticmethod
    def _psi(r: Array, **hyper):
        raise NotImplementedError  # ϕ′(r)/r  (finite at 0)

    @staticmethod
    def _phi_pp(r: Array, **hyper):
        raise NotImplementedError  # ϕ″(r)

    def _pair(self, x: Array, y: Array, **hyper: float) -> Array:
        """
        Evaluates the base kernel function between two samples.

        Args:
            x:
                A single sample of shape ``(d,)``.
            y:
                A single sample of shape ``(d,)``.
            **hyper:
                Additional kernel hyperparameters.

        Returns:
            Array:
                A scalar representing the kernel evaluation between x and y.
        """
        r = jnp.linalg.norm(x - y)
        return self._phi(r, **hyper)

    def __call__(self, X: Array, Y: Optional[Array] = None, **hyper: float) -> Array:
        """
        Evaluates the base kernel or its Gram matrix.

        If ``Y`` is not provided, computes the Gram matrix ``K(X, X)`` between all pairs in ``X``.
        If ``Y`` is provided, computes the cross Gram matrix ``K(X, Y)``.

        Args:
            X:
                An shape ``(n,d)`` array of input samples.
            Y:
                A shape ``(m,d)`` array of input samples. If None, defaults to ``X``.
            **hyper:
                Additional kernel hyperparameters.

        Returns:
            Array:
                The shape ``(n,m)`` Gram matrix of base kernel evaluations, or a scalar
                if both `X` and `Y` are single points.
        """
        if Y is None:
            Y = X

        X = jnp.atleast_2d(X)
        Y = jnp.atleast_2d(Y)

        pair = functools.partial(self._pair, **hyper)

        def gram(A, B):
            return vmap(lambda x: vmap(lambda y: pair(x, y))(B))(A)

        K = jit(gram)(X, Y)
        return K.squeeze()

    def _stein_pair(
        self, x: Array, y: Array, s_x: Array, s_y: Array, **hyper: Any
    ) -> Array:
        """
        Computes the pairwise Stein kernel entry between two points.

        Args:
            x:
                A single sample of shape ``(d,)``.
            y:
                A single sample of shape ``(d,)``.
            s_x:
                Score function evaluated at x of shape ``(d,)``.
            s_y:
                Score function evaluated at y of shape ``(d,)``.
            **hyper:
                Additional kernel hyperparameters.

        Returns:
            Array:
                The scalar Stein kernel evaluation at ``x`` and ``y``.
        """
        diff = x - y
        r = jnp.linalg.norm(diff)
        d = x.shape[0]

        phi = self._phi(r, **hyper)
        psi = self._psi(r, **hyper)  # finite at r=0
        phi_pp = self._phi_pp(r, **hyper)

        grad_x = psi * diff  # ∇_x k
        grad_y = -grad_x

        div_xy = -(phi_pp + (d - 1) * psi)  # divergence term

        return (
            div_xy
            + jnp.dot(grad_x, s_y)
            + jnp.dot(grad_y, s_x)
            + phi * jnp.dot(s_x, s_y)
        )

    def stein_matrix(
        self, X: Array, Y: Array, Sx: Array, Sy: Array, **hyper: float
    ) -> Array:
        """
        Computes the Stein kernel Gram matrix between two sets of samples.

        This returns an ``(n,m)`` block matrix, where each entry corresponds to a
        Stein kernel evaluation between points from X and Y with associated scores
        Sx and Sy, using the given kernel and hyperparameters.

        Args:
            X:
                A shape ``(n, d)`` array of samples.
            Y:
                A shape ``(m, d)`` array of samples.
            Sx:
                A shape ``(n, d)`` array of score function evaluations at ``X``.
            Sy:
                A shape ``(m, d)`` array of score function evaluations at ``Y``.
            **hyper:
                Additional kernel hyperparameters.

        Returns:
            Array:
                A shape ``(n,m)`` Gram matrix of Stein kernel evaluations.
        """
        stein_pair = functools.partial(self._stein_pair, **hyper)

        return jax.vmap(
            lambda x, sx: jax.vmap(lambda y, sy: stein_pair(x, y, sx, sy))(Y, Sy)
        )(X, Sx)

    @functools.partial(
        jit,
        static_argnums=(0,),  # self is arg 0 → static
        static_argnames=("lengthscale", "amplitude"),
    )
    def stein_matvec(
        self,
        v: Array,  # shape (n,)
        X: Array,  # shape (n,d)
        Sx: Array,  # shape (n,d)
        *,
        lengthscale: float = 1.,
        amplitude: float = 1.
    ) -> Array:
        """
        Matrix‐free mat‐vec for the Stein kernel:
            (K @ v)[i] = sum_j k_p(x_i, x_j) * v[j]
        without ever forming the full K.
        """
        # bind the hyper‐parameters into the pairwise function
        stein_pair = functools.partial(self._stein_pair, lengthscale=lengthscale, amplitude=amplitude)

        # for a fixed i, compute the i-th row of K dot v:
        def row_dot(x_i, sx_i):
            # shape (n,) of row i
            K_i = vmap(lambda x_j, sx_j: stein_pair(x_i, x_j, sx_i, sx_j))(X, Sx)
            # then dot with v
            return jnp.dot(K_i, v)

        # vectorise over i=0..n-1
        return vmap(row_dot)(X, Sx)

class Matern52Kernel(Kernel):
    name: str = "Matern52"
    display_name: str = "Matérn 5/2"

    @staticmethod
    def _phi(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        The radial profile :math:`\varphi(r)` of the Matérn 5/2 kernel (smoothness parameter
        :math:`\nu = 5/2`).

        This defines the scalar-valued kernel as a function of the Euclidean distance
        :math:`r = \lVert x - y \rVert`, with the following form:

        .. math::

            \varphi(r) = \sigma^{2} \left( 1 + c r + \tfrac{c^{2}}{3} r^{2} \right) e^{-c r},
            \quad c = \frac{\sqrt{5}}{\ell},

        where :math:`\sigma^2` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Pairwise Euclidean distance of shape ``(1,)``.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude:
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The kernel value :math:`\varphi(r)` evaluated at the given distance.
        """

        c = jnp.sqrt(5.0) / lengthscale
        return amplitude * (1.0 + c * r + (c**2 / 3.0) * r**2) * jnp.exp(-c * r)

    @staticmethod
    def _psi(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        Computes the first derivative of the Matérn 5/2 radial profile divided by :math:`r`,
        i.e. :math:`\psi(r) = \varphi'(r)/r`.

        This quantity is finite at :math:`r = 0` and is used in Stein kernel computations.

        .. math::

            \psi(r) = -\,\sigma^{2} \frac{c^{2}}{3} (1 + c r) e^{-c r},
            \quad c = \frac{\sqrt{5}}{\ell},

        where :math:`\sigma^{2}` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude::
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The scalar value :math:`\psi(r) = \varphi'(r) / r` evaluated at the given distance.
        """

        c = jnp.sqrt(5.0) / lengthscale
        return -amplitude * (c**2 / 3.0) * (1 + c * r) * jnp.exp(-c * r)

    @staticmethod
    def _phi_pp(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        Computes the second derivative of the Matérn 5/2 radial profile
        with respect to the distance :math:`r`, i.e. :math:`\varphi''(r)`.

        This quantity appears in Stein kernel constructions involving second-order terms.

        .. math::

            \varphi''(r) = -\,\sigma^{2} \frac{c^{2}}{3} \left(1 + c r - c^{2} r^{2} \right) e^{-c r},
            \quad c = \frac{\sqrt{5}}{\ell},

        where :math:`\sigma^{2}` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude:
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The scalar value :math:`\varphi''(r)` evaluated at the given distance.
        """
        c = jnp.sqrt(5.0) / lengthscale
        return (
            -amplitude * (c**2 / 3.0) * (1.0 + c * r - (c * r) ** 2) * jnp.exp(-c * r)
        )

    @functools.partial(
        jit,
        static_argnums=(0,),  # self is arg 0 → static
        static_argnames=("lengthscale", "amplitude"),
    )
    def stein_matrix(
        self,
        X: Array,
        Y: Array,
        Sx: Array,
        Sy: Array,
        *,
        lengthscale: float = 1.0,
        amplitude: float = 1.0
    ) -> Array:
        stein_pair = functools.partial(
            self._stein_pair, lengthscale=lengthscale, amplitude=amplitude
        )

        return jax.vmap(
            lambda x, sx: jax.vmap(lambda y, sy: stein_pair(x, y, sx, sy))(Y, Sy)
        )(X, Sx)


    @functools.partial(
        jit,
        static_argnums=(0,),  # self is arg 0 → static
        static_argnames=("lengthscale", "amplitude"),
    )
    def stein_matvec(
        self,
        v: Array,  # shape (n,)
        X: Array,  # shape (n,d)
        Sx: Array,  # shape (n,d)
        *,
        lengthscale: float = 1.,
        amplitude: float = 1.
    ) -> Array:
        """
        Matrix‐free mat‐vec for the Stein kernel:
            (K @ v)[i] = sum_j k_p(x_i, x_j) * v[j]
        without ever forming the full K.
        """
        # bind the hyper‐parameters into the pairwise function
        stein_pair = functools.partial(self._stein_pair, lengthscale=lengthscale, amplitude=amplitude)

        # for a fixed i, compute the i-th row of K dot v:
        def row_dot(x_i, sx_i):
            # shape (n,) of row i
            K_i = vmap(lambda x_j, sx_j: stein_pair(x_i, x_j, sx_i, sx_j))(X, Sx)
            # then dot with v
            return jnp.dot(K_i, v)

        # vectorise over i=0..n-1
        return vmap(row_dot)(X, Sx)

class Matern72Kernel(Kernel):
    name: str = "Matern72"
    display_name: str = "Matérn 7/2"

    @staticmethod
    def _phi(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        Computes the radial profile :math:`\varphi(r)` of the Matérn kernel with
        smoothness parameter :math:`\nu = 7/2`.

        This defines the scalar-valued kernel as a function of the Euclidean distance
        :math:`r = \lVert x - y \rVert`, with the following form:

        .. math::

            \varphi(r) = \sigma^{2} \left( 1 + c r + \tfrac{c^{2}}{3} r^{2} + \tfrac{c^{3}}{15} r^{3} \right) e^{-c r},
            \quad c = \frac{\sqrt{7}}{\ell},

        where :math:`\sigma^2` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude:
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The scalar kernel value :math:`\varphi(r)` evaluated at the given distance.
        """
        c = jnp.sqrt(7.0) / lengthscale  # c = √7 / ℓ
        return (
            amplitude
            * (1.0 + c * r + (c**2 / 3.0) * r**2 + (c**3 / 15.0) * r**3)
            * jnp.exp(-c * r)
        )

    @staticmethod
    def _psi(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        Computes the first derivative of the Matérn 7/2 radial profile divided by :math:`r`,
        i.e. :math:`\psi(r) = \varphi'(r) / r`.

        This quantity is finite at :math:`r = 0` and is used in Stein kernel computations.

        .. math::

            \psi(r) = \frac{\sigma^{2} c^{2}}{15} \left( 5 + 2 c r + c^{2} r^{2} \right) e^{-c r},
            \quad c = \frac{\sqrt{7}}{\ell},

        where :math:`\sigma^{2}` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude:
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The scalar value :math:`\psi(r) = \varphi'(r) / r` evaluated at the given distance.
        """

        c = jnp.sqrt(7.0) / lengthscale
        # ϕ′(r)/r = -σ² (c²/15) · (5 + 2 c r + (c r)²) · e^{-c r}
        return (
            -amplitude
            * (c**2 / 15.0)
            * (5.0 + 2.0 * c * r + (c * r) ** 2)
            * jnp.exp(-c * r)
        )

    @staticmethod
    def _phi_pp(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        Computes the second derivative of the Matérn 7/2 radial profile
        with respect to the distance :math:`r`, i.e. :math:`\varphi''(r)`.

        This quantity appears in Stein kernel constructions involving second-order terms.

        .. math::

            \varphi''(r) = \frac{\sigma^{2} c^{2}}{15}
            \left( -5 + c r - c^{2} r^{2} + c^{3} r^{3} \right) e^{-c r},
            \quad c = \frac{\sqrt{7}}{\ell},

        where :math:`\sigma^{2}` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude:
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The scalar value :math:`\varphi''(r)` evaluated at the given distance.
        """

        c = jnp.sqrt(7.0) / lengthscale
        # ϕ″(r) = σ² (c²/15) · (−5 + c r − (c r)² + (c r)³) · e^{-c r}
        return (
            amplitude
            * (c**2 / 15.0)
            * (-5.0 + c * r - (c * r) ** 2 + (c * r) ** 3)
            * jnp.exp(-c * r)
        )

    @functools.partial(
        jit,
        static_argnums=(0,),  # self is arg 0 → static
        static_argnames=("lengthscale", "amplitude"),
    )
    def stein_matrix(
        self,
        X: Array,
        Y: Array,
        Sx: Array,
        Sy: Array,
        *,
        lengthscale: float = 1.0,
        amplitude: float = 1.0,
    ) -> Array:
        stein_pair = functools.partial(
            self._stein_pair, lengthscale=lengthscale, amplitude=amplitude
        )

        return jax.vmap(
            lambda x, sx: jax.vmap(lambda y, sy: stein_pair(x, y, sx, sy))(Y, Sy)
        )(X, Sx)

    @functools.partial(
        jit,
        static_argnums=(0,),  # self is arg 0 → static
        static_argnames=("lengthscale", "amplitude"),
    )
    def stein_matvec(
        self,
        v: Array,  # shape (n,)
        X: Array,  # shape (n,d)
        Sx: Array,  # shape (n,d)
        *,
        lengthscale: float = 1.,
        amplitude: float = 1.
    ) -> Array:
        """
        Matrix‐free mat‐vec for the Stein kernel:
            (K @ v)[i] = sum_j k_p(x_i, x_j) * v[j]
        without ever forming the full K.
        """
        # bind the hyper‐parameters into the pairwise function
        stein_pair = functools.partial(self._stein_pair, lengthscale=lengthscale, amplitude=amplitude)

        # for a fixed i, compute the i-th row of K dot v:
        def row_dot(x_i, sx_i):
            # shape (n,) of row i
            K_i = vmap(lambda x_j, sx_j: stein_pair(x_i, x_j, sx_i, sx_j))(X, Sx)
            # then dot with v
            return jnp.dot(K_i, v)

        # vectorise over i=0..n-1
        return vmap(row_dot)(X, Sx)

class GaussianKernel(Kernel):
    name: str = "Gaussian"
    display_name: str = "Gaussian (RBF)"

    @staticmethod
    def _phi(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        Computes the radial profile :math:`\varphi(r)` of the Gaussian (RBF) kernel.

        This defines the scalar-valued kernel as a function of the Euclidean distance
        :math:`r = \lVert x - y \rVert`, with the following form:

        .. math::

            \varphi(r) = \sigma^2 \exp\left(-\frac{r^2}{2 \ell^2}\right),

        where :math:`\sigma^2` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude::
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The scalar kernel value :math:`\varphi(r)` evaluated at the given distance.
        """
        z = -0.5 * (r / lengthscale) ** 2
        return amplitude * jnp.exp(z)

    @staticmethod
    def _psi(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        Computes the first derivative of the Gaussian (RBF) radial profile divided by :math:`r`,
        i.e. :math:`\psi(r) = \varphi'(r) / r`.

        This quantity is finite at :math:`r = 0` and is used in Stein kernel computations.

        .. math::

            \psi(r) = -\frac{\sigma^2}{\ell^2} \exp\left(-\frac{r^2}{2 \ell^2}\right),

        where :math:`\sigma^2` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude::
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The scalar value :math:`\psi(r) = \varphi'(r) / r` evaluated at the given distance.
        """

        z = -0.5 * (r / lengthscale) ** 2
        base = amplitude * jnp.exp(z)
        return -base / lengthscale**2  # −σ² r / ℓ² · e^{z}

    @staticmethod
    def _phi_pp(r: Array, *, lengthscale: float = 1.0, amplitude: float = 1.0) -> Array:
        r"""
        Computes the second derivative of the Gaussian (RBF) radial profile
        with respect to the distance :math:`r`, i.e. :math:`\varphi''(r)`.

        This quantity appears in Stein kernel constructions involving second-order terms.

        .. math::

            \varphi''(r) = \frac{\sigma^2}{\ell^2}
            \left( \frac{r^2}{\ell^2} - 1 \right)
            \exp\left(-\frac{r^2}{2 \ell^2}\right),

        where :math:`\sigma^2` is the amplitude and :math:`\ell` is the lengthscale.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell>0`.
            amplitude::
                Amplitude parameter :math:`\sigma^2>0`.

        Returns:
            Array:
                The scalar value :math:`\varphi''(r)` evaluated at the given distance.
        """
        z = -0.5 * (r / lengthscale) ** 2
        base = amplitude * jnp.exp(z) / lengthscale**2
        return base * ((r / lengthscale) ** 2 - 1)  # σ²/ℓ² · ((r/ℓ)² − 1) · e^{z}

    @functools.partial(
        jit,
        static_argnums=(0,),  # self is arg 0 → static
        static_argnames=("lengthscale", "amplitude"),
    )
    def stein_matrix(
        self,
        X: Array,
        Y: Array,
        Sx: Array,
        Sy: Array,
        *,
        lengthscale: float = 1.0,
        amplitude: float = 1.0,
    ) -> Array:
        stein_pair = functools.partial(
            self._stein_pair, lengthscale=lengthscale, amplitude=amplitude
        )

        return jax.vmap(
            lambda x, sx: jax.vmap(lambda y, sy: stein_pair(x, y, sx, sy))(Y, Sy)
        )(X, Sx)
    
    @functools.partial(
        jit,
        static_argnums=(0,),  # self is arg 0 → static
        static_argnames=("lengthscale", "amplitude"),
    )
    def stein_matvec(
        self,
        v: Array,  # shape (n,)
        X: Array,  # shape (n,d)
        Sx: Array,  # shape (n,d)
        *,
        lengthscale: float = 1.,
        amplitude: float = 1.
    ) -> Array:
        """
        Matrix‐free mat‐vec for the Stein kernel:
            (K @ v)[i] = sum_j k_p(x_i, x_j) * v[j]
        without ever forming the full K.
        """
        # bind the hyper‐parameters into the pairwise function
        stein_pair = functools.partial(self._stein_pair, lengthscale=lengthscale, amplitude=amplitude)

        # for a fixed i, compute the i-th row of K dot v:
        def row_dot(x_i, sx_i):
            # shape (n,) of row i
            K_i = vmap(lambda x_j, sx_j: stein_pair(x_i, x_j, sx_i, sx_j))(X, Sx)
            # then dot with v
            return jnp.dot(K_i, v)

        # vectorise over i=0..n-1
        return vmap(row_dot)(X, Sx)

class IMQKernel(Kernel):
    name: str = "IMQ"
    display_name: str = "Inverse Multiquadric (IMQ)"

    @staticmethod
    def _phi(
        r: Array,
        *,
        lengthscale: float = 1.0,
        amplitude: float = 1.0,
        gamma: float = 1.0,
        beta: float = 0.5,
    ) -> Array:
        r"""
        Computes the radial profile :math:`\varphi(r)` of the Inverse Multiquadric (IMQ) kernel.

        This defines a scalar-valued kernel as a function of the Euclidean distance
        :math:`r = \lVert x - y \rVert`, using the form:

        .. math::

            \varphi(r) = \sigma^{2} \, u(r)^{-\beta},
            \quad u(r) = \gamma^{2} + \frac{r^{2}}{\ell^{2}},

        where :math:`\sigma^{2}` is the amplitude, :math:`\ell` is the lengthscale,
        and :math:`\gamma`, :math:`\beta` are additional hyperparameters.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell > 0`.
            amplitude::
                Amplitude parameter :math:`\sigma^{2} > 0`.
            gamma (float, optional):
                Offset parameter :math:`\gamma > 0`. Controls the flatness of the kernel.
            beta (float, optional):
                Exponent parameter :math:`\beta \in (0, 1)`. Controls the decay rate. Defaults to 0.5.

        Returns:
            Array:
                The scalar kernel value :math:`\varphi(r)` evaluated at the given distance.
        """
        u = gamma**2 + (r / lengthscale) ** 2
        return amplitude * u ** (-beta)

    @staticmethod
    def _psi(
        r: Array,
        *,
        lengthscale: float = 1.0,
        amplitude: float = 1.0,
        gamma: float = 1.0,
        beta: float = 0.5,
    ) -> Array:
        r"""
        Computes the first derivative of the Inverse Multiquadric (IMQ) radial profile
        divided by :math:`r`, i.e. :math:`\psi(r) = \varphi'(r) / r`.

        This quantity is finite at :math:`r = 0` and is used in Stein kernel computations.

        The IMQ kernel is defined as a function of the Euclidean distance
        :math:`r = \lVert x - y \rVert` by:

        .. math::

            \psi(r) = -\frac{2 \beta \sigma^{2}}{\ell^{2}} \, u(r)^{-(\beta + 1)},
            \quad u(r) = \gamma^{2} + \frac{r^{2}}{\ell^{2}},

        where :math:`\sigma^{2}` is the amplitude, :math:`\ell` is the lengthscale,
        and :math:`\gamma`, :math:`\beta` are additional hyperparameters.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell > 0`.
            amplitude::
                Amplitude parameter :math:`\sigma^{2} > 0`.
            gamma:
                Offset parameter :math:`\gamma > 0`. Controls the flatness of the kernel.
            beta:
                Exponent parameter :math:`\beta \in (0, 1)`. Controls the decay rate.

        Returns:
            Array:
                The scalar value :math:`\psi(r) = \varphi'(r) / r` evaluated at the given distance.
        """
        u = gamma**2 + (r / lengthscale) ** 2
        return -2.0 * beta * amplitude * 1 / lengthscale**2 * u ** (-(beta + 1.0))

    @staticmethod
    def _phi_pp(
        r: Array,
        *,
        lengthscale: float = 1.0,
        amplitude: float = 1.0,
        gamma: float = 1.0,
        beta: float = 0.5,
    ) -> Array:
        r"""
        Computes the second derivative of the Inverse Multiquadric (IMQ) radial profile
        with respect to the distance :math:`r`, i.e. :math:`\varphi''(r)`.

        This quantity is used in Stein kernel constructions involving second-order terms.

        The IMQ kernel is defined in terms of the squared offset function
        :math:`u(r) = \gamma^2 + r^2 / \ell^2`:

        .. math::

            \varphi''(r) = \sigma^{2} \, u(r)^{-(\beta + 2)}
            \left[ -\frac{2\beta}{\ell^{2}} u(r) + \frac{4\beta(\beta + 1) r^{2}}{\ell^{4}} \right],
            \quad u(r) = \gamma^{2} + \frac{r^{2}}{\ell^{2}},

        where :math:`\sigma^{2}` is the amplitude, :math:`\ell` is the lengthscale,
        and :math:`\gamma`, :math:`\beta` are additional hyperparameters.

        Args:
            r:
                Euclidean distance of shape ``(1,)``. Must represent a single scalar value.
            lengthscale:
                Lengthscale parameter :math:`\ell > 0`.
            amplitude::
                Amplitude parameter :math:`\sigma^{2} > 0`.
            gamma:
                Offset parameter :math:`\gamma > 0`. Controls the flatness of the kernel.
            beta:
                Exponent parameter :math:`\beta \in (0, 1)`. Controls the decay rate. Defaults to 0.5.

        Returns:
            Array:
                The scalar value :math:`\varphi''(r)` evaluated at the given distance.
        """

        u = gamma**2 + (r / lengthscale) ** 2
        return (
            amplitude
            * u ** (-(beta + 2.0))
            * (
                -2.0 * beta * u / lengthscale**2
                + 4.0 * beta * (beta + 1.0) * r**2 / lengthscale**4
            )
        )

    @functools.partial(
        jit,
        static_argnums=(0,),  # self is arg 0 → static
        static_argnames=("lengthscale", "amplitude", "gamma", "beta"),
    )
    def stein_matrix(
        self,
        X: Array,
        Y: Array,
        Sx: Array,
        Sy: Array,
        *,
        lengthscale: float = 1.0,
        amplitude: float = 1.0,
        gamma: float = 1.0,
        beta: float = 0.5,
    ) -> Array:
        stein_pair = functools.partial(
            self._stein_pair,
            lengthscale=lengthscale,
            amplitude=amplitude,
            gamma=gamma,
            beta=beta,
        )

        return jax.vmap(
            lambda x, sx: jax.vmap(lambda y, sy: stein_pair(x, y, sx, sy))(Y, Sy)
        )(X, Sx)


    @functools.partial(
        jit,
        static_argnums=(0,),  
        static_argnames=("lengthscale", "amplitude", "gamma", "beta"),
    )    
    def stein_matvec(
        self,
        v: Array,  # shape (n,)
        X: Array,  # shape (n,d)
        Sx: Array,  # shape (n,d)
        *,
        lengthscale: float = 1.,
        amplitude: float = 1.,
        gamma: float = 1.0,
        beta: float = 0.5
    ) -> Array:
        """
        Matrix‐free mat‐vec for the Stein kernel:
            (K @ v)[i] = sum_j k_p(x_i, x_j) * v[j]
        without ever forming the full K.
        """
        # bind the hyper‐parameters into the pairwise function
        stein_pair = functools.partial(
            self._stein_pair,
            lengthscale=lengthscale,
            amplitude=amplitude,
            gamma=gamma,
            beta=beta,
        )
        # for a fixed i, compute the i-th row of K dot v:
        def row_dot(x_i, sx_i):
            # shape (n,) of row i
            K_i = vmap(lambda x_j, sx_j: stein_pair(x_i, x_j, sx_i, sx_j))(X, Sx)
            # then dot with v
            return jnp.dot(K_i, v)

        # vectorise over i=0..n-1
        return vmap(row_dot)(X, Sx)