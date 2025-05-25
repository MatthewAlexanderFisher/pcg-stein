from dataclasses import dataclass
import jax, jax.numpy as jnp
from jax import Array
from typing import Tuple, Callable


@dataclass(frozen=True)
class LinearOperator:
    """
    Wraps any ``n→n`` operator so that
      - ``A @ v`` works for v. Shape = ``(n,)`` or ``(n, m)``,
      - ``A.T`` gives you the transpose operator,
      - ``.shape`` tells you ``(n, n)``.
    """
    matvec: Callable[[Array], Array]
    rmatvec: Callable[[Array], Array]
    n: int

    __array_priority__ = 10.0

    def __matmul__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == 1:
            # vector: just a single matvec
            return self.matvec(x)
        elif x.ndim == 2:
            # matrix: apply column-wise
            # this compiles to one fused kernel if matvec is jitted
            return jnp.stack([self.matvec(x[:, i]) for i in range(x.shape[1])], axis=1)
        else:
            raise ValueError(f"LinearOperator only supports 1D or 2D, got ndim={x.ndim}")

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.n)

    @property
    def T(self) -> "LinearOperator":
        # Transpose operator: swaps matvec<->rmatvec
        return LinearOperator(self.rmatvec, self.matvec, self.n)
