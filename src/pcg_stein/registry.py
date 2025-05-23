from pcg_stein.precon import (
    Nystrom,
    NystromRandom,
    NystromDiagonal,
    RandomisedSVD,
    FITC,
    BlockJacobi,
)

from pcg_stein.kernel import Matern52Kernel, Matern72Kernel, GaussianKernel, IMQKernel

from pcg_stein.distribution import BayesianLogisticRegression

PRECON_REGISTRY = {
    cls.name: cls
    for cls in [
        Nystrom,
        NystromRandom,
        NystromDiagonal,
        RandomisedSVD,
        FITC,
        BlockJacobi,
    ]
}

KERNEL_REGISTRY = {
    cls.name: cls for cls in [Matern52Kernel, Matern72Kernel, GaussianKernel, IMQKernel]
}

DISTRIBUTION_REGISTRY = {cls.name: cls for cls in [BayesianLogisticRegression]}
