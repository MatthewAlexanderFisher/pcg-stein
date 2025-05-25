import jax, jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from typing import Optional, Any

from pcg_stein.pcg import pcg
from pcg_stein.distribution import Distribution
from pcg_stein.kernel import Kernel
from pcg_stein.precon import Preconditioner
from pcg_stein.registry import PRECON_REGISTRY


def est_err(w, K):
    """
    Worst-case error of Stein integration.
    """
    qdr = jnp.dot(jnp.dot(K, w), w)
    denominator = jnp.dot(jnp.ones(jnp.shape(w)[0]), w)
    return jnp.sqrt(qdr) / denominator


def make_main_plot(
    result_df, labels, metric="gain", ncols=3, fig_mul=3, cbar_label="Gain", title=None
):

    df = result_df.copy()
    # pick the non-NA parameter for every row
    df["param"] = np.where(df["nugget"].notna(), df["nugget"], df["block_size"])

    df_mean = df.groupby(["precon", "lengthscale", "param"], as_index=False)[
        metric
    ].mean()

    df_se = df.groupby(["precon", "lengthscale", "param"], as_index=False)[metric].sem()

    precons = sorted(df_mean["precon"].unique())  # 6 of them
    n_rows, n_cols = 2, 3  # grid shape

    # 1) Compute values and global vmin/vmax for colorbar
    all_vals = []

    for pre in precons:
        heat = (
            df_mean[df_mean["precon"] == pre]
            .pivot(index="param", columns="lengthscale", values=metric)
            .values
        )
        all_vals.append(heat)

    all_vals = np.concatenate([h.flatten() for h in all_vals])
    vmin, vmax = all_vals.min(), all_vals.max()
    cmax = max(abs(vmin), abs(vmax))

    # 2) Make subplots
    n = len(precons)
    cols = ncols
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_mul * cols, fig_mul * rows), constrained_layout=False
    )

    for i, pre in enumerate(precons):
        ax = axes.flat[i]
        heat = (
            df_mean[df_mean["precon"] == pre]
            .pivot(index="param", columns="lengthscale", values=metric)
            .sort_index()
        )

        heat_se = (
            df_se[df_se["precon"] == pre]
            .pivot(index="param", columns="lengthscale", values=metric)
            .sort_index()
        )

        se_data = heat_se.values  # for plotting standard error strings
        mean_data = heat.values

        display_name = PRECON_REGISTRY[pre]().display_name

        sns.heatmap(  # draw heatmap
            heat,
            ax=ax,
            cmap="coolwarm_r",
            vmin=-cmax,
            vmax=+cmax,
            cbar=False,  # turn off individual colorbars
            xticklabels=labels["lengthscale_labels"],
            yticklabels=(
                labels["block_labels"]
                if pre == "BlockJacobi"
                else labels["nugget_labels"]
            ),
        )

        thresh = cmax * 2 / 3  # threshold for text color change

        for i in range(se_data.shape[0]):  # draw standard errors
            for j in range(se_data.shape[1]):
                mean_val = mean_data[i, j]  # used to decide color
                se_val = se_data[i, j]
                color = "black" if mean_val < thresh and mean_val > -thresh else "white"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{se_val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                )

        ax.set_title(display_name, fontsize=12)
        ax.set_xlabel(r"$\ln \ell $", fontsize=10)
        ax.set_ylabel(r"$b$" if pre == "BlockJacobi" else r"$\ln \eta$", fontsize=10)
        ax.invert_yaxis()

    # hide any empty axes
    for ax in axes.flat[len(precons) :]:
        ax.axis("off")

    # adjust so there's space for the colorbar
    plt.tight_layout()

    # 3) Add one global colorbar
    # place it on the right [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
    norm = plt.Normalize(vmin=-cmax, vmax=+cmax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=norm)
    sm.set_array([])  # only needed for the colorbar

    cbar = fig.colorbar(sm, cax=cbar_ax, label=cbar_label)
    cbar.ax.yaxis.label.set_size(12)  # or whatever size you prefer

    plt.subplots_adjust(right=0.9, top=0.95, bottom=0.1)

    # create title
    if title is not None:
        fig.subplots_adjust(top=0.88)
        fig.suptitle(r"\textbf{" + title + "}", fontsize=14)
        line = Line2D(
            [0.06, 0.94],
            [0.935, 0.935],
            transform=fig.transFigure,
            color="black",
            linewidth=0.3,
        )
        fig.add_artist(line)

    return fig, axes


class PCG_Experiment:

    def __init__(
        self,
        dist: Distribution,
        dist_sample: Array,
        kernel: Kernel,
        preconditioners: list[Preconditioner],
    ):
        """
        Initialises the PCG experiment with the target distribution, sample data, kernel, and preconditioners.

        Args:
            dist (Distribution): The target distribution defining the log density and score function.
            dist_sample (Array): A set of samples drawn from the target distribution.
            kernel (Kernel): A positive-definite kernel function used to construct the Stein kernel matrix.
            preconditioners (list[Preconditioner]): A list of preconditioners to be applied.
        """

        self.dist = dist  # posterior
        self.X = dist_sample  # posterior samples
        self.Sx = self.dist.score(self.X)  # posterior scores

        self.n, self.d = self.X.shape

        self.kernel = kernel
        self.preconditioners = preconditioners

        self.wce = None

    def _stein_matrix(self, **kernel_kwargs: Any):
        return self.kernel.stein_matrix(
            self.X, self.X, self.Sx, self.Sx, **kernel_kwargs
        )

    def set_reference_wce(
        self, wce: Optional[float] = None, maxiter: int = 10_000, **hyper: Any
    ):
        """
        Sets reference worst-case error to be used as termination criterion in PCG for the experiment.
        """
        if wce is None:
            stein_matrix = self._stein_matrix(**hyper)
            _, _, _, errs = pcg(
                stein_matrix, b=jnp.ones(self.n), maxiter=maxiter, rtol=0.0, atol=0.0
            )
            self.wce = errs[-1]
        else:
            self.wce = wce

    def __call__(
        self,
        precon_key: Array,
        *,
        wce_mul: float = 1.1,
        maxiter: int = 1_000,
        kernel_kwargs: dict | None = None,
        precon_kwargs: dict | None = None,
        debug_mode: bool = True,
    ):
        """
        Runs a PCG experiment. Compares the close

        Args:
            precon_key: Array
                Used in random preconditioners.
            wce_mul: float
                The number multiplied by the reference worst-case error to act as termination criterion.
            max_iter: int, optional
                Maximum number of iterations of PCG to perform before failure
            kernel_kwargs  : dict, optional
                Extra args forwarded only to kernel.stein_matrix().
            precon_kwargs  : dict, optional
                Extra args forwarded only to each preconditioner.
        Returns:
            A dictionary of gains for each preconditioner. The gain is defined as log(1 + m_CG) - log(1 + m_PCG),
            where m_CG/m_PCG are the number of iterations of CG/PCG required for
            the the worst-case error termination criterion to be fulfilled.
        """

        debug_output = []

        kernel_kwargs = {} if kernel_kwargs is None else kernel_kwargs
        precon_kwargs = {} if precon_kwargs is None else precon_kwargs

        stein_matrix = self._stein_matrix(**kernel_kwargs)

        # compute reference worst-case error if required:
        if self.wce is None:
            self.set_reference_wce(**kernel_kwargs)

        wce_tol = (
            wce_mul * self.wce
        )  # worst-case error tolerance (termination criterion in PCG)

        # define b in Mx = b
        b = jnp.ones(self.n)

        # run CG without preconditioner
        cg_output = pcg(
            stein_matrix, b=b, maxiter=maxiter, rtol=0.0, atol=0.0, wce_tol=wce_tol
        )

        m_cg = cg_output[1]

        if debug_mode is True:
            debug_dict = self.check_failure_conditions(
                cg_output,
                precon=None,
                precon_mat=None,
                maxiter=maxiter,
                wce_tol=wce_tol,
                kernel_kwargs=kernel_kwargs,
                precon_kwargs=None,
            )
            if debug_dict is not None:
                debug_output.append(debug_dict)

        gain_dict = {}

        # run PCG for each preconditioner
        for precon in self.preconditioners:
            precon_key, _ = jax.random.split(precon_key, 2)
            precon_mat = precon(precon_key, stein_matrix, **precon_kwargs)

            pcg_output = pcg(
                stein_matrix,
                b=b,
                maxiter=maxiter,
                rtol=0.0,
                atol=0.0,
                wce_tol=wce_tol,
                M_inv=precon_mat,
            )

            m_pcg = pcg_output[1]
            gain = jnp.log(1 + m_cg) - jnp.log(1 + m_pcg)
            gain_dict[precon.name] = gain

            if debug_mode is True:
                debug_dict = self.check_failure_conditions(
                    pcg_output,
                    precon=precon,
                    precon_mat=precon_mat,
                    maxiter=maxiter,
                    wce_tol=wce_tol,
                    kernel_kwargs=kernel_kwargs,
                    precon_kwargs=precon_kwargs,
                )
                if debug_dict is not None:
                    debug_output.append(debug_dict)

        return gain_dict

    def check_failure_conditions(
        self,
        pcg_output: tuple[Array, ...],
        precon: Optional[Preconditioner] = None,
        precon_mat: Optional[Array] = None,
        precon_key: Optional[Array] = None,
        maxiter: int = 1_000,
        wce_tol: Optional[float] = 0.0,
        kernel_kwargs: dict | None = None,
        precon_kwargs: dict | None = None,
    ):

        x, k, res, wce = pcg_output

        nan_check = jnp.any(jnp.isnan(x))
        termination_check = bool(k == maxiter)

        if precon is None:
            name = "CG"
        else:
            name = precon.name

        if nan_check:
            print(f"nan check failed for {name}")
        if termination_check:
            print(f"Termination check failed for {name} (maxiter {maxiter} met)")

        fail_bool = nan_check or termination_check

        if fail_bool:
            debug_dict = {
                "pcg_output": pcg_output,
                "precon_mat": precon_mat,
                "precon_key": precon_key,
                "kernel_kwargs": kernel_kwargs,
                "precon_kwargs": precon_kwargs,
                "wce_tol": wce_tol,
            }
            return debug_dict
        else:
            return None
