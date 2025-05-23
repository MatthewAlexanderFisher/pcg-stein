import jax, jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Any

from pcg_stein.pcg import pcg
from pcg_stein.distribution import Distribution
from pcg_stein.kernel import Kernel
from pcg_stein.precon import Preconditioner


def est_err(w, K):
    """
    Worst-case error of Stein integration.
    """
    qdr = jnp.dot(jnp.dot(K, w), w)
    denominator = jnp.dot(jnp.ones(jnp.shape(w)[0]), w)
    return jnp.sqrt(qdr) / denominator


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


def plot_heatmap_data(
    all_heatmap_data, preconditioners=None, grid_shape=(3, 3), figsize=(12, 10)
):
    """
    Plots heatmaps for each preconditioner using averaged data and standard errors.

    Parameters:
      all_heatmap_data : dict
          A dictionary containing keys 'data', 'xticklabels', and 'yticklabels'.
          - all_heatmap_data['data'] should be a dict mapping each preconditioner name
            to an array-like of shape (num_experiments, ...), where the remaining dimensions
            form the heatmap.
          - Similarly, 'xticklabels' and 'yticklabels' should contain labels for each preconditioner.
      preconditioners : list of str, optional
          List of preconditioner names to plot. If None, a default list is used.
      grid_shape : tuple of int, optional
          The (rows, columns) of the subplot grid.
      figsize : tuple of int, optional
          Size of the overall figure.

    Returns:
      fig : matplotlib.figure.Figure
          The generated figure.
      axes : array of Axes
          The axes array.
    """
    # Set a default list of preconditioners if none is provided.
    if preconditioners is None:
        preconditioners = [
            "Block Jacobi",
            "Nystrom",
            "Nystrom (diagonal sampling)",
            "FITC",
            "Nystrom (random projection)",
            "Randomized SVD",
        ]

    # Calculate averaged data and standard errors for each preconditioner.
    averaged_data = {}
    standard_errors = {}
    for pre in preconditioners:
        # Convert the collected data (assumed to be a list or array of heatmap data)
        # to a numpy array.
        all_data = np.array(all_heatmap_data["data"][pre])
        # Replace NaNs/infs with zeros.
        all_data = np.nan_to_num(all_data, nan=0.0, posinf=0.0, neginf=0.0)
        averaged_data[pre] = np.mean(all_data, axis=0)
        standard_errors[pre] = np.std(all_data, axis=0) / np.sqrt(all_data.shape[0])

    # Determine the global color scale.
    max_abs_value = max(np.abs(averaged_data[pre]).max() for pre in preconditioners)
    vmin = -max_abs_value
    vmax = max_abs_value

    # Create the grid of subplots.
    fig, axes = plt.subplots(*grid_shape, figsize=figsize)
    axes = axes.flatten()

    # Loop over preconditioners and plot each heatmap.
    for i, pre in enumerate(preconditioners):
        ax = axes[i]
        # Use Seaborn's heatmap. Annotate with the standard error.
        sns.heatmap(
            averaged_data[pre],
            xticklabels=all_heatmap_data["xticklabels"][pre],
            yticklabels=all_heatmap_data["yticklabels"][pre],
            ax=ax,
            cmap="RdBu",
            annot=standard_errors[pre],
            cbar=False,
            vmin=vmin,
            vmax=vmax,
        )
        # Optionally adjust the name for display.
        title_str = pre
        if pre == "Randomized SVD":
            title_str = "Randomised SVD"
        ax.set_title(title_str, fontsize=20)
        ax.invert_yaxis()
        ax.set_xlabel(r"$\log_{10} \ell$", fontsize=15)
        if pre == "Block Jacobi":
            ax.set_ylabel(r"$b$", fontsize=15)
        elif pre == "Spectral":
            ax.set_ylabel(r"$r$", fontsize=15)
        else:
            ax.set_ylabel(r"$\log_{10} \eta$", fontsize=15)

    # If there are extra axes in the grid (unused subplots), turn them off.
    for j in range(len(preconditioners), len(axes)):
        axes[j].axis("off")

    # Add a single colorbar for the entire figure.
    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.3])  # Adjust as needed for your layout.
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.text(
        0.5,
        1.05,
        "Gain",
        ha="center",
        va="center",
        fontsize=15,
        transform=cbar.ax.transAxes,
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
    return fig, axes
