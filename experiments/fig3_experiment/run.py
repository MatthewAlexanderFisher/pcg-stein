#!/usr/bin/env python
from tqdm.auto import tqdm
import os, argparse, math, yaml
from functools import partial
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-b",
        "--cpu",
        type=bool,
        default=True,
        help="Use CPU or GPU boolean (defaults to True: use CPU)",
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default="experiment.yaml",
        help="Path to the YAML configuration file.",
    )
    p.add_argument(
        "-s",
        "--stem",
        type=str,
        default="fig3",
        help="Base name (stem) for the output PDF saved to the results/ directory.",
    )
    return p.parse_args()


args = parse_args()
use_cpu = args.cpu  # Boolean True means use CPU (Default)
config_path = args.config  # Path to experiment config file
stem = args.stem  # Base name for output PDF

if use_cpu:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    # Tell XLA/Eigen to multi-thread on CPU
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intrasession=true"
    os.environ["OMP_NUM_THREADS"] = "16"  # or however many cores you want
    os.environ["MKL_NUM_THREADS"] = "16"


import jax, jax.numpy as jnp
from pcg_stein.registry import PRECON_REGISTRY, KERNEL_REGISTRY, DISTRIBUTION_REGISTRY
from pcg_stein.pcg import pcg

jax.config.update("jax_enable_x64", True)

for device in jax.devices():
    print(device)

# Load config file
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Static constants from config
REPLICATES = cfg["replicates"]  # number of experimental replicates (MCMC samples)
WCE_MUL = cfg["wce_mul"]  # constant multiplying reference worst-case error
N_ITER_WCE_REFERENCE = cfg[
    "n_iter_wce_reference"
]  # number of iterations used in CG to compute reference worst case error
MAXITER = cfg["maxiter"]  # max CG/PCG iterations until termination criterion met
LENGTHSCALES = [
    math.exp(x) for x in map(float, cfg["ln_lengthscales"])
]  # kernel lengthscale values looped over
NUGGETS = [
    math.exp(x) for x in map(float, cfg["ln_nuggets"])
]  # preconditioner nugget values looped over
BLOCK_SIZES = list(
    map(int, cfg["block_sizes"])
)  # block sizes for Block Jacobi (looped over)
M_RANK = int(cfg["m_rank"])  # rank of low-rank approximation

# Kernel
kernel = KERNEL_REGISTRY[cfg["kernel"]]()

# Preconditioner List
precon_list = [PRECON_REGISTRY[name]() for name in cfg["precons"]]

# Distribution
DIST_CLASS = DISTRIBUTION_REGISTRY[cfg["dist"]]  # class of distribution
NUM_SYNTH_SAMPLES = int(cfg["num_synth_samples"])
NUM_MCMC_SAMPLES = int(cfg["num_mcmc_samples"])
TRUE_BETA = jnp.array(list(map(float, cfg["true_beta"])))
SIGMA_PRIOR = float(cfg["sigma_prior"])  # std of prior

dist_synth_key = jax.random.key(int(cfg["dist_synth_key"]))

dist = DIST_CLASS.from_synthetic(
    dist_synth_key, NUM_SYNTH_SAMPLES, TRUE_BETA, sigma_prior=SIGMA_PRIOR
)

# Define PCG used to compute reference WCE:

pcg_ref = partial(pcg, rtol=0.0, atol=0.0, wce_tol=0.0, maxiter=N_ITER_WCE_REFERENCE)

# -------------- Define an experiment over one-replicate ------------


def process_one_rep(args_tuple):
    rep, X_h, Sx_h = args_tuple

    rep_key = jax.random.key(rep)  # key used in this replication

    rows = []  # forms the rows of the eventual dataframe

    lengthscale_pbar = tqdm(LENGTHSCALES, position=1, leave=False)
    for lengthscale in lengthscale_pbar:
        lengthscale_pbar.set_description(f"Lengthscale = {lengthscale:.3f}")

        # define linear system
        K = kernel.stein_matrix(X_h, X_h, Sx_h, Sx_h, lengthscale=lengthscale)
        b = jnp.ones(NUM_MCMC_SAMPLES)  # i.e. shape (X_h.shape[0], )

        # compute reference worst-case error
        _, _, _, wces = pcg_ref(K, b)
        wce_reference = float(wces[-1])  # final worse-case error
        wce_tol = wce_reference * WCE_MUL  # worst-case error tolerance

        # compute CG's m_cg value (no need to rerun CG)
        mask = wces <= wce_tol
        m_cg = int(jnp.argmax(mask)) + 1  # m_cg

        precon_pbar = tqdm(precon_list, position=2, leave=False)
        for precon in precon_pbar:
            precon_pbar.set_description(f"Preconditioner = {precon.name}")

            precon_kwarg_loop = BLOCK_SIZES if precon.name == "BlockJacobi" else NUGGETS

            precon_args_pbar = tqdm(precon_kwarg_loop, position=3, leave=False)
            for precon_arg in precon_args_pbar:
                if precon.name == "BlockJacobi":
                    precon_args_pbar.set_description(f"Block Size = {precon_arg}")
                else:
                    precon_args_pbar.set_description(f"Nugget = {precon_arg:.3f}")

                # key for preconditioner
                rep_key, precon_key = jax.random.split(rep_key, 2)

                # kwargs for preconditioner
                precon_kwargs = (
                    {"nugget": None, "block_size": precon_arg, "m": M_RANK}
                    if precon.name == "BlockJacobi"
                    else {"nugget": precon_arg, "block_size": None, "m": M_RANK}
                )

                # compute preconditioner matrix
                precon_mat = precon(precon_key, K, **precon_kwargs)

                # perform PCG
                x_pcg, m_pcg, res_pcg, wce_pcg = pcg(
                    K,
                    b,
                    rtol=0.0,
                    atol=0.0,
                    wce_tol=wce_tol,
                    maxiter=MAXITER,
                    M_inv=precon_mat,
                )

                rows.append(
                    {
                        "replicate": rep,
                        "lengthscale": lengthscale,
                        "precon": precon.name,
                        **precon_kwargs,
                        "m_cg": float(m_cg),
                        "m_pcg": float(m_pcg),
                    }
                )

    return rows


# --------------- Sample MCMC and format replicate arguments -----------------

mcmc_key = jax.random.key(1)

rep_data = []

mcmc_pbar = tqdm(range(REPLICATES), desc="MCMC", unit="rep")
for rep in mcmc_pbar:
    mcmc_pbar.set_description(f"MCMC replicate {rep}")

    mcmc_key, subkey = jax.random.split(mcmc_key, 2)

    # MCMC sample and its score
    X_rep = dist.sample(subkey, NUM_MCMC_SAMPLES)
    Sx_rep = dist.score(X_rep)

    rep_data.append((rep, X_rep, Sx_rep))

# --------------- Perform PCG vs. CG comparisons -----------------

results = []
for item in tqdm(rep_data, desc="Replicates"):
    results.extend(process_one_rep(item))

# --------------- Save output -----------------

outdir = Path("results")
outdir.mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(outdir / f"{stem}.csv", index=False)
print(f"Saved results/{stem}.csv")
