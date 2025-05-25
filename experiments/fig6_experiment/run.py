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
        default="fig6",
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
MAXITER = cfg["maxiter"]  # max CG/PCG iterations until termination criterion met
LENGTHSCALES = [
    math.exp(x) for x in map(float, cfg["ln_lengthscales"])
]  # kernel lengthscale values looped over
NUGGETS = [
    math.exp(x) for x in map(float, cfg["ln_nuggets"])
]  # preconditioner nugget values looped over
M_RANK = int(cfg["m_rank"])  # rank of low-rank approximation

# Kernel
kernel = KERNEL_REGISTRY[cfg["kernel"]]()

# Preconditioner List
precon_list = [PRECON_REGISTRY[name]() for name in cfg["precons"]]

# Distribution
DIST_CLASS = DISTRIBUTION_REGISTRY[cfg["dist"]]  # class of distribution
NUM_SYNTH_SAMPLES = int(cfg["num_synth_samples"])
NUM_MCMC_SAMPLES = int(cfg["num_mcmc_samples"])
NUM_MCMC_SAMPLES_GOLD = int(cfg["num_mcmc_samples_gold"])
TRUE_BETA = jnp.array(list(map(float, cfg["true_beta"])))
SIGMA_PRIOR = float(cfg["sigma_prior"])  # std of prior

dist_synth_key = jax.random.key(int(cfg["dist_synth_key"]))

dist = DIST_CLASS.from_synthetic(
    dist_synth_key, NUM_SYNTH_SAMPLES, TRUE_BETA, sigma_prior=SIGMA_PRIOR
)


# -------------- Define an experiment over one-replicate ------------


def process_one_rep(args_tuple):
    rep, X_h, Sx_h, f_evals = args_tuple

    rep_key = jax.random.key(rep)  # key used in this replication

    cg_long_rows = []  # forms the rows of the eventual dataframe
    pcg_long_rows = []

    lengthscale_pbar = tqdm(LENGTHSCALES, position=1, leave=False)
    for lengthscale in lengthscale_pbar:
        lengthscale_pbar.set_description(f"Lengthscale = {lengthscale:.3f}")

        # define linear system
        K = kernel.stein_matrix(X_h, X_h, Sx_h, Sx_h, lengthscale=lengthscale)
        b = jnp.ones(NUM_MCMC_SAMPLES)  # i.e. shape (X_h.shape[0], )

        # perform reference CG
        x_cg, m_cg, res_cg, wce_cg, quads_cg = pcg(
            K, b, rtol=0.0, atol=0.0, wce_tol=0.0, maxiter=MAXITER, f_evals=f_evals
        )

        for i in range(quads_cg.shape[0]):
            for d in range(quads_cg.shape[1]):
                cg_long_rows.append(
                    {
                        "replicate": rep,
                        "lengthscale": lengthscale,
                        "quad_index": i,
                        "dim": d,
                        "value": float(quads_cg[i, d]),
                        "method": "cg",
                    }
                )

        precon_pbar = tqdm(precon_list, position=2, leave=False)
        for precon in precon_pbar:
            precon_pbar.set_description(f"Preconditioner = {precon.name}")

            precon_kwarg_loop = NUGGETS

            precon_args_pbar = tqdm(precon_kwarg_loop, position=3, leave=False)
            for precon_arg in precon_args_pbar:
                precon_args_pbar.set_description(f"Nugget = {precon_arg:.3f}")

                # key for preconditioner
                rep_key, precon_key = jax.random.split(rep_key, 2)

                # kwargs for preconditioner
                precon_kwargs = {"nugget": precon_arg, "m": M_RANK}

                # compute preconditioner matrix
                precon_mat = precon(precon_key, K, **precon_kwargs)

                # perform PCG
                x_pcg, m_pcg, res_pcg, wce_pcg, quads_pcg = pcg(
                    K,
                    b,
                    rtol=0.0,
                    atol=0.0,
                    wce_tol=0.0,
                    maxiter=MAXITER,
                    f_evals=f_evals,
                    M_inv=precon_mat,
                )

                for i in range(quads_pcg.shape[0]):
                    for d in range(quads_pcg.shape[1]):
                        pcg_long_rows.append(
                            {
                                "replicate": rep,
                                "lengthscale": lengthscale,
                                "precon": precon.name,
                                **precon_kwargs,
                                "quad_index": i,
                                "dim": d,
                                "value": float(quads_pcg[i, d]),
                                "method": "pcg",
                            }
                        )

    return cg_long_rows, pcg_long_rows


# --------------- Sample MCMC and format replicate arguments -----------------

mcmc_key = jax.random.key(1)

rep_data = []
monte_carlo_rows = []

mcmc_pbar = tqdm(range(REPLICATES), desc="MCMC", unit="rep")
for rep in mcmc_pbar:
    mcmc_pbar.set_description(f"MCMC replicate {rep}")

    mcmc_key, subkey = jax.random.split(mcmc_key, 2)

    # MCMC sample and its score
    X_rep = dist.sample(subkey, NUM_MCMC_SAMPLES)
    Sx_rep = dist.score(X_rep)

    # Compute the evaluations of f at X_rep.
    f_evals = X_rep  # For this experiment, f is the identity function

    # Compute Monte-Carlo estimate
    rep_data.append((rep, X_rep, Sx_rep, f_evals))

    # monte-carlo approximations
    for d in range(f_evals.shape[1]):
        monte_carlo_rows.append(
            {
                "replicate": rep,
                "dim": d,
                "value": float(f_evals[:, d].mean()),
                "gold_standard": False,
            }
        )

# --------------- Compute gold-standard Monte-Carlo Estimates -----------------
mcmc_key, subkey = jax.random.split(mcmc_key, 2)
X_gold = dist.sample(subkey, NUM_MCMC_SAMPLES_GOLD)

for d in range(X_gold.shape[1]):
    monte_carlo_rows.append(
        {
            "replicate": -1,
            "dim": d,
            "value": float(X_gold[:, d].mean()),
            "gold_standard": True,
        }
    )


# --------------- Perform PCG vs. CG comparisons -----------------

cg_results = []
pcg_results = []
for item in tqdm(rep_data, desc="Replicates", leave=False):
    cg_result, pcg_result = process_one_rep(item)
    cg_results.extend(cg_result)
    pcg_results.extend(pcg_result)

# --------------- Save outputs -----------------

outdir = Path("results")
outdir.mkdir(exist_ok=True)

pd.DataFrame(cg_results).to_csv(outdir / f"{stem}_cg.csv", index=False)
print(f"Saved results/{stem}_cg.csv")

pd.DataFrame(pcg_results).to_csv(outdir / f"{stem}_pcg.csv", index=False)
print(f"Saved results/{stem}_pcg.csv")

pd.DataFrame(monte_carlo_rows).to_csv(outdir / f"{stem}_mc.csv", index=False)
print(f"Saved results/{stem}_mc.csv")
