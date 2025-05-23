# Preconditioner and Lengthscale Sweep Experiment

This experiment varies the kernel lengthscale over 5 values and preconditioner parameters over 5 values and compares the iteration counts of CG vs. various PCG preconditioners.

- **Config**: See `config.yaml`
- **Reproduce**: Run `python run.py` - can also `python run_workers -b=4` to do batched computation.
- **Main metric**: $m_CG / m_PCG$ (iteration count ratio)
- **Output**: Results saved in `results/*.csv`

Used in: **Figure 1** of the paper.
