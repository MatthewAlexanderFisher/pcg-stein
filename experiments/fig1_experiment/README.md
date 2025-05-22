# Preconditioner and Lengthscale Sweep Experiment

This experiment varies the kernel lengthscale over 5 values and preconditioner parameters over 5 values and compares the iteration counts of CG vs. various PCG preconditioners.

- **Config**: See `config.yml`
- **Reproduce**: Run `python run.py`
- **Main metric**: $m_CG / m_PCG$ (iteration count ratio)
- **Output**: Results saved in `results/*.parquet`

Used in: **Figure 1** of the paper.
