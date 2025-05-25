# Preconditioner and Lengthscale Sweep Experiment

This experiment varies the kernel lengthscale over 5 values and preconditioner parameters over 5 values and compares the iteration counts of CG vs. various PCG preconditioners using the Gaussian (RBF) Kernel:

$$ k(x,y) = \sigma^2 \exp\left(-\frac{\lVert x-y \rVert^2}{2\ell^2}\right).  $$

- **Config**: See `config.yaml`
- **Reproduce**: Run `python run.py` or `python run_batch.py` to run batched PCG (batches the 5 different preconditioner parameter values). 
- **Main metrics**: Comparisons of $m_{\text{CG}}$ and $m_{{PCG}}$ (gain)
- **Output**: Results saved in `results/*.csv`

Used in: **Figure 4** of the paper.
