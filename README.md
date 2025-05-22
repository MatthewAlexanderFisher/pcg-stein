# pcg-stein

Implements preconditioned conjugate gradient (PCG) methods to solve the system

$$ K x = y, $$

where $K$ is a symmetric PSD gram matrix produced by a Stein kernel.

[![Docs](https://img.shields.io/badge/docs-pcg--stein-blue)](https://matthewalexanderfisher.github.io/pcg-stein)


---

## Preconditioners

### 


---

## Stein Kernels

### Kernel Derivatives

For a probability density $p$ and base kernel $k$, the Langevin Stein kernel takes the form:

$$ k_p(x,y) = \nabla_x \cdot \nabla_y k(x,y) + \nabla_x k(x,y) \cdot \nabla_y \log p(y) + \nabla_y k(x,y) \cdot \nabla_x \log p(x) + k(x,y) \nabla_x \log p(x) \cdot \nabla_y \log p(y). $$

An issue with performing automatic differentiation to compute the divergence and gradient of the base kernel, is the derivative at $x=y$. For isotropic base kernels $k \in C^2$, there is a removable singularity. However, automatic differentiation is unable to handle this cleanly and robustly. 

To address this, we restrict our implementation to Langevin stein kernels for isotropic base kernels $k$, where the first and second derivatives are provided with the singularity removed. An *isotropic* kernel is a kernel $k:\mathbb{R}^d\times \mathbb{R}^d \rightarrow \mathbb{R}$ of the form

$$
k(x,y)=\varphi\bigl(r\bigr) \qquad r = \lVert x-y\rVert.  
$$

We can compute its gradient and divergence in closed form:

#### The gradients $\nabla_x k(x,y)$ and $\nabla_y k(x,y)$
   Let

   $$
     u = x - y,\quad r = \lVert u \rVert \quad\text{so}\quad \nabla_x r = \frac{u}{r}.
   $$

   Then by the chain‐rule

   $$
     \nabla_x k(x,y)= \varphi'(r) \nabla_x r = \varphi'(r) \frac{x - y}{\lVert x-y\rVert}.
   $$

   Similarly $\nabla_y k(x,y) = -\nabla_x k(x,y)$.

#### The mixed divergence $\nabla_x\cdot\nabla_y k(x,y)$
   We seek

   $$
     \nabla_x \cdot \nabla_y k(x,y) =  \nabla_x\cdot\Bigl[-\varphi'(r) \frac{u}{r}\Bigr] = -\nabla_x \cdot\Bigl[\underbrace{\tfrac{\varphi'(r)}{r}}_{g(r)}u\Bigr].
   $$

   Now in $d$ dimensions one has

   $$
     \nabla_x \cdot\bigl[g(r) u\bigr] = g(r)\underbrace{\nabla_x\!\cdot u}_{=d} +u\cdot\nabla_x g(r) = dg(r)+g'(r)\frac{u\cdot u}{r}
     = d \frac{\varphi'(r)}{r} + \bigl(\tfrac{\varphi'(r)}{r}\bigr)'r.
   $$

   But

   $$
     \Bigl(\frac{\varphi'(r)}{r}\Bigr)' = \frac{\varphi''(r) r - \varphi'(r)}{r^2} \quad\rightarrow\quad r \Bigl(\frac{\varphi'(r)}{r}\Bigr)' = \varphi''(r) - \frac{\varphi'(r)}{r}.
   $$

   Hence

   $$
     \nabla_x \cdot\nabla_y k(x,y) = -\Bigl[d \frac{\varphi'(r)}{r} +\bigl(\varphi''(r)-\tfrac{\varphi'(r)}{r}\bigr)\Bigr]= -\Bigl[\varphi''(r) + (d-1)\frac{\varphi'(r)}{r}\Bigr].
   $$

Thus, our final required derivatives to construct the Stein kernel are:

$$
\boxed{
\begin{aligned}
\nabla_x k(x,y) 
&= \varphi'\bigl(\lVert x-y\rVert\bigr) \frac{x-y}{\lVert x-y\rVert} = \frac{\varphi'(r)}{r}(x-y),\\
\nabla_y k(x,y) 
&= -\varphi'\bigl(\lVert x-y\rVert\bigr) \frac{x-y}{\lVert x-y\rVert} = -\frac{\varphi'(r)}{r}(x-y),\\
\nabla_x \cdot\nabla_y k(x,y)
&= -\Bigl[\varphi''\bigl(\lVert x-y\rVert\bigr)
         + (d-1)\frac{\varphi'\bigl(\lVert x-y\rVert\bigr)}{\lVert x-y\rVert}\Bigr] = -\Bigl[\varphi''\bigl(r \bigr)
         + (d-1)\frac{\varphi'\bigl(r\bigr)}{r}\Bigr] .
\end{aligned}
}
$$

### Kernel implementations — notation and conventions

To implement a Stein kernel we need, besides $\varphi$ itself, its first
derivative divided by $r$ (to remove the singularity) and its second derivative:

$$
\psi(r) := \frac{\varphi'(r)}{r}=\frac{1}{r}\frac{\partial\varphi}{\partial r},\qquad \varphi''(r)=\frac{\partial^{2}\varphi}{\partial r^{2}}.
$$

Each kernel class provides internal scalar-valued radial functions, used in batched Stein computations:

```python
_phi(r, **hyper)      # profile            ϕ(r)
_psi(r, **hyper)      # derivative / r     ϕ′ / r   (finite at 0)
_phi_pp(r, **hyper)   # second derivative  ϕ″(r)  = d²ϕ/dr²
```

Here, `hyper` are the hyperparameter keyword arguments. All kernels **always** at least include the two scalar hyperparameters:

1. **Lengthscale ($\ell > 0$):** Keyword `lengthscale=`. The $\ell$ is shared across all coordinates.
2. **Amplitude ($\sigma^2 > 0$):** Keyword `amplitude=`. 

For an isotropic base kernel $k$, the lengthscale and amplitude are defined as follows:

$$
k(x,y; \ell,\sigma^{2}) = \sigma^{2} \varphi \bigl(r/\ell\bigr),\qquad r=\lVert x-y\rVert.
$$

The following kernels are currently implemented:

| **Kernel class** | **Radial Profile** $\varphi(r)$                                                                             | **Extra hyperparameters**                          |
| ---------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `Matern52Kernel` | $\sigma^2 \left( 1 + cr + \tfrac{c^2}{3} r^2 \right) e^{-cr},\quad c = \tfrac{\sqrt{5}}{\ell}$                       | *(none)* — fixed smoothness $\nu = \tfrac{5}{2}$   |
| `Matern72Kernel` | $\sigma^2 \left( 1 + cr + \tfrac{c^2}{3} r^2 + \tfrac{c^3}{15} r^3 \right) e^{-cr},\quad c = \tfrac{\sqrt{7}}{\ell}$ | *(none)* — fixed smoothness $\nu = \tfrac{7}{2}$   |
| `GaussianKernel` | $\sigma^2 \exp\left(-\tfrac{r^2}{2\ell^2}\right)$                                                           | *(none)*                                           |
| `IMQKernel`      | $\sigma^2 \left( \gamma^2 + \tfrac{r^2}{\ell^2} \right)^{-\beta}$                                           | `gamma` → $\gamma > 0$, `beta` → $\beta \in (0,1)$ |


---

#### Matérn–5/2 kernel (`Matern52Kernel`)

The Matérn family is indexed by the smoothness parameter $\nu$.
For $\nu=\tfrac52$ the profile is

$$
\varphi(r) = \sigma^{2}\Bigl( 1 + c r + \tfrac{c^{2}}{3} r^{2} \Bigr) e^{-c r},
\quad
c=\frac{\sqrt5}{\ell}, \quad  r=\lVert x-y\rVert .
$$
Thus, the required terms are:
$$
\boxed{
\begin{aligned}
\varphi(r)   &= \sigma^{2}\Bigl(1 + c r + \tfrac{c^{2}}{3}r^{2}\Bigr) e^{-c r}, \\
\psi(r) = \frac{\varphi'(r)}{r}  &= - \sigma^{2} \frac{c^{2}}{3}\Bigl(1 + c r\Bigr) e^{-c r}, \\
\varphi''(r) &= - \sigma^{2} \frac{c^{2}}{3}\Bigl(1 + c r - c^{2} r^{2}\Bigr) e^{-c r}.
\end{aligned}
}
$$


---

#### Matérn–7/2 kernel (`Matern72Kernel`)

For the smoothness parameter $\nu=\tfrac72$ the Matérn profile contains a degree-3 polynomial:

$$
\varphi(r)
=\sigma^2 \Bigl(1+c r+\tfrac{c^{2}}{3}r^{2}+\tfrac{c^{3}}{15}r^{3}\Bigr) e^{-c r},
\quad
c=\frac{\sqrt7}{\ell}, \quad  r=\lVert x-y\rVert .
$$

Thus, the required terms are:

$$
\boxed{
\begin{aligned}
\varphi(r)
  &=\sigma^{2}\Bigl(1 + c r + \tfrac{c^{2}}{3}r^{2}+\tfrac{c^{3}}{15}r^{3}\Bigr)e^{-c r}, \\
\psi(r) = \frac{\varphi'(r)}{r}
  &=-\frac{\sigma^{2}c^{2}}{15}
      \Bigl(5+2c r+c^{2}r^{2}\Bigr)e^{-c r}, \\
\varphi''(r)
  &=\frac{\sigma^{2}c^{2}}{15}
      \Bigl(-5+c r-c^{2}r^{2}+c^{3}r^{3}\Bigr)e^{-c r}.
\end{aligned}}
$$

---

#### Gaussian Kernel (`GaussianKernel`)

The profile for the Gaussian kernel takes the form:

$$
\varphi(r)
=\sigma^2 \exp\left(-\frac{r^2}{2\ell^2}\right), \quad r=\lVert x-y\rVert.
$$

Thus, the required terms are:

$$
\boxed{
\begin{aligned}
\varphi(r)
  &=\sigma^2 \exp\left(-\frac{r^2}{2\ell^2}\right), \\ 
\psi(r) = \frac{\varphi'(r)}{r}
  &= -\frac{\sigma^2}{\ell^2} \exp\left(-\frac{r^2}{2\ell^2}\right) , \\
\varphi''(r)
  &= \frac{\sigma^2}{\ell^2}\left(\frac{r^2}{\ell^2} - 1 \right)\exp\left(-\frac{r^2}{2\ell^2}\right) .
\end{aligned}}
$$

---
#### Inverse-Multiquadric kernel (`IMQKernel`)

The inverse-multiquadric (IMQ) family augments the usual length-scale
and amplitude with two **shape parameters**

| symbol   | constraint  | Python keyword |
| -------- | ----------- | -------------- |
| $\gamma$ | $\gamma>0$  | `gamma=`       |
| $\beta$  | $0<\beta<1$ | `beta=`        |

Its radial profile is

$$
\varphi(r) = \sigma^{2}\Bigl(\gamma^{2}+\tfrac{r^{2}}{\ell^{2}}\Bigr)^{-\beta}, \qquad r=\lVert x-y\rVert.
$$

Writing $u(r)=\gamma^{2}+\tfrac{r^{2}}{\ell^{2}}$, the required terms to evaluate the Stein kernel are:

$$
\boxed{
\begin{aligned}
\varphi(r)
&=\sigma^{2}u(r)^{-\beta}, \\
\psi(r) = \frac{\varphi'(r)}{r}
&=- \frac{2\beta\sigma^{2}}{\ell^{2}} u(r)^{-(\beta+1)}, \\
\varphi''(r)
&=\sigma^{2} u(r)^{-(\beta+2)}
  \Bigl[
     -\frac{2\beta}{\ell^{2}} u(r)
     +
     \frac{4\beta(\beta+1) r^{2}}{\ell^{4}}
  \Bigr].
\end{aligned}}
$$
