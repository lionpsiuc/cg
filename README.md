# Case Studies in High-Performance Computing

## Assignment 3 - Conjugate Gradient Solver

### Mathematical Background

As per the assignment instructions, we are tasked with implementing a Poisson problem on a unit square.

#### The Poisson Problem

We are given the Poisson problem on the unit square $\Omega=(0,1)^2$:

```math
\begin{cases}-\Delta(x)=f(x)&\text{on the interior of }\Omega,\\u(x)=0&\text{on the boundary of }\partial\Omega,\end{cases}
```

where $\Delta u(x)=\frac{\partial^2u}{\partial x_1^2}+\frac{\partial^2u}{\partial x_2^2}$ is the Laplacian operator.

#### Discretisation Approach

##### Grid

We discretise the domain using a uniform grid with $N$ internal points in each direction:

- The grid spacing is given by $h=\frac{1}{N+1}$.
- Our grid points are $(x_1,x_2)=(ih,jh)$ for $i,j=1,2,\ldots,N$.
- The boundary points occur at $i,j=0$ or $i,j=N+1$ in the case that $u=0$.

##### Finite Difference Approximation

For the Laplacian operator, we use the standard five-point stencil approximation:

```math
-\Delta(ih,jh)\approx-\left[\frac{u((i+1)h,jh)+u((i-1)h,jh)+u(ih,(j+1)h)+u(ih,(j-1)h)-4u(ih,jh)}{h^2}\right].
```

Furthermore, we rearrange in order to match the form $Ax=b$:

```math
\frac{4u(ih,jh)-u((i+1)h,jh)-u((i-1)h,jh)-u(ih,(j+1)h)-u(ih,(j-1)h)}{h^2}=f(ih,jh).
```

> Note that we do not show the derivation of this explicitly; this is because it is taken from the High-Performance Computing Software II module. More specifically, please see `notes.pdf` where the derivation is given in full.

##### Ordering of the Grid Points

We use row-major ordering:

- Move left to right, and then top to bottom.
- Our index $k=jN+i$ for a grid point $(ih,jh)$.
