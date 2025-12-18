# Coarse-Graining Message-Passing Algorithms

This project examines how coarse-graining message-passing algorithms—specifically Thouless–Anderson–Palmer (TAP) and mean-field (MF)—affects our ability to recover those algorithms from simulated data.

## Overview

We run TAP and MF inference on a Boltzmann machine and then coarse-grain the resulting marginal beliefs over nodes at each time point. Coarse-graining is performed either **temporally** or **spatially**, and we analyze how this loss of resolution impacts our ability to infer the underlying message-passing dynamics.

## Experiments

### Temporal Coarse-Graining
- **`coarse_grain_temporal_exact.ipynb`**
- **`coarse_grain_temporal_adam.ipynb`**

In these notebooks, temporal coarse-graining is applied by averaging node marginals across time points while also applying a low-pass filter. We then attempt to recover the original TAP or MF update rules from the temporally smoothed signals.

### Spatial Coarse-Graining
- **`coarse_grain_spatial.ipynb`**

Here, nodes are decimated on a lattice-shaped Boltzmann machine. The renormalization group is used to derive an effective (coarse-grained) energy function, which is then used to study how well TAP and MF dynamics can be recovered after spatial coarse-graining.

## Key Result

Across both temporal and spatial coarse-graining regimes, these experiments show that it is difficult to reliably recover the original TAP and MF algorithms using only coarse-grained data.

## Environment

- **Local**:  
  - Conda environment: `inferring_inference`  
  - Python version: 3.9.23  
  - OS: macOS

- **Cluster**:  
  - Environment: `vnv_coarse`  
  - Cluster: `demiurge`
