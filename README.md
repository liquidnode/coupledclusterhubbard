# Coupled Cluster Theory for the Hubbard model

Application of the augmented lagrangian method (https://arxiv.org/abs/2403.16381) to a coupled cluster description of the Hubbard model. Using this method, the roots of the coupled cluster equations can be found efficiently by a constrained optimization via Lagrange multipliers. The L-BFGS algorithm from pytorch is used to optimize the given min-max problem. This repository uses and contains code from https://github.com/fevangelista/wicked. Here, we consider coupled cluster singles and doubles with the neel groundstate as a single slater determinant reference. This code is run with "python ccsdhubbard.py <N> <U>", where N refers to the cluster size NxN and U is the interaction strength. 

Energies close to "LeBlanc, James PF, et al. Physical Review X 5.4 (2015): 041041." are found:

4x4 cluster at U=4 and half-filling:   -0.827412
6x6 cluster at U=8 and half-filling:   -0.515061
8x8 cluster at U=8 and half-filling:   -0.515140
10x10 cluster at U=8 and half-filling: -0.515134
