# Figures and raw data

Self-contained: each `main_plot_*.py` reads only the plain-text `data_*` files in
this directory and writes a `.png`. `config.py` fixes the colour scheme,
`libformat.py` the log-axis tick formatting.

| script | data | figure |
| --- | --- | --- |
| `main_plot_unbiaseness.py` | `data_unbiasedness` | `unbiaseness.png` |
| `main_plot_threshold.py` | `data_threshold` | `threshold.png` |
| `main_plot_scaling.py` | `data_scaling` | `scaling.png` |
| `main_plot_samples.py` | `data_scaling` | `samples.png` |
| `diagrams/plot_diagram.py` | --- | `diagrams/diagram.png` |

## data_unbiasedness
2500 independent STC-MP2 runs on BN_2x2 / cc-pVTZ, frozen core, screening
threshold 1e-2, requested stochastic error 3e-4 $E_h$. Columns: stochastic
exchange energy $K$ and total MP2 correlation energy. The reference value is in
the header comment.

## data_threshold
Screening-threshold sweep on BN_4x4 / cc-pVTZ, 48 threads, requested error
3e-4 $E_h$. For each threshold: the average number of virtuals kept per
occupied, the number of stochastic samples drawn, wall times, the deterministic
part of the exchange energy, the total exchange energy, and the total MP2 error.
`tau = 0.2, 0.5` keep nothing deterministically, i.e. pure stochastic sampling.

## data_scaling
One row per system (BN sheets and linear alkanes, cc-pVTZ, frozen core), STC at
screening threshold 1e-2 and requested error 3e-4 $E_h$. All timings are wall
times on 48 cores of one node. `t_SCF_2it_incore` / `t_SCF_2it_outcore` are two
SCF iterations timed in and out of core; `NA` where the run was not possible in
the given memory. DLPNO-MP2 and RI-MP2 timings are ORCA.

## data_large
STC-MP2 on five large molecules, cc-pVTZ and cc-pVQZ, frozen core, requested error
3e-4 $E_h$, 8 threads on one node.  One row per system and screening threshold.
`t_Rov` is the construction of the fitted three-index tensor, `t_laplace` the
dressing of Eq. 4 summed over the eight quadrature points, `t_direct` the exact
direct term, `t_exchange` the stochastic exchange, and `t_std` the pilot pass that
sizes the sample count.  `t_MP2` is the driver's own total, equal to
`t_laplace + t_direct + t_exchange`; `t_job` is the SLURM elapsed time of the whole
job, which additionally contains `t_Rov`, `t_loc`, `t_warm` and about 40 s of
interpreter start-up.  `Rov_GB` is the size of one dressed tensor.
The valinomycin/cc-pVQZ run started before its PySCF reference had finished, so its
errors were reconstructed afterwards from the reference energies; the reported
values are identical in form to those the driver prints itself.

Geometries: c3gc is the L7 complex (Sedl{\'a}k et al.), taken verbatim from a
published set; valinomycin and vancomycin are MMFF94-optimised conformers generated
from their PubChem SMILES, not literature geometries.
