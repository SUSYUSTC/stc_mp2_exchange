# Stochastic tensor contraction for the MP2 exchange energy

Code, data and plotting scripts for *Stochastic Tensor Contraction for Efficient
MP2 Exchange* (J. Sun and G. K.-L. Chan).

The exchange term of Laplace-transformed density-fitted MP2 is the only $O(N^5)$
contraction of the method. This repository implements its evaluation by
stochastic tensor contraction (STC), combined with a hybrid
deterministic--stochastic split and grouped sampling of the auxiliary index, so
that the term costs $O(N^3)$ deterministic setup plus $O(N^2)$ sampling at a
fixed absolute error.

## Layout

    scf.py                        Hartree--Fock, cached to data_MP2/<label>/RHF.chk
    mp2.py                        PySCF DF-MP2 reference energies (optional)
    laplace_mp2_exchange_stc.py   the STC-MP2 driver
    stc_repeat.py                 repeated sampling on fixed deterministic input
    libstc.py                     Laplace transform, screening, guides, estimators
    libquad.py                    minimax Laplace quadrature (Remez)
    stc_cc/                       the parts of the STC library this work uses
    data/                         Hackbusch minimax tables (see data/README.md)
    lattice/                      BN sheet and n-alkane geometries
    molecules/                    the five large molecules
    figures_and_results/          paper data files and plotting scripts

`stc_cc/` is an extract of the STC library of the companion paper, containing
only what the MP2 code calls: the alias samplers (`sample`, `alias_numba`),
tensor helpers (`la`, `utils`), the PySCF interface (`utils_pyscf`) and the
orbital localizers (`orbopt`).

## Running

    pip install numpy scipy torch numba pyscf psutil

Run the scripts from the repository root: `stc_cc` is imported as a local
package, and the cached SCF and reference files are written to `data_MP2/`
relative to the working directory.

    python scf.py lattice/BN_2x2.xyz cc-pvtz --save
    python mp2.py lattice/BN_2x2.xyz cc-pvtz --save          # reference, optional
    python laplace_mp2_exchange_stc.py lattice/BN_2x2.xyz cc-pvtz 3e-4 \
        -M 8 -screens 0.01 --vir_oao --group_aux -stdp 1.4 -verbose 2

The third positional argument is the requested standard error of the
correlation energy in Hartree. `-screens` is the threshold $\tau$ of the
deterministic--stochastic split, `--group_aux` samples the auxiliary index in
atom-sized groups, and `--vir_oao` spans the virtual space with projected
orthogonalized atomic orbitals. Frozen core is the default. If the DF-MP2
reference file is absent the driver runs as usual and omits the error lines.

Production runs in the paper used `-M 8 -screens 0.01 -stdp 1.4 --vir_oao
--group_aux` at a requested error of 3e-4 Hartree, on eight CPU cores.

## Figures

Each `figures_and_results/main_plot_*.py` reads only the plain-text `data_*`
file beside it and writes a `.png` and a `.pdf`; `figures_and_results/README.md`
documents the columns. The data files were extracted from the run logs of the
calculations described in the paper.

    cd figures_and_results && python main_plot_scaling.py
