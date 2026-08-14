#!/usr/bin/env python
# Stage 2: the PySCF DF-MP2 reference. The SCF is loaded, never recomputed: the
# chkfile must already exist or this errors out.
import os
import sys

print(*sys.argv[1:])
os.environ['CUDA_HOME'] = ''
os.environ['CUDA_PATH'] = ''
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
this_dir = os.path.dirname(__file__)

if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
    nthreads = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
else:
    nthreads = os.cpu_count()

# One cache per thread count. torchinductor bakes the thread count into the
# generated OpenMP code (#pragma omp parallel num_threads(N)), so a shared cache
# lets a run reuse kernels compiled for a different N -- silently, and fatally for
# any thread-scaling measurement.
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(this_dir, f'cache_{nthreads}')

if 'SLURM_MEM_PER_NODE' in os.environ:
    os.environ['PYSCF_MAX_MEMORY'] = str(int(int(os.environ['SLURM_MEM_PER_NODE']) * 0.9))
else:
    import psutil
    os.environ['PYSCF_MAX_MEMORY'] = str(int(psutil.virtual_memory().available / 1e6 * 0.6))

import argparse

parser = argparse.ArgumentParser(description='PySCF DF-MP2 reference for the STC pipeline')
parser.add_argument('xyzfile', type=str)
parser.add_argument('basis', type=str)
parser.add_argument('-charge', type=int, default=0)
parser.add_argument('--all_electron', action='store_true',
                    help='correlate the core as well; the frozen core is the default')
parser.add_argument('-verbose', type=int, default=1)
parser.add_argument('--save', action='store_true', help='write dfmp2[_fc].txt under data_MP2/<system>/')
options = parser.parse_args()
# Frozen core everywhere by default, so it cannot be forgotten on the command line.
options.frozen_core = not options.all_electron

import time
import pyscf
pyscf.lib.logger.TIMER_LEVEL = 4
import pyscf.gto
import pyscf.df

from stc_cc import utils, utils_pyscf
utils.set_num_threads(nthreads)
utils_pyscf.make_df_eig()

import libstc

# The three reference energies, written as labelled text so they stay greppable
# alongside the run logs. Only this script produces the file.
dfmp2_labels = ['E_mp2_pyscf', 'E_exchange_pyscf', 'E_direct_pyscf']


def save_dfmp2_energies(path, energies):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for label, value in zip(dfmp2_labels, energies):
            f.write(f'{label:28s} {libstc.fe(value)}\n')


basename = os.path.basename(options.xyzfile)
label = basename[:-4] if basename.endswith('.xyz') else basename
directory = libstc.prepared_dir(label, options.basis, options.charge)

# do_scf would silently run the SCF if the chkfile were absent, so check first.
libstc.require_file(
    libstc.scf_chk_path(directory),
    f'python scf.py {options.xyzfile} {options.basis} -charge {options.charge} --save')

mol, atom_naux = libstc.build_mol(options.xyzfile, options.basis, options.charge)
ncore = utils_pyscf.get_ncore(mol) if options.frozen_core else 0

if options.verbose >= 1:
    libstc.print_value('system', options.xyzfile, flush=True)
    libstc.print_value('basis', options.basis, flush=True)
    libstc.print_value('charge', f'{options.charge:d}', flush=True)
    libstc.print_value('frozen_core', str(options.frozen_core), flush=True)
    libstc.print_value('ncore', f'{ncore:d}', flush=True)
    libstc.print_value('save', str(options.save), flush=True)
    libstc.print_value('directory', directory, flush=True)

rhf = libstc.load_scf(mol, directory)
libstc.print_value('scf_energy', libstc.fe(rhf.e_tot), flush=True)

with_df_ri = libstc.make_ri_df(mol)
rhf.with_df = with_df_ri
libstc.print_value('auxbasis_corr', str(with_df_ri.auxbasis), flush=True)

# The reference must use the same auxiliary basis as the STC path, otherwise the
# reported error is a density-fitting difference rather than a sampling error.
# DFMP2 would otherwise inherit the SCF's JK-fitting basis (dfmp2.py reuses
# mf.with_df whenever the mean field has one). The frozen core must match too,
# or a frozen-core STC energy is compared against an all-electron reference.
t0 = time.perf_counter()
pt_pyscf = rhf.DFMP2()
pt_pyscf.with_df = with_df_ri
pt_pyscf.frozen = ncore if options.frozen_core else None
E_mp2_pyscf, _ = pt_pyscf.kernel(with_t2=False)
libstc.print_time_value('time_dfmp2', time.perf_counter() - t0, flush=True)

# PySCF reports same/opposite spin correlation energies. In the notation used
# here, e_corr_os = -J and e_corr_ss = K - J, so K = ss - os and J = -os.
E_exchange_pyscf = pt_pyscf.e_corr_ss - pt_pyscf.e_corr_os
E_direct_pyscf = -pt_pyscf.e_corr_os
energies = [E_mp2_pyscf, E_exchange_pyscf, E_direct_pyscf]

for name, value in zip(dfmp2_labels, energies):
    libstc.print_value(name, libstc.fe(value), flush=True)

if options.save:
    path = libstc.dfmp2_path(directory, options.frozen_core)
    save_dfmp2_energies(path, energies)
    libstc.print_value('dfmp2_file', path, flush=True)
