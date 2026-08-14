#!/usr/bin/env python
# Stage 1: the SCF alone. With --save the converged orbitals go to RHF.chk in the
# per-system directory, which mp2.py and the STC driver then load instead of
# repeating the most expensive step.
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

parser = argparse.ArgumentParser(description='SCF for the STC Laplace-MP2 pipeline')
parser.add_argument('xyzfile', type=str)
parser.add_argument('basis', type=str)
parser.add_argument('-charge', type=int, default=0)
parser.add_argument('-verbose', type=int, default=1)
parser.add_argument('-max_cycle', type=int, default=None,
                    help='cap the SCF iteration count; for timing runs 2 is enough, '
                         'and the convergence check is then skipped')
parser.add_argument('--outcore', action='store_true',
                    help='cap max_memory at twice the Rov size so the DF tensor is '
                         'kept on disk, emulating the large-system regime')
parser.add_argument('--save', action='store_true', help='write RHF.chk under data_MP2/<system>/')
options = parser.parse_args()

import time
import pyscf
pyscf.lib.logger.TIMER_LEVEL = 4
import pyscf.gto
import pyscf.df
import pyscf.scf

from stc_cc import utils, utils_pyscf
utils.set_num_threads(nthreads)
utils_pyscf.make_df_eig()

import libstc

basename = os.path.basename(options.xyzfile)
label = basename[:-4] if basename.endswith('.xyz') else basename
directory = libstc.prepared_dir(label, options.basis, options.charge)

mol, atom_naux = libstc.build_mol(options.xyzfile, options.basis, options.charge)

if options.outcore:
    # pyscf takes the in-core branch iff nao_pair*naux*8/1e6 < .9*max_memory, and
    # DF(mol) copies mol.max_memory at construction, so the cap has to be in place
    # before do_scf. Twice Rov is what the STC run itself would have to spare, so
    # this is the memory the SCF would really see at 5000+ orbitals rather than an
    # artificial floor -- and max_memory still sizes the outcore build blocks.
    ncore = utils_pyscf.get_ncore(mol)
    nocc = mol.nelectron // 2 - ncore
    nvir = mol.nao - mol.nelectron // 2
    naux_ri = pyscf.df.addons.make_auxmol(
        mol, pyscf.df.addons.make_auxbasis(mol, mp2fit=True)).nao_nr()
    rov_mb = nocc * nvir * naux_ri * 8 / 1e6
    # Floored at 2 GB: twice Rov is only a few MB for the smallest systems, which
    # starves the SCF itself rather than just spilling the DF. The floor leaves the
    # handful of smallest systems in core, where outcore timings are meaningless
    # anyway; everything from a couple of GB of DF tensor upwards still spills.
    # Never above what SLURM actually granted (mol.max_memory is PYSCF_MAX_MEMORY
    # here): for the largest systems 2*Rov exceeds the whole allocation, and
    # budgeting more than we hold buys an OOM rather than an in-core run.
    mol.max_memory = min(max(2 * rov_mb, 2000), mol.max_memory)

if options.verbose >= 1:
    libstc.print_value('system', options.xyzfile, flush=True)
    libstc.print_value('basis', options.basis, flush=True)
    libstc.print_value('charge', f'{options.charge:d}', flush=True)
    libstc.print_value('nao', f'{mol.nao:d}', flush=True)
    libstc.print_value('nocc', f'{mol.nelectron // 2:d}', flush=True)
    libstc.print_value('save', str(options.save), flush=True)
    libstc.print_value('outcore', str(options.outcore), flush=True)
    libstc.print_value('max_cycle', str(options.max_cycle), flush=True)
    libstc.print_value('max_memory_MB', f'{mol.max_memory:.0f}', flush=True)
    libstc.print_value('directory', directory, flush=True)

if options.save:
    os.makedirs(directory, exist_ok=True)

if options.max_cycle is not None:
    # do_scf runs the SCF itself and takes no max_cycle, so set the class default
    # it will inherit rather than duplicating the chkfile handling here.
    pyscf.scf.hf.SCF.max_cycle = options.max_cycle

t0 = time.perf_counter()
# dir=None means do_scf neither reads nor writes a chkfile, so --save is the only
# thing that makes this run persist.
rhf = utils_pyscf.do_scf(mol, df=True, save_df=False, dir=directory if options.save else None)
# A capped run is a timing run and is expected not to converge; retrying would
# defeat the cap.
if not rhf.converged and options.max_cycle is None:
    rhf.verbose = 4
    rhf.kernel()
libstc.print_time_value('time_scf', time.perf_counter() - t0, flush=True)

libstc.print_value('auxbasis_scf', str(rhf.with_df.auxbasis), flush=True)
libstc.print_value('naux_scf', f'{rhf.with_df.get_naoaux():d}', flush=True)
libstc.print_value('scf_converged', str(bool(rhf.converged)), flush=True)
libstc.print_value('scf_energy', libstc.fe(rhf.e_tot), flush=True)
if options.save:
    libstc.print_value('scf_chkfile', libstc.scf_chk_path(directory), flush=True)
