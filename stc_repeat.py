#!/usr/bin/env python
# Many independent STC exchange estimates for one system at one screening
# threshold, for the sampling-distribution statistics.
#
# Everything deterministic is built once: localization, Rov, the Laplace
# transform, the direct term, the screen and the sampler. Only the sampling is
# repeated. The driver's -nrepeat instead redoes the whole calculation, which for
# BN_4x4 is 2500 x 300 s against 269 s + 2500 x 31 s here.
#
# Each repeat gets its own ScreenResult, so the per-beta accumulation is
# bit-identical to the driver's; the sample count is fixed once per beta so every
# repeat is drawn from the same distribution.
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

parser = argparse.ArgumentParser(description='Repeated STC exchange sampling for one system')
parser.add_argument('xyzfile', type=str)
parser.add_argument('basis', type=str)
parser.add_argument('target_error', type=float)
parser.add_argument('-screen', type=float, required=True)
parser.add_argument('-nrepeat', type=int, default=50)
parser.add_argument('-seed', type=int, default=0, help='torch seed; use the job index so jobs are independent')
parser.add_argument('-out', type=str, default=None, help='write one line per repeat here')
parser.add_argument('-orb', type=str, default='PM', choices=['canonical', 'PM', 'Boys'])
parser.add_argument('-charge', type=int, default=0)
parser.add_argument('-M', type=int, default=8)
parser.add_argument('-stdp', type=float, default=1.4)
parser.add_argument('-std_tol', type=float, default=2e-2)
parser.add_argument('-verbose', type=int, default=1)
parser.add_argument('--all_electron', action='store_true')
parser.add_argument('--vir_oao', action='store_true')
parser.add_argument('--legendre', action='store_true')
parser.add_argument('--group_aux', action='store_true')
options = parser.parse_args()
options.frozen_core = not options.all_electron

import time
import numpy as np
import torch
import pyscf
pyscf.lib.logger.TIMER_LEVEL = 4
import pyscf.gto
import pyscf.df

from stc_cc import utils, utils_pyscf
utils.set_num_threads(nthreads)
utils_pyscf.make_df_eig()

import libstc
libstc.std_tol = options.std_tol


def load_dfmp2_energies(path):
    values = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            values[parts[0]] = float(parts[1])
    return [values[label] for label in ('E_mp2_pyscf', 'E_exchange_pyscf', 'E_direct_pyscf')]


basename = os.path.basename(options.xyzfile)
label = basename[:-4] if basename.endswith('.xyz') else basename
directory = libstc.prepared_dir(label, options.basis, options.charge)
chkfile = libstc.require_file(
    libstc.scf_chk_path(directory),
    f'python scf.py {options.xyzfile} {options.basis} -charge {options.charge} --save')
dfmp2_file = libstc.require_file(
    libstc.dfmp2_path(directory, options.frozen_core),
    f'python mp2.py {options.xyzfile} {options.basis} -charge {options.charge} --save')

mol, atom_naux = libstc.build_mol(options.xyzfile, options.basis, options.charge)
aux_sectors = libstc.make_aux_sectors(atom_naux)

rhf = libstc.load_scf(mol, directory)
with_df_ri = libstc.make_ri_df(mol)
rhf.with_df = with_df_ri
E_mp2_pyscf, E_exchange_pyscf, E_direct_pyscf = load_dfmp2_energies(dfmp2_file)

ncore = utils_pyscf.get_ncore(mol) if options.frozen_core else 0
nao = mol.nao
nocc0 = mol.nelectron // 2
o = slice(ncore, nocc0)
v = slice(nocc0, None)
C = rhf.mo_coeff
Cocc = C[:, o]
Cvir = C[:, v]
eocc = torch.from_numpy(rhf.mo_energy[o].copy())
evir = torch.from_numpy(rhf.mo_energy[v].copy())

t0 = time.perf_counter()
Cocc_local, Cvir_local, Uaux = libstc.get_local_orbitals(
    mol, Cocc, Cvir, options.orb, localize_vir=not options.vir_oao)
if options.vir_oao:
    w, V = np.linalg.eigh(mol.intor('int1e_ovlp'))
    Cvir_local = Cvir @ (Cvir.T @ (V * np.sqrt(w)) @ V.T)
assert Uaux is None, 'auxiliary rotation is incompatible with atom-blocked aux sectors'
Cocc_local = torch.from_numpy(np.ascontiguousarray(Cocc_local))
Cvir_local = torch.from_numpy(np.ascontiguousarray(Cvir_local))
libstc.print_time_value('time_localization', time.perf_counter() - t0, flush=True)

t0 = time.perf_counter()
R = libstc.get_DF_ov_blockwise(with_df_ri, Cocc_local.numpy(), Cvir_local.numpy())
S = mol.intor('int1e_ovlp')
SC = S @ rhf.mo_coeff
F_ao = torch.from_numpy((SC * rhf.mo_energy) @ SC.T)
Focc = Cocc_local.T @ F_ao @ Cocc_local
Fvir = Cvir_local.T @ F_ao @ Cvir_local
libstc.print_time_value('time_DF_transform', time.perf_counter() - t0, flush=True)

driver = libstc.STCLaplaceMP2(R, Focc, Fvir, eocc, evir, verbose=options.verbose,
                              sectors=aux_sectors if options.group_aux else None)
beta_weight = driver.get_beta_list(M=options.M, minimax=not options.legendre)
proxy = driver.beta_std_proxy(options.stdp)
target_variance = options.target_error**2 * proxy / torch.sum(proxy)

if options.verbose >= 1:
    libstc.print_value('system', options.xyzfile, flush=True)
    libstc.print_value('basis', options.basis, flush=True)
    libstc.print_value('screen', libstc.screen_label(options.screen), flush=True)
    libstc.print_value('nrepeat', f'{options.nrepeat:d}', flush=True)
    libstc.print_value('seed', f'{options.seed:d}', flush=True)
    libstc.print_value('target_error', libstc.fe(options.target_error), flush=True)

results = [libstc.ScreenResult(options.screen) for _ in range(options.nrepeat)]
E_direct_lap = torch.zeros((), dtype=driver.R.dtype)
nsamples_beta = []

t_sample = 0.0
t0_all = time.perf_counter()
state = None
for ibeta, ((beta, coeff), t_beta) in enumerate(zip(beta_weight, target_variance)):
    state = None
    state = driver.make_beta_state(beta, coeff)
    E_direct_lap = E_direct_lap + state.direct_mp2()
    state.build_screen(options.screen)
    state.build_sampler(sectors=driver.sectors)
    # estimate_std is itself stochastic, so a per-job seed here would give each job
    # a slightly different sample count and the pooled histogram would mix
    # distributions. Seed the pilot identically everywhere, then switch to the
    # per-job stream for production so the draws stay independent across jobs.
    torch.manual_seed(9000 + ibeta)
    nsamples = libstc.allocate_from_targets(state.estimate_std(), t_beta)
    nsamples_beta.append(int(nsamples))
    torch.manual_seed(options.seed * 100003 + ibeta)
    t0 = time.perf_counter()
    for irep in range(options.nrepeat):
        state.STC_exchange(nsamples)
        results[irep].add(state)
    t_sample += time.perf_counter() - t0
state = None

E_direct_lap = E_direct_lap.item()
libstc.print_value('nsamples_per_beta', ' '.join(str(n) for n in nsamples_beta), flush=True)
libstc.print_time_value('time_sampling_total', t_sample, flush=True)
libstc.print_time_value('time_total', time.perf_counter() - t0_all, flush=True)

# E_mp2 = E_exchange - 2 E_direct, as in the driver
energies = []
for res in results:
    E_exchange = res.energy.item()
    energies.append((E_exchange, E_exchange - 2.0 * E_direct_lap, res.std.item()))

E_arr = np.array([e[1] for e in energies])
libstc.print_value('mean_MP2_energy', libstc.fe(float(np.mean(E_arr))), flush=True)
libstc.print_value('std_MP2_energy', libstc.fe(float(np.std(E_arr, ddof=1))), flush=True)
libstc.print_value('pyscf_MP2_energy', libstc.fe(E_mp2_pyscf), flush=True)
libstc.print_value('mean_MP2_error', libstc.fe(float(np.mean(E_arr)) - E_mp2_pyscf), flush=True)
libstc.print_value('predicted_std', libstc.fe(energies[0][2]), flush=True)

if options.out is not None:
    os.makedirs(os.path.dirname(options.out) or '.', exist_ok=True)
    with open(options.out, 'w') as f:
        f.write(f'# {options.xyzfile} {options.basis} screen {options.screen} '
                f'seed {options.seed} nrepeat {options.nrepeat}\n')
        f.write(f'# pyscf_MP2_energy {libstc.fe(E_mp2_pyscf)}\n')
        f.write(f'# E_exchange E_mp2 predicted_std\n')
        for E_exchange, E_mp2, std in energies:
            f.write(f'{libstc.fe(E_exchange)} {libstc.fe(E_mp2)} {libstc.fe(std)}\n')
    libstc.print_value('out', options.out, flush=True)
