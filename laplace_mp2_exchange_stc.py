#!/usr/bin/env python
import os
import sys
print(*sys.argv[1:])
# Central job scripts can override these defaults. Locally, keep PySCF memory
# moderate and use a stable in-tree torchinductor cache unless told otherwise.
os.environ['CUDA_HOME'] = ''
os.environ['CUDA_PATH'] = ''
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['TORCHINDUCTOR_CPP_DYNAMIC_THREADS'] = '1'
this_dir = os.path.dirname(__file__)

if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
    nthreads = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
else:
    nthreads = os.cpu_count()

# One cache per thread count. torchinductor bakes the thread count into the
# generated OpenMP code (#pragma omp parallel num_threads(N)), so a shared cache
# lets a run reuse kernels compiled for a different N -- silently, and fatally for
# any thread-scaling measurement.
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(this_dir, f'cache_dynamic')

if 'SLURM_MEM_PER_NODE' in os.environ:
    os.environ['PYSCF_MAX_MEMORY'] = str(int(int(os.environ['SLURM_MEM_PER_NODE']) * 0.9))
else:
    import psutil
    os.environ['PYSCF_MAX_MEMORY'] = str(int(psutil.virtual_memory().available / 1e6 * 0.6))

import argparse

parser = argparse.ArgumentParser(description='STC estimator for Laplace-transformed MP2 exchange energy')
parser.add_argument('xyzfile', type=str)
parser.add_argument('basis', type=str)
parser.add_argument('target_error', type=float)
parser.add_argument('-orb', type=str, default='PM', choices=['canonical', 'PM', 'Boys'])
parser.add_argument('-charge', type=int, default=0)
parser.add_argument('-M', type=int, default=8)
parser.add_argument('-nsamples', type=int, default=None)
parser.add_argument('-screens', nargs='+', type=float, required=True)
parser.add_argument('-block_size', type=int, default=None)
parser.add_argument('--nowarmup_compile', action='store_true')
parser.add_argument('--all_electron', action='store_true',
                    help='correlate the core as well; the frozen core is the default')
parser.add_argument('--static_compile', action='store_true')
parser.add_argument('-verbose', type=int, default=1)
parser.add_argument('--vir_oao', action='store_true', help='span the virtual space with the virtual projection of the Lowdin AO basis (nao redundant, atom-local columns) instead of localized virtual MOs; -orb then only affects the occupied orbitals')
parser.add_argument('--legendre', action='store_true', help='use Gauss-Legendre in the substituted variable, which knows only Delta_min, instead of the default minimax (Chebyshev) grid fitted over the actual denominator range [Delta_min, Delta_max]')
parser.add_argument('--group_aux', action='store_true', help='sample one atom-sized group of auxiliary indices and evaluate the whole group exactly, instead of sampling a single auxiliary index')
parser.add_argument('-stdp', type=float, default=None, help='use the a-priori trace proxy weight*(Tr exp(beta Focc) Tr exp(-beta Fvir))**p to set per-beta target errors instead of allocating from the measured sigma_beta ratios; p ~ 1.4 is a good default')
parser.add_argument('-std_tol', type=float, default=2e-2, help='target relative accuracy of the STC std estimate, default to be 0.02')
parser.add_argument('-nrepeat', type=int, default=1, help='repeat STC sampling this many times; >1 enables empirical vs predicted std comparison')
options = parser.parse_args()
# Frozen core everywhere by default: an all-electron reference silently compared
# against a frozen-core STC energy is a hard error to spot downstream.
options.frozen_core = not options.all_electron

if options.static_compile:
    os.environ['STC_TORCH_COMPILE_DYNAMIC'] = '0'

# libstc reads STC_TORCH_COMPILE_DYNAMIC at import time, so argparse must happen before this import.
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
    # Labelled text written by mp2.py; pick out the three values this script needs.
    values = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            values[parts[0]] = float(parts[1])
    return [values[label] for label in ('E_mp2_pyscf', 'E_exchange_pyscf', 'E_direct_pyscf')]


def prepare_tensors(mol, rhf, options, with_df):
    if options.frozen_core:
        ncore = utils_pyscf.get_ncore(mol)
    else:
        ncore = 0

    nao = mol.nao
    nocc0 = mol.nelectron // 2
    nocc = nocc0 - ncore
    nvir = nao - nocc0
    # Not get_naoaux(): that builds the AO three-index tensor when _cderi is None,
    # which is exactly what make_ri_df avoids.
    naux = with_df.auxmol.nao_nr()
    o = slice(ncore, nocc0)
    v = slice(nocc0, None)

    if options.verbose >= 1:
        libstc.print_value('nao', f'{nao:d}', flush=True)
        libstc.print_value('ncore', f'{ncore:d}', flush=True)
        libstc.print_value('nocc', f'{nocc:d}', flush=True)
        libstc.print_value('nvir', f'{nvir:d}', flush=True)
        libstc.print_value('naux', f'{naux:d}', flush=True)
        Rov_size = nocc * nvir * naux * 8 / 1024**2
        libstc.print_value('Rov size (MB)', f'{Rov_size:.1f}', flush=True)

    C = rhf.mo_coeff
    Cocc = C[:, o]
    Cvir = C[:, v]
    eocc = torch.from_numpy(rhf.mo_energy[o].copy())
    evir = torch.from_numpy(rhf.mo_energy[v].copy())

    t0 = time.perf_counter()
    Cocc_local, Cvir_local, Uaux = libstc.get_local_orbitals(mol, Cocc, Cvir, options.orb, localize_vir=not options.vir_oao)
    if options.vir_oao:
        # Virtual projection of the Lowdin AO basis: Cvir W with W = Cvir^T S^{1/2}.
        # W W^T = Cvir^T S Cvir = I, so W is a Parseval frame and every contraction
        # over the virtual index is unchanged, even though the nao columns span only
        # nvir dimensions. Atom-local by construction, so no virtual localization.
        w, V = np.linalg.eigh(mol.intor('int1e_ovlp'))
        Cvir_local = Cvir @ (Cvir.T @ (V * np.sqrt(w)) @ V.T)
    # A non-trivial Uaux would rotate the auxiliary index and destroy its atom
    # blocking. The restricted -orb choices all return None; assert rather than assume.
    assert Uaux is None, 'auxiliary rotation is incompatible with atom-blocked aux sectors'
    if options.verbose >= 1:
        libstc.print_time_value('time_localization', time.perf_counter() - t0, flush=True)

    Cocc_local = torch.from_numpy(Cocc_local.copy())
    Cvir_local = torch.from_numpy(Cvir_local.copy())

    t0 = time.perf_counter()
    R = libstc.get_DF_ov_blockwise(
        with_df,
        Cocc_local.numpy(),
        Cvir_local.numpy(),
    )
    # F C = S C e with C^-1 = C^T S, so F = S C e C^T S. Exact for the canonical
    # orbitals on the chkfile, and O(nao^3) rather than the J/K build get_fock
    # would trigger -- which, with no cderi in memory, means a full DF build.
    S = mol.intor('int1e_ovlp')
    SC = S @ rhf.mo_coeff
    F_ao = torch.from_numpy((SC * rhf.mo_energy) @ SC.T)
    Focc = Cocc_local.T @ F_ao @ Cocc_local
    Fvir = Cvir_local.T @ F_ao @ Cvir_local
    if options.verbose >= 1:
        libstc.print_value('R_shape', str(tuple(R.shape)), flush=True)
        libstc.print_value('Rov size (MB)', f'{np.prod(R.shape) * 8 / 1024**2:.1f}', flush=True)
        libstc.print_value('Rov file', R.filename, flush=True)
        libstc.print_time_value('time_DF_transform', time.perf_counter() - t0, flush=True)

    return R, Focc, Fvir, eocc, evir


def print_options(options, screens):
    libstc.print_value('system', options.xyzfile, flush=True)
    libstc.print_value('basis', options.basis, flush=True)
    libstc.print_value('orbital', options.orb, flush=True)
    libstc.print_value('target_error', libstc.fe(options.target_error), flush=True)
    libstc.print_value('quadrature_M', f'{options.M:d}', flush=True)
    libstc.print_value('quadrature_a', libstc.ff(libstc.a), flush=True)
    libstc.print_value('quadrature', 'gauss-legendre' if options.legendre else 'minimax', flush=True)
    libstc.print_value('pilot_nsamples', f'{libstc.pilot_nsamples:d}', flush=True)
    libstc.print_value('std_tol', libstc.ff(libstc.std_tol), flush=True)
    libstc.print_value('std_nrun', f'{libstc.std_nrun:d}', flush=True)
    libstc.print_value('screens', ' '.join(str(screen) for screen in screens), flush=True)
    libstc.print_value('stdp', 'None' if options.stdp is None else libstc.ff(options.stdp), flush=True)
    libstc.print_value('batch_size', f'{libstc.batch_size:d}', flush=True)
    libstc.print_value('warmup_nsamples', f'{libstc.warmup_nsamples:d}', flush=True)
    libstc.print_value('min_nsamples', f'{libstc.min_nsamples:d}', flush=True)
    libstc.print_value('block_size', str(options.block_size), flush=True)
    libstc.print_value('group_aux', str(options.group_aux), flush=True)
    libstc.print_value('numba_group_evaluation', str(libstc.numba_group_evaluation), flush=True)
    libstc.print_value('vir_oao', str(options.vir_oao), flush=True)
    libstc.print_value('warmup_compile', str(not options.nowarmup_compile), flush=True)
    libstc.print_value('compile_dynamic', str(libstc.compile_dynamic), flush=True)


xyzfile = options.xyzfile
basis = options.basis
basename = os.path.basename(xyzfile)
label = basename[:-4] if basename.endswith('.xyz') else basename
directory = libstc.prepared_dir(label, basis, options.charge)

screens = options.screens

# The SCF is produced by scf.py and is required: do_scf would silently rerun it if
# the chkfile were missing, so check before calling it. The PySCF reference from
# mp2.py is only used to report errors, so it is optional -- without it everything
# is computed as usual and the error lines are omitted.
chkfile = libstc.require_file(
    libstc.scf_chk_path(directory),
    f'python scf.py {xyzfile} {basis} -charge {options.charge} --save')
dfmp2_file = libstc.dfmp2_path(directory, options.frozen_core)
if not os.path.exists(dfmp2_file):
    dfmp2_file = None

mol, atom_naux = libstc.build_mol(xyzfile, basis, options.charge)
aux_sectors = libstc.make_aux_sectors(atom_naux)

if options.group_aux and options.block_size is not None:
    parser.error('-block_size and --group_aux are mutually exclusive: group mode needs no aux alignment')

if options.verbose >= 1:
    print_options(options, screens)
    libstc.print_value('aux_sectors', libstc.describe_sectors(aux_sectors), flush=True)
    libstc.print_value('scf_chkfile', chkfile, flush=True)
    libstc.print_value('dfmp2_file', dfmp2_file if dfmp2_file else 'none', flush=True)

rhf = libstc.load_scf(mol, directory)
with_df_ri = libstc.make_ri_df(mol)
rhf.with_df = with_df_ri
libstc.print_value('auxbasis_corr', str(with_df_ri.auxbasis), flush=True)

if dfmp2_file is None:
    E_mp2_pyscf = E_exchange_pyscf = E_direct_pyscf = None
else:
    E_mp2_pyscf, E_exchange_pyscf, E_direct_pyscf = load_dfmp2_energies(dfmp2_file)

R, Focc, Fvir, eocc, evir = prepare_tensors(mol, rhf, options, with_df_ri)

driver = libstc.STCLaplaceMP2(R, Focc, Fvir, eocc, evir, verbose=options.verbose,
                              sectors=aux_sectors if options.group_aux else None)


def print_summary(res, E_direct_lap):
    # Only the exchange term is sampled, so that is what this reports. The total MP2
    # energy is assembled once at the end, after every screen and repeat is done.
    E_exchange_stc = res.energy.item()
    E_exchange_screen = res.energy_screen.item()
    E_mp2_stc = E_exchange_stc - 2.0 * E_direct_lap
    time_total = driver.time_direct + driver.time_transform + res.time_exchange
    libstc.print_value('summary_screen', libstc.screen_label(res.screen), flush=True)
    libstc.print_value('production_nsamples_total', f'{res.nsamples_total:d}', flush=True)
    libstc.print_time_value('time_MP2_exchange', res.time_exchange, flush=True)
    libstc.print_time_value('time_MP2_total', time_total, flush=True)
    if driver.verbose >= 1:
        libstc.print_value('screen_done', libstc.screen_label(res.screen), flush=True)
    libstc.print_value('exchange_energy_screen', libstc.fe(E_exchange_screen), flush=True)
    libstc.print_value('exchange_energy', libstc.fe(E_exchange_stc), flush=True)
    if E_exchange_pyscf is not None:
        libstc.print_value('pyscf_exchange_energy', libstc.fe(E_exchange_pyscf), flush=True)
        libstc.print_value('exchange_error', libstc.fe(E_exchange_stc - E_exchange_pyscf), flush=True)
    libstc.print_value('exchange_std', libstc.fe(res.std.item()), flush=True)
    return E_mp2_stc


# Total MP2 energy of every repeat, per screen, for the -nrepeat summary.
repeat_energies = {screen: [] for screen in screens}

for irep in range(options.nrepeat):
    beta_weight = driver.get_beta_list(M=options.M, minimax=not options.legendre)
    if not options.nowarmup_compile:
        driver.warmup_compile(screens[0], block_size=options.block_size)
    results = [libstc.ScreenResult(screen) for screen in screens]
    driver.time_transform = 0.0
    driver.time_direct = 0.0

    if options.stdp is None:
        # Default allocation needs every sigma_beta before any beta can be sized, so
        # all beta points stay resident and the screens are the outer loop.
        t0 = time.perf_counter()
        states = [driver.make_beta_state(beta, coeff) for beta, coeff in beta_weight]
        driver.time_transform = time.perf_counter() - t0
        E_direct_lap = sum(state.direct_mp2() for state in states)
        driver.time_direct = sum(state.time_direct for state in states)
        for res in results:
            for state in states:
                state.build_screen(res.screen)
                state.build_sampler(block_size=options.block_size, sectors=driver.sectors)
            nsamples_all = libstc.allocate_samples([state.estimate_std() for state in states],
                                                   options.target_error)
            for state, nsamples in zip(states, nsamples_all):
                state.STC_exchange(nsamples)
                res.add(state, verbose=driver.verbose)
    else:
        # With a-priori per-beta error targets each beta only needs its own sigma, so
        # beta is the outer loop and each point is released as soon as it is finished.
        # Peak memory holds one Rhalf and one sampler instead of M of each.
        proxy = driver.beta_std_proxy(options.stdp)
        target_variance = options.target_error**2 * proxy / torch.sum(proxy)
        E_direct_lap = torch.zeros((), dtype=driver.R.dtype)
        state = None
        for (beta, coeff), t_beta in zip(beta_weight, target_variance):
            # Release the previous point before allocating the next one. Assignment
            # alone evaluates make_beta_state first, so the old Rhalf would still be
            # alive while the new one is allocated, doubling the largest tensor.
            state = None
            state = driver.make_beta_state(beta, coeff)
            driver.time_transform += state.time_transform
            E_direct_lap = E_direct_lap + state.direct_mp2()
            driver.time_direct += state.time_direct
            for res in results:
                state.build_screen(res.screen)
                state.build_sampler(block_size=options.block_size, sectors=driver.sectors)
                nsamples = libstc.allocate_from_targets(state.estimate_std(), t_beta)
                state.STC_exchange(nsamples)
                res.add(state, verbose=driver.verbose)
    E_direct_lap = E_direct_lap.item()

    if driver.verbose >= 1:
        libstc.print_time_value('time_laplace_transform', driver.time_transform, flush=True)
        libstc.print_time_value('time_direct', driver.time_direct, flush=True)
    libstc.print_value('direct_energy', libstc.fe(E_direct_lap), flush=True)
    if E_direct_pyscf is not None:
        libstc.print_value('pyscf_direct_energy', libstc.fe(E_direct_pyscf), flush=True)
        libstc.print_value('direct_error', libstc.fe(E_direct_lap - E_direct_pyscf), flush=True)

    for res in results:
        if driver.verbose >= 1:
            print('', flush=True)
            libstc.print_value('screen_started', libstc.screen_label(res.screen), flush=True)
            libstc.print_value('screen_nvir_per_occ_avg', libstc.ff(res.nvir_per_occ_avg), flush=True)
            libstc.print_time_value('time_build_screen', res.time_build_screen, flush=True)
            libstc.print_time_value('time_table', res.time_table, flush=True)
            libstc.print_time_value('time_sampler', res.time_sampler, flush=True)
            libstc.print_time_value('time_exact', res.time_exact, flush=True)
            libstc.print_time_value('time_std', res.time_std, flush=True)
            libstc.print_time_value('time_sample', res.time_sample, flush=True)
            libstc.print_time_value('time_sample_draw', res.time_sample_draw, flush=True)
            libstc.print_time_value('time_sample_evaluate', res.time_sample_evaluate, flush=True)
        repeat_energies[res.screen].append(print_summary(res, E_direct_lap))

# Total MP2 energy = K - 2J, with K sampled and J exact for this quadrature. Both
# carry the same Laplace error, so a large residual here with a small exchange_error
# points at M rather than at the sampling.
for screen in screens:
    energies = np.array(repeat_energies[screen])
    print('', flush=True)
    libstc.print_value('total_screen', libstc.screen_label(screen), flush=True)
    libstc.print_value('total_nrepeat', f'{len(energies):d}', flush=True)
    libstc.print_value('total_MP2_energy', libstc.fe(np.mean(energies)), flush=True)
    if E_mp2_pyscf is not None:
        libstc.print_value('pyscf_MP2_energy', libstc.fe(E_mp2_pyscf), flush=True)
        libstc.print_value('total_MP2_error', libstc.fe(np.mean(energies) - E_mp2_pyscf), flush=True)
    if len(energies) > 1:
        # Empirical spread over the repeats, to compare against the predicted
        # exchange_std of a single repeat.
        libstc.print_value('total_MP2_energy_std', libstc.fe(np.std(energies, ddof=1)), flush=True)
