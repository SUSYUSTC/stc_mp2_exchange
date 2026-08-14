import os
import time
import atexit

# numba would otherwise start its own thread pool alongside torch's OpenMP pool and
# the two oversubscribe the cores: the ragged group dot measured 54 ns/sample with
# the default layer against 26 ns with 'omp'. Must be set before numba is imported.
os.environ.setdefault('NUMBA_THREADING_LAYER', 'omp')

import numba
import numpy as np
import torch
import pyscf
import pyscf.gto
import pyscf.df
import pyscf.df.incore
import pyscf.scf
import pyscf.cc  # do not delete!
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.outcore import balance_partition

import libquad
import stc_cc.la as la
import stc_cc.sample as sample
import stc_cc.utils_pyscf as utils_pyscf
import stc_cc.orbopt as orbopt


batch_size = 20000000
# Rov lives on disk; the Laplace transform reads it back in x-blocks of about this
# many bytes. Small enough that the block workspace is a rounding error next to
# Rhalf, large enough that the two matmuls stay efficient.
aux_block_bytes = 512 * 1024**2
# Auxiliary block target when transforming the DF tensor to Rov. Independent of
# PYSCF_MAX_MEMORY, which is sized for the SCF and is far too large here.
df_block_bytes = 512 * 1024**2
a = 0.5
std_tol = 2e-2
std_nrun = 16
pilot_nsamples = int(os.environ.get('STC_PILOT_NSAMPLES', 10000))
warmup_nsamples = 1000000
# Floor on any sample count: what a beta point is allocated, what a branch of it
# draws, and the shortest batch handed to the compiled kernel. Keeping it above 1
# also keeps inductor off its size-0/1 specializations.
min_nsamples = 10
compile_dynamic = os.environ.get('STC_TORCH_COMPILE_DYNAMIC', '1') not in ('0', 'False', 'false')
# Evaluate the grouped auxiliary sum with the numba ragged kernel instead of the
# torch bucket-by-sector one. Both give the same numbers to one ulp; numba is
# 1.7-8.2x faster because it can branch on the group length, so a sample gathers
# only its own group instead of every sector's.
numba_group_evaluation = True

table_timing_labels = [
    'time_table_ABC',
    'time_table_branch',
]

sampler_timing_labels = [
    'time_sampler_x',
]


def fe(x):
    return f'{x:.12e}'


def ff(x):
    return f'{x:.6e}'


def print_value(label, value, flush=False):
    print(f'{label:28s} {value}', flush=flush)


def print_time_value(label, value, flush=False):
    print(f'{label:28s} {value:.4f}', flush=flush)


def compile_func(func):
    return torch.compile(func, dynamic=compile_dynamic)


def ceil_divide(n, m):
    # Integer ceiling division, staying in integers rather than rounding a float.
    # Gives 0 for n = 0, as floor division of -1 by a positive m is -1.
    return (n - 1) // m + 1


def get_average_until_convergence(func, tol, init_nsamples, nrun):
    nsamples = init_nsamples
    nsamples_total = 0
    while True:
        xs = func(nsamples)
        nsamples_total += nsamples * nrun
        ratios = torch.std(xs, dim=-1) / torch.mean(xs, dim=-1) / torch.sqrt(torch.tensor(xs.shape[-1]))
        assert not torch.isnan(ratios).any()
        if torch.max(ratios) < torch.tensor(tol):
            break
        nsamples *= 2
    return torch.mean(xs, dim=-1), nsamples, nsamples_total, ratios.max().item()


def get_local_orbitals(mol, Cocc, Cvir, orb, localize_vir=True):
    if orb == 'canonical':
        return Cocc, Cvir, None

    Cocc_atomic = utils_pyscf.localize_atomic(mol, Cocc)
    # localize_vir=False: the caller replaces the virtuals wholesale (--vir_oao),
    # so skip their optimization, which is the expensive half (nvir >> nocc).
    Cvir_atomic = utils_pyscf.localize_atomic(mol, Cvir) if localize_vir else Cvir
    builder_continuous = lambda params: torch.optim.LBFGS(params, max_iter=5, history_size=5, line_search_fn='strong_wolfe')

    if orb == 'PM':
        Uocc = orbopt.PM(mol, Cocc_atomic).run(builder_continuous, 1e-4)
        if not localize_vir:
            return Cocc_atomic @ Uocc, Cvir, None
        Uvir = orbopt.PM(mol, Cvir_atomic).run(builder_continuous, 1e-4)
        return Cocc_atomic @ Uocc, Cvir_atomic @ Uvir, None

    if orb == 'Boys':
        Uocc = orbopt.Boys(mol, Cocc_atomic).run(builder_continuous, 1e-4)
        if not localize_vir:
            return Cocc_atomic @ Uocc, Cvir, None
        Uvir = orbopt.Boys(mol, Cvir_atomic).run(builder_continuous, 1e-4)
        return Cocc_atomic @ Uocc, Cvir_atomic @ Uvir, None

    raise RuntimeError('unknown orbital option')


def prepared_dir(label, basis, charge):
    # One directory per (system, basis, charge). The SCF chkfile inside it is loaded
    # by do_scf with no validation of atoms or basis, so anything that changes the
    # geometry ordering or the basis must land in a different directory.
    return os.path.join('data_MP2', f'{label}_{basis}_chg{charge}')


def scf_chk_path(directory):
    # do_scf builds this name itself from the SCF type; mirror it so consumers can
    # check for the file before calling do_scf, which would otherwise recompute.
    return os.path.join(directory, 'RHF.chk')


def dfmp2_path(directory, frozen_core):
    return os.path.join(directory, 'dfmp2_fc.txt' if frozen_core else 'dfmp2.txt')


def require_file(path, hint):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found; produce it with:\n    {hint}')
    return path


def sort_atoms_by_aux(mol):
    # Auxiliary functions are blocked by atom, so sorting atoms by their per-atom
    # naux makes every set of equal-sized atoms a contiguous, rectangular slab of the
    # aux axis. Doing it here, before any calculation, means nothing downstream ever
    # has to permute a tensor. Physically a no-op: total energies are invariant.
    auxmol = pyscf.df.addons.make_auxmol(mol, pyscf.df.addons.make_auxbasis(mol, mp2fit=True))
    atom_naux = [int(s[3] - s[2]) for s in auxmol.aoslice_by_atom()]
    order = sorted(range(mol.natm), key=lambda i: (-atom_naux[i], i))
    if order == list(range(mol.natm)):
        return mol, [atom_naux[i] for i in order]
    coords = mol.atom_coords()
    atoms = [(mol.atom_symbol(i), coords[i]) for i in order]
    mol_sorted = pyscf.gto.Mole(atom=atoms, unit='Bohr', basis=mol.basis, charge=mol.charge, verbose=mol.verbose)
    mol_sorted.build()
    return mol_sorted, [atom_naux[i] for i in order]


def make_ri_df(mol):
    # The SCF keeps the default JK-fitting auxiliary basis, which is what that set
    # is designed for. Correlation integrals instead use the RI/MP2-fitting set:
    # it is tabulated for every element here (no even-tempered fallback, so no
    # Na/Mg-style outliers), uniform within a periodic table row, and smaller than
    # JKFIT at double zeta.
    #
    # Deliberately not built. The fit couples only the auxiliary index and the MO
    # transform only mu,nu, so the two commute and can be applied in either order.
    # Fitting in the AO basis means forming (nao_pair, naux); fitting after the
    # transform means forming only (nocc*nvir, naux), smaller by nao_pair/(nocc*nvir)
    # -- 8.6x at BN_8x8, 1450 GB against 168 GB. Both consumers take the second
    # route: PySCF's DFMP2 switches to _init_mp_df_eris_direct when _cderi is None,
    # and get_DF_ov_blockwise below does the same thing by hand.
    with_df = pyscf.df.DF(mol)
    with_df.auxbasis = pyscf.df.make_auxbasis(mol, mp2fit=True)
    with_df.auxmol = pyscf.df.addons.make_auxmol(mol, with_df.auxbasis)
    with_df.verbose = 4
    return with_df


def load_scf(mol, directory):
    # Not do_scf: that routes through get_DF, which calls build() eagerly and
    # unconditionally, so loading a converged SCF would contract the whole
    # JK-fitting three-index tensor. Nothing here needs it -- the SCF never
    # iterates -- and callers that want integrals set with_df themselves (mp2.py
    # to the RI set). density_fit leaves an unbuilt DF in place, so any caller
    # that does need JK integrals still builds them on first use.
    rhf = pyscf.scf.RHF(mol).density_fit()
    rhf.verbose = 4
    _, attrs = pyscf.scf.chkfile.load_scf(scf_chk_path(directory))
    for key, value in attrs.items():
        setattr(rhf, key, value)
    # pyscf does not store the convergence flag in the chkfile, so mf.converged
    # stays at its class default of False. MP2 branches on it and would drop into
    # the non-canonical iterative kernel, which DFMP2 raises NotImplementedError
    # for. scf.py only leaves a converged result behind, so restore it.
    rhf.converged = True
    return rhf


def build_mol(xyzfile, basis, charge):
    mol = pyscf.gto.Mole(atom=xyzfile, basis=utils_pyscf.get_basis(basis),
                         charge=charge, verbose=0)
    mol.build()
    return sort_atoms_by_aux(mol)


class RovStore:
    # An h5py Dataset does not own its file, and H5TmpFile deletes the file as soon
    # as the handle is collected, so the two have to be kept together. Reads return
    # numpy; only the slices the Laplace transform asks for are ever materialized.
    def __init__(self, handle, dataset):
        self.handle = handle
        self.dataset = dataset
        self.shape = dataset.shape
        self.filename = handle.filename
        # torch dtype, so callers can size accumulators off the store directly
        self.dtype = torch.from_numpy(np.empty(0, dtype=dataset.dtype)).dtype
        # Closing deletes the file. Registered here rather than left to __del__ so
        # it runs while h5py is still importable: a finalizer during interpreter
        # teardown raises inside h5py and the file can survive a crash.
        atexit.register(self.close)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.dataset = None

    def __getitem__(self, key):
        return self.dataset[key]


def get_DF_ov_blockwise(with_df, Cocc_local, Cvir_local):
    # Build the local occupied-virtual DF tensor R_{xia} without ever forming the AO
    # three-index tensor: raw 3-centre integrals are contracted to MO one auxiliary
    # block at a time, and the fitting metric is applied afterwards to the assembled
    # (naux, nocc*nvir) array. See make_ri_df for why the two commute.
    #
    # Rov is resident during the build and written to disk once at the end. That
    # write is not for capacity -- Rov fits -- but for the Laplace transform, which
    # holds a second Rov-sized Rhalf and so reads this one back an x-block at a time.
    #
    # xia is the layout the Laplace transform consumes; only the per-beta Rhalf it
    # produces is in iax order. The file lives in PYSCF_TMPDIR and is deleted when
    # the dataset goes out of scope.
    mol = with_df.mol
    auxmol = with_df.auxmol
    nao = Cocc_local.shape[0]
    nocc = Cocc_local.shape[1]
    nvir = Cvir_local.shape[1]
    nbas = mol.nbas
    nao_pair = nao * (nao + 1) // 2
    naux = auxmol.nao_nr()

    # The symmetric inverse square root, matching make_df_eig: that patch forces the
    # AO path onto matrix_operation(j2c, 1/sqrt) too, so both routes produce the same
    # R rather than merely the same (ia|jb). Any square root of J reproduces the
    # integrals -- they differ by an orthogonal rotation of the auxiliary index -- but
    # only this one is right for STC, which reads structure off that index: it drops
    # no vectors, so naux stays equal to sum(atom_naux) as make_aux_sectors assumes,
    # and being symmetric it privileges no auxiliary ordering the way a triangular
    # Cholesky factor would.
    metric = utils_pyscf.matrix_operation(
        pyscf.df.incore.fill_2c2e(mol, auxmol), lambda x: 1 / np.sqrt(x))

    # Rov is resident for the whole build. Rhalf does not exist yet, so the full
    # budget is free here; it is only the Laplace transform, which holds Rhalf, that
    # needs Rov on disk.
    Rov = np.empty((naux, nocc * nvir))
    mo = np.asarray(np.hstack((Cocc_local, Cvir_local)), order='F')
    ijslice = (0, nocc, nocc, nocc + nvir)

    int3c = pyscf.gto.moleintor.ascint3(mol._add_suffix('int3c2e'))
    atm_f, bas_f, env_f = pyscf.gto.mole.conc_env(
        mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc_f = pyscf.gto.moleintor.make_loc(bas_f, int3c)
    cintopt = pyscf.gto.moleintor.make_cintopt(atm_f, bas_f, env_f, int3c)

    # Block the auxiliary index, keeping all of mu,nu: the transform needs every AO
    # pair to produce any (i,a). nr_e2 contracts straight out of the packed s2 form,
    # so no nao**2 intermediate is ever unpacked -- at nao 5276 that alone would be
    # 223 MB per auxiliary function. The two buffers are the raw block and its
    # transpose.
    blksize = max(1, df_block_bytes // (nao_pair * 8 * 2))
    ranges = balance_partition(auxmol.ao_loc, blksize)
    auxlen = max(dk for _, _, dk in ranges)
    buf0 = np.empty(auxlen * nao_pair)
    buf0T = np.empty(auxlen * nao_pair)

    x1 = 0
    for kshl0, kshl1, dk in ranges:
        x0, x1 = x1, x1 + dk
        shls_slice = (0, nbas, 0, nbas, nbas + kshl0, nbas + kshl1)
        pqL = pyscf.gto.moleintor.getints3c(int3c, atm_f, bas_f, env_f, shls_slice,
                                            1, 's2ij', ao_loc_f, cintopt, out=buf0)
        Lpq = pyscf.lib.transpose(pqL, out=buf0T)
        Rov[x0:x1] = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2')
        pqL = Lpq = None
    assert x1 == naux, f'auxiliary blocks summed to {x1}, expected {naux}'
    buf0 = buf0T = None

    # Apply the metric now, in the MO basis. Blocked over (i,a) purely to bound the
    # temporary: the columns are independent, so this is exact, not an approximation.
    fitblk = max(1, df_block_bytes // (naux * 8))
    for p0 in range(0, nocc * nvir, fitblk):
        p1 = min(p0 + fitblk, nocc * nvir)
        Rov[:, p0:p1] = metric @ Rov[:, p0:p1]

    handle = pyscf.lib.H5TmpFile()
    dataset = handle.create_dataset('Rov', (naux, nocc, nvir), dtype='f8')
    for x0 in range(0, naux, max(1, auxlen)):
        x1 = min(x0 + max(1, auxlen), naux)
        dataset[x0:x1] = Rov[x0:x1].reshape(x1 - x0, nocc, nvir)
    Rov = None
    return RovStore(handle, dataset)


def maybe_pad_last_dim(M, block_size):
    # Block sampling is only meant to align the auxiliary x conditional. Padding
    # zero rows in the last dimension leaves all probabilities and contractions
    # unchanged, but gives the block sampler a multiple of block_size.
    if block_size is None:
        return M
    n = M.shape[-1]
    nfull = ceil_divide(n, block_size) * block_size
    if nfull == n:
        return M
    out = torch.zeros(M.shape[:-1] + (nfull, ), dtype=M.dtype)
    out[..., :n] = M
    return out


def apply_R_laplace_exp(R, Eocc, Evir):
    # R is always stored xia. Equivalent to einsum('ij,ab,jbx->iax') on an iax
    # tensor, but done as two right-side dense matmuls; starting from xia makes
    # the output Rhalf contiguous in iax, which is the layout the samplers use.
    nx, nj, nb = R.shape
    ni = Eocc.shape[0]
    na = Evir.shape[0]
    out_a = torch.empty((na, nx, nj), dtype=R.dtype, device=R.device)
    torch.matmul(Evir, R.reshape((nx * nj, nb)).T, out=out_a.reshape((na, nx * nj)))
    out_i = torch.empty((ni, na, nx), dtype=R.dtype, device=R.device)
    torch.matmul(Eocc, out_a.reshape((na * nx, nj)).T, out=out_i.reshape((ni, na * nx)))
    return out_i


def build_laplace_Rhalf(R, beta, weight, Focc, Fvir):
    # R is the on-disk Rov dataset. The auxiliary index is a spectator of the
    # transform, so it is read back one x-block at a time and each result is written
    # straight into its slice of Rhalf. Peak is then Rhalf plus one block, instead of
    # Rov + Rhalf + the (nvir, naux, nocc) intermediate that a single call allocates.
    Eocc = torch.linalg.matrix_exp(0.5 * beta * Focc)
    Evir = torch.linalg.matrix_exp(-0.5 * beta * Fvir)
    naux, nocc, nvir = R.shape
    Rhalf = torch.empty((nocc, nvir, naux), dtype=Eocc.dtype)
    nx_block = max(1, aux_block_bytes // (nocc * nvir * 8))
    for x0 in range(0, naux, nx_block):
        x1 = min(x0 + nx_block, naux)
        block = torch.from_numpy(R[x0:x1])
        Rhalf[:, :, x0:x1] = apply_R_laplace_exp(block, Eocc, Evir)
    # In place: Rhalf is the single largest tensor in the program, and the
    # out-of-place form would hold a second full copy while scaling it.
    Rhalf *= weight ** 0.25
    return Rhalf


def exchange_from_R(Rhalf):
    # Positive MP2 exchange-like quantity K for one weighted Laplace point:
    # K = sum_{ijabxy} R_{iax} R_{jbx} R_{iby} R_{jay}.
    return torch.einsum('iax,jbx,iby,jay->', Rhalf, Rhalf, Rhalf, Rhalf)


def direct_from_R(Rhalf):
    # Positive MP2 direct quantity J for one weighted Laplace point. Flattening
    # (i,a) gives G_xy = sum_{ia} R_{iax} R_{iay}, then J = sum_xy G_xy^2.
    Rmat = Rhalf.reshape((-1, Rhalf.shape[2]))
    G = Rmat.T @ Rmat
    return torch.sum(G**2)


def build_ov_screen(Rhalf, threshold):
    Rov = torch.sqrt(torch.einsum('iax,iax->ia', Rhalf, Rhalf))
    return Rov > threshold


def screen_info(ov_screen, nocc, nvir):
    if ov_screen is None:
        return nocc * nvir, nvir, nvir, float(nvir)
    screen_rows = torch.sum(ov_screen, dim=1)
    screen_nov = torch.sum(screen_rows).item()
    screen_min = torch.min(screen_rows).item()
    screen_max = torch.max(screen_rows).item()
    #screen_mean = torch.mean(screen_rows.to(torch.float64)).item()
    screen_mean = torch.sqrt(torch.mean(screen_rows.to(torch.float64)**2)).item()
    return screen_nov, screen_min, screen_max, screen_mean


def exchange_screen_from_R(Rhalf, ov_screen):
    # Exact exchange contribution inside the screened local domains. For each
    # occupied pair ij, only virtuals kept by both occupied rows are included.
    nocc = Rhalf.shape[0]
    E = torch.tensor(0.0, dtype=Rhalf.dtype)
    for i in range(nocc):
        for j in range(nocc):
            screen = ov_screen[i] & ov_screen[j]
            if not torch.any(screen):
                continue
            Ri = Rhalf[i, screen]
            Rj = Rhalf[j, screen]
            G = Ri @ Rj.T
            E += torch.einsum('ab,ba->', G, G)
    return E


@compile_func
def build_RBov(Rhalf, Bx):
    # RBov[i,a] = sum_x |R_{iax}| Bx[x] is the row factor for the sampled a
    # conditional. Contracting against Bx inside this compiled kernel avoids
    # materializing a full |R| tensor.
    return torch.abs(Rhalf) @ Bx


@compile_func
def build_Bx(Rhalf):
    # Auxiliary factor Bx[x] = sqrt(sum_{ia} R_{iax}^2). Written as sum/sqrt
    # because this compiled form was faster than vector_norm for the real Rhalf
    # layout produced by the Laplace transform.
    return torch.sqrt(torch.sum(Rhalf**2, dim=(0, 1)))


class Sector:
    # A run of consecutive atoms that all carry the same number of auxiliary
    # functions. naux_per_atom and natom give the shape of its slab of the aux axis,
    # x_start locates that slab, and atom_start is the index of its first atom.
    #
    # Registered as a pytree node holding no tensors, only its four integers as
    # aux_data. That puts the layout in the treespec, which compares by value, so two
    # molecules with the same sector structure reuse the same compiled graphs instead
    # of guarding on object identity.
    def __init__(self, naux_per_atom, natom, atom_start, x_start):
        self.naux_per_atom = naux_per_atom
        self.natom = natom
        self.atom_start = atom_start
        self.x_start = x_start

    def flatten(self):
        return (), (self.naux_per_atom, self.natom, self.atom_start, self.x_start)

    @classmethod
    def unflatten(cls, children, aux_data):
        return cls(*aux_data)


torch.utils._pytree.register_pytree_node(Sector, Sector.flatten, Sector.unflatten)


def make_aux_sectors(atom_naux):
    # Auxiliary functions are blocked by atom, and the driver pre-sorts atoms so that
    # all atoms sharing the same per-atom naux are contiguous. Each such run of atoms
    # is a sector, whose slab of the aux axis reshapes to (natom, naux_per_atom) with
    # no padding. One atom is one sampling group, so the group index is simply the
    # atom index.
    #
    # These four integers are all evaluation needs; membership needs nothing extra,
    # since a sector is a contiguous range of atoms.
    atom_naux = [int(size) for size in atom_naux]
    sectors = []
    atom_start = 0
    x_start = 0
    while atom_start < len(atom_naux):
        naux_per_atom = atom_naux[atom_start]
        natom = 0
        while atom_start + natom < len(atom_naux) and atom_naux[atom_start + natom] == naux_per_atom:
            natom += 1
        sectors.append(Sector(naux_per_atom, natom, atom_start, x_start))
        atom_start += natom
        x_start += natom * naux_per_atom
    assert x_start == sum(atom_naux)
    sizes = [sec.naux_per_atom for sec in sectors]
    # A size appearing in two different runs means the atoms were not sorted, so the
    # sector slabs would not be rectangular.
    assert len(set(sizes)) == len(sizes), f'atoms are not sorted by naux: sector sizes {sizes}'
    # A tuple, not a list: it rides in Guide's pytree aux_data and so must be hashable.
    return tuple(sectors)


def sector_view(M, sec):
    # Zero-copy (..., natom, naux_per_atom) view of one sector's slab of the aux
    # axis, for Rhalf (i,a,x) or Bx (x,).
    slab = M[..., sec.x_start:sec.x_start + sec.natom * sec.naux_per_atom]
    return slab.reshape(slab.shape[:-1] + (sec.natom, sec.naux_per_atom))


def describe_sectors(sectors):
    return ' '.join(f'{sec.naux_per_atom}x{sec.natom}' for sec in sectors)


@compile_func
def build_RBovg(Rslab, Bslab):
    # Group weight RBovg[i,a,m] = sum_{x in atom m} |R_{iax}| Bx[x] for one sector.
    # Written as abs/mul/sum so inductor fuses it into a single pass and never
    # materializes |R|.
    return torch.sum(torch.abs(Rslab) * Bslab, dim=-1)


def build_group_table(Rhalf, Bx, sectors):
    parts = [build_RBovg(sector_view(Rhalf, sec), sector_view(Bx, sec)) for sec in sectors]
    RBovg = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
    return RBovg.contiguous()


@compile_func
def group_dot_sector(Rs, u_iajbm, offset):
    # sum over the whole group: Rs has a static trailing extent a_s, so this compiles
    # to one specialization per sector with only the batch dimension dynamic.
    u_i, u_a, u_j, u_b, u_m = u_iajbm
    u_m_offset = u_m - offset
    return torch.sum(Rs[u_i, u_a, u_m_offset] * Rs[u_j, u_b, u_m_offset], dim=-1)


def group_dot_torch(sectors, Rhalf, u_iajbm):
    u_i, u_a, u_j, u_b, u_m = u_iajbm
    if len(sectors) == 1:
        sec = sectors[0]
        return group_dot_sector(sector_view(Rhalf, sec), (u_i, u_a, u_j, u_b, u_m), sec.atom_start)
    out = torch.zeros(u_m.shape[0], dtype=Rhalf.dtype)
    for sec in sectors:
        match = (u_m >= sec.atom_start) & (u_m < sec.atom_start + sec.natom)
        out[match] = group_dot_sector(sector_view(Rhalf, sec), (u_i[match], u_a[match], u_j[match], u_b[match], u_m[match]), sec.atom_start)
    return out


@numba.njit(parallel=True, fastmath=True, cache=True)
def group_dot_ragged(R, gstart, glen, u_i, u_a, u_j, u_b, u_m, out):
    # The ragged sum written directly: the trip count is the group's own length, so a
    # sample gathers exactly its group and nothing else. Not expressible in torch,
    # whose ops have no data-dependent trip count -- that is what forces the bucketing
    # below. np.dot (-> BLAS ddot) is load-bearing: an explicit "for k in range(L)" is
    # 4-9x slower because numba will not vectorize a variable-trip-count reduction.
    for n in numba.prange(u_m.shape[0]):
        m = u_m[n]
        s = gstart[m]
        L = glen[m]
        out[n] = np.dot(R[u_i[n], u_a[n], s:s + L], R[u_j[n], u_b[n], s:s + L])
    return out


_group_index_cache = {}


def sector_group_index(sectors):
    # (start, length) per atom group, flattened across sectors, as the numba kernel
    # wants them. Cached because it is rebuilt on every batch otherwise.
    key = tuple((sec.naux_per_atom, sec.natom, sec.x_start) for sec in sectors)
    if key not in _group_index_cache:
        gstart = np.concatenate([sec.x_start + sec.naux_per_atom * np.arange(sec.natom) for sec in sectors])
        glen = np.concatenate([np.full(sec.natom, sec.naux_per_atom) for sec in sectors])
        _group_index_cache[key] = (gstart.astype(np.int64), glen.astype(np.int64))
    return _group_index_cache[key]


def group_dot_numba(sectors, Rhalf, u_iajbm):
    # .numpy() aliases the torch storage, so nothing is copied here.
    gstart, glen = sector_group_index(sectors)
    u_m = u_iajbm[4]
    out = torch.empty(u_m.shape[0], dtype=Rhalf.dtype)
    group_dot_ragged(Rhalf.numpy(), gstart, glen,
                     *[u.contiguous().numpy() for u in u_iajbm], out.numpy())
    return out


def group_dot(sectors, Rhalf, u_iajbm):
    # Atoms in different sectors have different lengths, so gathering per sample would
    # be ragged. Bucket the batch by sector instead: each bucket is dense with a
    # uniform trailing extent. Only S buckets are needed, not S^2, because this sum
    # depends on a single group index.
    if numba_group_evaluation:
        return group_dot_numba(sectors, Rhalf, u_iajbm)
    else:
        return group_dot_torch(sectors, Rhalf, u_iajbm)


class Guide:
    # The tensors the evaluators need, given already-sampled indices: the transformed
    # tensor, the loop-broken row/column magnitudes and the total partition function,
    # plus the aux sector layout that says how to view them. Registered as a pytree
    # node so it can be handed straight to a compiled function.
    #
    # Nothing here depends on the branch or draws anything: the samplers live on
    # BetaSampler. That keeps the structure identical for screened and unscreened
    # beta points and for every branch, so the evaluators compile exactly once.
    def __init__(self, Rhalf, Aov, Bx, RBovg, Z, sectors):
        self.Rhalf = Rhalf
        self.Aov = Aov
        self.Bx = Bx
        self.RBovg = RBovg
        self.Z = Z
        self.sectors = sectors

    def flatten(self):
        return (self.Rhalf, self.Aov, self.Bx, self.RBovg, self.Z, self.sectors), None

    @classmethod
    def unflatten(cls, children, aux_data):
        return cls(*children)


torch.utils._pytree.register_pytree_node(Guide, Guide.flatten, Guide.unflatten)


@compile_func
def sample_indices(sampler_a, sampler_b, sampler_x, u_i, u_j, random_state):
    # Draw b|ij, a|ij, then x|ia and y|ja. Shared by both evaluators: with grouping
    # sampler_x draws an atom group, so u_x/u_y are group indices, but the chain is
    # the same.
    (u_b, ), random_state = sampler_b.sample_indices((u_i, u_j), random_state)
    (u_a, ), random_state = sampler_a.sample_indices((u_i, u_j), random_state)
    (u_x, ), random_state = sampler_x.sample_indices((u_i, u_a), random_state)
    (u_y, ), random_state = sampler_x.sample_indices((u_j, u_a), random_state)
    uabxy = (u_a, u_b, u_x, u_y)
    return uabxy


@compile_func
def combine_group(guide, S1, S2, u_ijabmn):
    # The two group sums replace R_{jbx} R_{iby} and the |R| factors that used to
    # cancel to a sign, and the guide integrated over the group gives RBovg in place
    # of |R_{iax}| Bx[x].
    u_i, u_j, u_a, u_b, u_m, u_n = u_ijabmn
    denom = guide.Aov[u_i, u_b] * guide.Aov[u_j, u_b] * guide.RBovg[u_i, u_a, u_m] * guide.RBovg[u_j, u_a, u_n]
    values = guide.Z * S1 * S2 / denom
    return torch.sum(values), values


def evaluate_group(guide, u_ijabmn):
    # u_x/u_y are atom groups here: the whole group is summed instead of a single
    # auxiliary index. Not compiled as a whole because group_dot loops over aux
    # sectors in Python; only the per-sector dot and the final combine are compiled.
    u_i, u_j, u_a, u_b, u_m, u_n = u_ijabmn
    sectors, Rhalf = guide.sectors, guide.Rhalf
    S1 = group_dot(sectors, Rhalf, (u_i, u_a, u_j, u_b, u_m))
    S2 = group_dot(sectors, Rhalf, (u_i, u_b, u_j, u_a, u_n))
    return combine_group(guide, S1, S2, u_ijabmn)


def make_x_sampler(Rhalf, Bx, RBovg, sectors, block_size):
    # Shared by every branch: the auxiliary conditional never sees the screening
    # mask, so it is built once. It is also by far the largest table.
    if sectors is not None:
        # One alias row per (i,a) over natom groups instead of naux auxiliary
        # functions, so this table is orders of magnitude smaller.
        return sample.BatchedAliasSampler(1, RBovg)
    if block_size is None:
        return sample.BatchedAliasSampler(1, Rhalf, Bx)
    else:
        return sample.AlignedBatchedBlockSampler(1, block_size, True, Rhalf, Bx)


@compile_func
def evaluate_perx(guide, u_ijabxy):
    # The |R_{iax}| and |R_{jay}| the guide used cancel against the true factors and
    # leave only their signs; the surviving R_{jbx} R_{iby} are divided by the norms
    # that replaced them. Branch-independent, so this compiles once and is reused for
    # every branch and for the unscreened case.
    u_i, u_j, u_a, u_b, u_x, u_y = u_ijabxy
    Rhalf, Aov, Bx = guide.Rhalf, guide.Aov, guide.Bx
    denom = Aov[u_i, u_b] * Aov[u_j, u_b] * Bx[u_x] * Bx[u_y]
    sign = torch.sign(Rhalf[u_i, u_a, u_x]) * torch.sign(Rhalf[u_j, u_a, u_y])
    values = guide.Z * Rhalf[u_j, u_b, u_x] * Rhalf[u_i, u_b, u_y] * sign / denom
    # Return individual values so the caller can compute sumsq and scatter
    # per-(i,j) stratum sums outside torch.compile for reliability.
    return torch.sum(values), values


def quota_counts(nsamples, weight):
    # Deterministic proportional split of nsamples over a short weight vector, the
    # same rule make_marginal_nsamples_quota uses. Done directly because that helper
    # is compiled and would specialize on a length-1 vector, which is exactly the
    # unscreened single-branch case.
    edges = torch.zeros(len(weight) + 1, dtype=torch.float64)
    torch.cumsum(weight.to(torch.float64), 0, out=edges[1:])
    edges = (edges / edges[-1] * nsamples).to(torch.int64)
    return torch.diff(edges)


def make_index_batches(counts, batch_size):
    # Turn an integer (i,j) count table into (u_i, u_j) index vectors. The samples
    # are split into ceil(total / batch_size) batches of as near-equal a size as
    # possible -- 10 samples in 3 batches gives 4, 3, 3 -- so every batch respects
    # batch_size, none is much smaller than the rest, and there is no short trailing
    # batch needing a special case. Strata are walked in increasing flat order, so a
    # batch may begin and end part way through one.
    cum = torch.zeros(counts.numel() + 1, dtype=torch.int64)
    torch.cumsum(counts.reshape(-1).to(torch.int64), 0, out=cum[1:])
    total = int(cum[-1])
    if total == 0:
        return
    nbatch = ceil_divide(total, batch_size)
    size, extra = divmod(total, nbatch)
    edges = [0]
    for k in range(nbatch):
        edges.append(edges[-1] + size + (1 if k < extra else 0))
    for begin, end in zip(edges[:-1], edges[1:]):
        # Strata overlapping [begin, end), and how much of each falls inside.
        first = int(torch.searchsorted(cum, torch.tensor(begin, dtype=torch.int64), right=True)) - 1
        last = int(torch.searchsorted(cum, torch.tensor(end - 1, dtype=torch.int64), right=True)) - 1
        inside = (torch.clamp(cum[first + 1:last + 2], max=end)
                  - torch.clamp(cum[first:last + 1], min=begin)).to(torch.int)
        u_flat = torch.repeat_interleave(torch.arange(first, last + 1, dtype=torch.int), inside,
                                         output_size=end - begin)
        yield la.unravel_index(u_flat, counts.shape)


class BetaSampler:
    # The stochastic remainder at one Laplace point: one or more disjoint branches.
    # The exactly evaluated screened block belongs to BetaState, which adds it to the
    # mean; sample() returns the stochastic part alone.
    #
    # The branch index is drawn jointly with (i,j) from marginal, so it is just
    # another sampled index and the branch count only changes a loop bound. There is
    # a single Z, the total over all branches: a sample lands in branch k with
    # probability Z_k/Z, so the value normalization is always the total, never the
    # branch-local mass.
    #
    # Fields: guide (branch-independent tensors, see Guide), samplers_ab (one
    # (a, b) sampler pair per branch) and marginal, shaped (branch, i, j).
    def __init__(self, Rhalf, block_size=None, ov_screen=None, sectors=None, timings=None):
        # Every table and sampler is built here, straight from Rhalf, in dependency
        # order, so nothing is attached after construction. Pass a dict as timings to
        # collect the breakdown; it is deliberately not stored on the object, since a
        # dict of floats in a pytree's aux_data would force a recompile every call.
        #
        # Padding, when requested, is only along x; it is harmless for zero-padded
        # probabilities and lets the auxiliary conditional use block sampling.
        if timings is None:
            timings = {}
        if block_size is not None:
            Rhalf = maybe_pad_last_dim(Rhalf, block_size)

        t_tables = time.perf_counter()
        Aov, Bx, RBov, RBovg = self.build_guide_tables(Rhalf, sectors=sectors, timings=timings)
        t0 = time.perf_counter()
        self.samplers_ab = []
        marginals = []
        for a_table, b_table in self.branch_tables(Aov, RBov, ov_screen):
            self.samplers_ab.append((sample.BatchedAliasSampler(1, a_table),
                                     sample.BatchedAliasSampler(1, b_table)))
            marginals.append(torch.sum(a_table, dim=2) * torch.sum(b_table, dim=2))
        timings['time_table_branch'] = time.perf_counter() - t0
        self.marginal = torch.stack(marginals)
        Z = torch.sum(self.marginal)
        timings['table'] = time.perf_counter() - t_tables

        t_sampler = time.perf_counter()
        self.sampler_x = make_x_sampler(Rhalf, Bx, RBovg, sectors, block_size)
        timings['sampler'] = time.perf_counter() - t_sampler

        self.guide = Guide(Rhalf, Aov, Bx, RBovg, Z, sectors)
        # Split of one sample() call: drawing indices is random access, evaluating
        # them is contiguous. Reset per call so they share a scope with the wall time
        # the caller measures around it; the caller accumulates across calls.
        self.time_draw = 0.0
        self.time_evaluate = 0.0

    @staticmethod
    def build_guide_tables(Rhalf, sectors=None, timings=None):
        # Row/column magnitudes of the loop-broken guide: Aov feeds the b conditional,
        # Bx the auxiliary conditional, RBov the a conditional. With sectors the per-atom
        # sums RBovg are formed first and RBov recovered from them, which replaces the
        # separate nov x naux pass rather than adding one.
        t0 = time.perf_counter()
        Aov = torch.linalg.vector_norm(Rhalf, dim=2)
        Bx = build_Bx(Rhalf)
        if sectors is None:
            RBovg = None
            RBov = build_RBov(Rhalf, Bx)
        else:
            RBovg = build_group_table(Rhalf, Bx, sectors)
            RBov = torch.sum(RBovg, dim=2)
        if timings is not None:
            timings['time_table_ABC'] = time.perf_counter() - t0
        return Aov, Bx, RBov, RBovg

    @staticmethod
    def branch_tables(Aov, RBov, ov_screen):
        # One (a_table, b_table) pair of dense (i,j,v) conditional weights per branch.
        #
        # The screened exact block is "a and b both inside the local domain". The
        # stochastic remainder is the complement, split into two disjoint branches:
        #   0: a outside the domain, b unrestricted
        #   1: a inside the domain,  b outside
        # Unscreened is a single unmasked branch, which is why nothing downstream
        # special-cases it.
        #
        # These are nocc^2 * nvir, well below the nov * naux auxiliary alias table that
        # dominates a beta point, so they are simply materialized. Handing the alias
        # sampler a factorized pair instead would avoid the dense copy where the mask
        # happens to factorize, but saves only a few MB at realistic sizes.
        RBov_outer = RBov[:, None, :] * RBov[None, :, :]
        Aov_outer = Aov[:, None, :] * Aov[None, :, :]
        if ov_screen is None:
            return [(RBov_outer, Aov_outer)]
        screen = ov_screen[:, None, :] & ov_screen[None, :, :]
        return [(RBov_outer * ~screen, Aov_outer),                 # a outside, b unrestricted
                (RBov_outer * screen, Aov_outer * ~screen)]        # a inside,  b outside

    @property
    def Z(self):
        return self.guide.Z

    @property
    def nbranch(self):
        return self.marginal.shape[0]

    def sample_one(self, ibranch, u_i, u_j, count_time=False):
        # The branch is resolved here, in Python, so the compiled kernels never see
        # the branch count and are shared across branches. Grouping changes only the
        # evaluator; the index draw is common to both.
        #
        # count_time is off by default so the warmup pass, whose first call through
        # each kernel pays for compilation, does not pollute the split.
        guide = self.guide
        random_state = sample.get_random_state()
        t0 = time.perf_counter() if count_time else 0.0
        u_abxy = sample_indices(*self.samplers_ab[ibranch], self.sampler_x, u_i, u_j, random_state)
        t1 = time.perf_counter() if count_time else 0.0
        u_ijabxy = (u_i, u_j) + u_abxy
        evaluate = evaluate_group if guide.sectors is not None else evaluate_perx
        value_sum, values = evaluate(guide, u_ijabxy)
        if count_time:
            self.time_draw += t1 - t0
            self.time_evaluate += time.perf_counter() - t1
        return value_sum, values

    def sample_counts(self, ibranch, counts, batch_size=batch_size, count_time=False):
        # The expensive part is the compiled sample/evaluate kernel; everything here
        # is bookkeeping.
        dtype = self.guide.Rhalf.dtype
        value_sum = torch.zeros((), dtype=dtype)
        value_sumsq = torch.zeros((), dtype=dtype)
        for u_i, u_j in make_index_batches(counts, batch_size):
            batch_sum, values = self.sample_one(ibranch, u_i, u_j, count_time=count_time)
            value_sum += batch_sum
            value_sumsq += torch.sum(values ** 2)
        return value_sum, value_sumsq

    def sample(self, nsamples, batch_size=batch_size, count_time=False):
        nsamples = max(int(nsamples), min_nsamples)
        self.time_draw = 0.0
        self.time_evaluate = 0.0
        dtype = self.guide.Rhalf.dtype
        if self.Z == 0:
            zero = torch.zeros((), dtype=dtype)
            return zero, zero
        # Split the request over branches first, so sample_counts never sees a
        # handful of samples. Lifting a small branch to the floor can push the true
        # total slightly above nsamples, which is harmless: variance_of_mean below
        # already uses the per-branch counts actually drawn, and the nsamples that
        # scales it here is divided out again by the caller.
        branch_mass = self.marginal.reshape(self.nbranch, -1).sum(1)
        weight = branch_mass / self.Z
        nper = quota_counts(nsamples, weight)
        nper = torch.where(branch_mass > 0, torch.clamp(nper, min=min_nsamples),
                           torch.zeros_like(nper))

        # A branch lifted to the floor is over-sampled relative to its share, so
        # each branch is weighted by Z_k/Z explicitly instead of relying on n_k
        # being proportional to Z_k. With no lifting this is value_sum / nsamples.
        mean = torch.zeros((), dtype=dtype)
        variance_of_mean = torch.zeros((), dtype=dtype)
        for ibranch in range(self.nbranch):
            n = int(nper[ibranch])
            if n == 0:
                continue
            prob = self.marginal[ibranch].reshape(-1) / branch_mass[ibranch]
            counts = sample.make_marginal_nsamples(n, prob).reshape(self.marginal.shape[1:])
            branch_sum, branch_sumsq = self.sample_counts(ibranch, counts, batch_size=batch_size,
                                                           count_time=count_time)
            # Plain sample variance of the drawn values. The (i,j) quota allocation
            # makes the draws not quite i.i.d., so this slightly overestimates the
            # true variance by the between-stratum term -- about 1%, in the safe
            # direction. The stratified estimator that measured that term needs
            # >~5 samples per stratum and collapses to zero below one, a regime
            # grouping reaches routinely, so it is not worth keeping.
            branch_var = (branch_sumsq - branch_sum**2 / n) / max(n - 1, 1)
            mean = mean + weight[ibranch] * branch_sum / n
            variance_of_mean = variance_of_mean + weight[ibranch]**2 * branch_var / n
        # Report a per-sample sigma, so Var[mean] = sigma^2 / nsamples as both the
        # allocation and the ScreenResult accumulator assume.
        std = torch.sqrt(torch.clamp(variance_of_mean * nsamples, min=0.0))
        return mean, std


class BetaState:
    def __init__(self, R, Focc, Fvir, beta, weight, verbose=0):
        # Per-beta state owns the transformed Rhalf tensor and all quantities
        # derived from it. Keeping beta-local timing here makes verbose=2 useful
        # without passing timing dictionaries through the top-level script.
        #
        # Rhalf already carries weight**0.25. Four Rhalf factors appear in both the
        # direct and exchange contractions, so summing beta contributions of those
        # directly gives the weighted Laplace quadrature.
        t0 = time.perf_counter()
        self.Rhalf = build_laplace_Rhalf(R, beta, weight, Focc, Fvir)
        self.time_transform = time.perf_counter() - t0
        self.beta = beta
        self.weight = weight
        self.verbose = verbose
        self.ov_screen = None
        self.screen = None
        self.sampler = None
        self.timings = {}
        self.info = screen_info(None, self.Rhalf.shape[0], self.Rhalf.shape[1])
        self.time_screen = 0.0
        self.time_table = 0.0
        self.time_sampler = 0.0
        self.time_exact = 0.0
        self.time_full_exact = 0.0
        self.time_direct = 0.0
        self.time_std = 0.0
        self.time_sample = 0.0
        self.time_sample_draw = 0.0
        self.time_sample_evaluate = 0.0
        self.std = None
        self.std_nsamples_total = 0
        self.std_final_nsamples = 0
        self.std_uncertainty = None
        self.mean = None
        self.sample_std = None
        self.nsamples = None
        self.E_full = None
        self.E_direct = None
        self.E_exact_screen = torch.tensor(0.0, dtype=self.Rhalf.dtype)

    def build_screen(self, screen):
        self.screen = screen
        t0 = time.perf_counter()
        # screen=None or screen=0 both mean pure stochastic (no exact domain).
        # A threshold of exactly 0.0 would screen in ALL pairs (Rov > 0 is always
        # True), leaving Z=0 and crashing estimate_std. Treat 0 as None instead.
        if screen is None or screen == 0:
            self.ov_screen = None
        else:
            self.ov_screen = build_ov_screen(self.Rhalf, screen)
        self.info = screen_info(self.ov_screen, self.Rhalf.shape[0], self.Rhalf.shape[1])
        self.time_screen = time.perf_counter() - t0
        return self

    def build_sampler(self, block_size=None, sectors=None):
        self.timings = {}
        # Drop the previous screen's sampler first: plain assignment evaluates the
        # right side before rebinding, so the old alias tables would stay alive
        # while the new ones are built. They go as nocc^2 * nvir, which is tens of
        # GB on the larger flakes.
        self.sampler = None
        self.sampler = BetaSampler(self.Rhalf, block_size=block_size, ov_screen=self.ov_screen,
                                   sectors=sectors, timings=self.timings)
        self.time_table = self.timings['table']
        self.time_sampler = self.timings['sampler']
        # The exact screened block belongs here, not to the sampler. Padding in
        # BetaSampler is only along x and adds zero columns, so the unpadded
        # Rhalf gives an identical contraction.
        t0 = time.perf_counter()
        if self.ov_screen is None:
            self.E_exact_screen = torch.zeros((), dtype=self.Rhalf.dtype)
        else:
            self.E_exact_screen = exchange_screen_from_R(self.Rhalf, self.ov_screen)
        self.time_exact = time.perf_counter() - t0
        return self

    def exact_exchange(self):
        if self.E_full is None:
            t0 = time.perf_counter()
            self.E_full = exchange_from_R(self.Rhalf)
            self.time_full_exact = time.perf_counter() - t0
        return self.E_full

    def direct_mp2(self):
        # Direct MP2 has no sign problem in this representation, so we compute
        # it deterministically once per beta and never include it in the sample
        # variance or sample allocation.
        if self.E_direct is None:
            t0 = time.perf_counter()
            self.E_direct = direct_from_R(self.Rhalf)
            self.time_direct = time.perf_counter() - t0
        return self.E_direct

    def estimate_std(self):
        # Short-circuit: if the sampler has no stochastic part (Z=0, e.g. the
        # exchange is fully evaluated exactly), the stochastic variance is zero.
        if self.sampler.Z == 0:
            self.std = torch.tensor(0.0, dtype=self.Rhalf.dtype)
            self.std_final_nsamples = 0
            self.std_nsamples_total = 0
            self.std_uncertainty = 0.0
            return self.std

        def estimate_variance(nsamples):
            variances = []
            for irun in range(std_nrun):
                mean_beta, std_beta = self.sampler.sample(nsamples, count_time=False)
                variances.append(std_beta**2)
            return torch.stack(variances)

        t0 = time.perf_counter()
        var_beta, nsamples_estimation, nsamples_cumulative, uncertainty = get_average_until_convergence(
            estimate_variance, std_tol, pilot_nsamples, std_nrun,
        )
        self.time_std = time.perf_counter() - t0
        self.std = torch.sqrt(var_beta)
        self.std_final_nsamples = nsamples_estimation * std_nrun
        self.std_nsamples_total = nsamples_cumulative
        self.std_uncertainty = uncertainty
        return self.std

    def STC_exchange(self, nsamples):
        t0 = time.perf_counter()
        mean, self.sample_std = self.sampler.sample(nsamples, count_time=True)
        self.nsamples = max(int(nsamples), min_nsamples)
        self.mean = self.E_exact_screen + mean
        self.time_sample = time.perf_counter() - t0
        self.time_sample_draw = self.sampler.time_draw
        self.time_sample_evaluate = self.sampler.time_evaluate
        return self.mean, self.sample_std

    def warmup_compile(self):
        self.sampler.sample(warmup_nsamples, count_time=False)


def screen_label(screen):
    if screen is None:
        return 'None'
    return f'{screen:.1e}'


class ScreenResult:
    # Running totals for one screening threshold, folded in one beta point at a
    # time. Purely an accumulator: it does not decide the order beta points are
    # visited in, so the same object serves a screen-outer and a beta-outer loop.
    def __init__(self, screen):
        self.screen = screen
        self.energy = 0.0
        self.energy_screen = 0.0
        self.variance = 0.0
        self.nsamples_total = 0
        self.nvir_per_occ = []
        self.time_build_screen = 0.0
        self.time_table = 0.0
        self.time_sampler = 0.0
        self.time_exact = 0.0
        self.time_std = 0.0
        self.time_sample = 0.0
        self.time_sample_draw = 0.0
        self.time_sample_evaluate = 0.0

    def add(self, state, verbose=0):
        # state must already have been through build_screen / build_sampler /
        # estimate_std / STC_exchange for this screen. Reporting lives here so both
        # loop orders emit the same per-beta line.
        nsamples = state.nsamples
        self.energy = self.energy + state.mean
        self.energy_screen = self.energy_screen + state.E_exact_screen
        self.variance = self.variance + state.sample_std**2 / nsamples
        self.nsamples_total += nsamples
        self.nvir_per_occ.append(state.info[3])
        self.time_build_screen += state.time_screen
        self.time_table += state.time_table
        self.time_sampler += state.time_sampler
        self.time_exact += state.time_exact + state.time_full_exact
        self.time_std += state.time_std
        self.time_sample += state.time_sample
        # Each phase total splits into the two kernels; the remainder is the
        # batching and accumulation around them.
        self.time_sample_draw += state.time_sample_draw
        self.time_sample_evaluate += state.time_sample_evaluate
        if verbose >= 2:
            print(f'beta {state.beta:8.4f}  screen {screen_label(self.screen):>8s}  '
                  f'nsamples {nsamples:8.3e}  nvir_per_occ {state.info[3]:8.3f}  '
                  f'direct {state.time_direct:10.4f}  '
                  f'exact {state.time_exact:10.4f}  table {state.time_table:10.4f}  '
                  f'std {state.time_std:10.4f}  prod {state.time_sample:10.4f}  '
                  f'draw {state.time_sample_draw:10.4f}  evaluate {state.time_sample_evaluate:10.4f}', flush=True)

    @property
    def nvir_per_occ_avg(self):
        return float(np.mean(self.nvir_per_occ))

    @property
    def std(self):
        return torch.sqrt(self.variance)

    @property
    def time_exchange(self):
        return (self.time_build_screen + self.time_table + self.time_sampler
                + self.time_exact + self.time_sample)


def allocate_samples(stds, target_error):
    # Optimal allocation for independent beta estimators at equal cost:
    # n_beta ~ Sigma * sigma_beta. Needs every sigma before it can size any beta,
    # which is what forces the screen-outer loop when no proxy is supplied.
    stds = torch.stack(list(stds))
    counts = torch.ceil(torch.sum(stds) * stds / target_error**2).to(torch.int64)
    return [int(n) for n in torch.clamp(counts, min=min_nsamples)]


def allocate_from_targets(std, target_variance):
    # Per-beta variance target from an a-priori proxy: the beta only needs its own
    # sigma, so it can be sized during a single visit.
    return max(int(np.ceil((std**2 / target_variance).item())), min_nsamples)


class STCLaplaceMP2:
    def __init__(self, R, Focc, Fvir, eocc, evir, verbose=0, sectors=None):
        # Driver for all beta points. It intentionally owns only coarse workflow
        # operations; the beta-specific objects contain the actual tables and
        # samplers. R is always xia; only the per-beta Rhalf it produces is iax.
        self.R = R
        self.Focc = Focc
        self.Fvir = Fvir
        self.eocc = eocc
        self.evir = evir
        self.verbose = verbose
        self.sectors = sectors
        self.beta_weight = None
        # Totals over beta for the parts that do not depend on the screen. Per-screen
        # totals live in ScreenResult; the caller owns the loop that fills both.
        self.time_transform = 0.0
        self.time_direct = 0.0
        self.time_warmup_compile = 0.0
        self.warmup_done = False

    def screen_value(self, screen):
        return screen_label(screen)

    def get_beta_list(self, M=8, a=a, minimax=False):
        Delta_min = (2 * torch.min(self.evir) - 2 * torch.max(self.eocc)).item()
        Delta_max = (2 * torch.max(self.evir) - 2 * torch.min(self.eocc)).item()
        if minimax:
            self.beta_weight, error = libquad.get_minimax_quadrature(Delta_min, Delta_max, M)
            if self.verbose >= 1:
                print_value('quadrature_range', fe(Delta_max / Delta_min), flush=True)
                # Absolute error of the 1/Delta fit; the energy error is this weighted
                # by the (positive) numerators, so it is an upper bound on the bias.
                print_value('quadrature_minimax_error', fe(error), flush=True)
        else:
            self.beta_weight = libquad.get_gauss_legendre_quadrature(Delta_min, M, a)
        return self.beta_weight

    def warmup_compile(self, screen, block_size=None):
        # Warm up on one representative beta point so production timings mostly
        # measure sampling/evaluation instead of first-use torch compilation.
        if self.warmup_done:
            return
        t0 = time.perf_counter()
        beta, coeff = self.beta_weight[-1]
        beta_state = self.make_beta_state(beta, coeff)
        beta_state.build_screen(screen)
        beta_state.build_sampler(block_size=block_size, sectors=self.sectors)
        beta_state.warmup_compile()
        self.time_warmup_compile = time.perf_counter() - t0
        self.warmup_done = True
        if self.verbose >= 1:
            print_time_value('time_warmup_compile', self.time_warmup_compile, flush=True)

    def make_beta_state(self, beta, coeff):
        # Transient per-beta state. The caller must drop the reference as soon as
        # the beta point is finished; that is what keeps peak memory at one Rhalf
        # plus one sampler instead of M of each.
        return BetaState(self.R, self.Focc, self.Fvir, beta, coeff, verbose=self.verbose)

    def beta_std_proxy(self, p):
        # Cheap a-priori estimate of the relative sigma_beta profile, needing only
        # orbital energies: sigma_beta ~ weight_beta * (Tr exp(beta Focc) *
        # Tr exp(-beta Fvir))**p. The trace is invariant under the occupied and
        # virtual rotations used for localization, so the canonical eigenvalues
        # give the same value as expm of the localized Fock blocks. Empirically
        # p ~ 1.4 reproduces the measured allocation to a few percent in the
        # optimal-allocation metric. The overall scale is irrelevant: only ratios
        # are used.
        dtype = self.eocc.dtype
        proxy = []
        for beta, weight in self.beta_weight:
            trace_occ = torch.sum(torch.exp(beta * self.eocc))
            trace_vir = torch.sum(torch.exp(-beta * self.evir))
            proxy.append(weight * (trace_occ * trace_vir)**p)
        return torch.stack(proxy).to(dtype)
