import os
import functools
import numpy as np
import pyscf
import pyscf.lib
import pyscf.df
import pyscf.pbc.df

df_eig = False


def wrap(func, **fixed_kwargs):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        kwargs.update(fixed_kwargs)
        return func(*args, **kwargs)
    return wrapped


def matrix_operation(A, func):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = func(eigvals)
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


def make_df_eig():
    global df_eig
    if df_eig:
        return

    def _eig_decompose(dev, j2c, lindep=pyscf.df.incore.LINEAR_DEP_THR):
        return matrix_operation(j2c, lambda x: 1 / np.sqrt(x))

    def eigenvalue_decomposed_metric(self, j2c):
        result = matrix_operation(j2c, lambda x: 1 / np.sqrt(x))
        return result, None, 'ED'

    pyscf.df.incore.cholesky_eri = wrap(pyscf.df.incore.cholesky_eri, decompose_j2c='eig')
    pyscf.df.incore._eig_decompose = _eig_decompose
    pyscf.pbc.df.rsdf_builder._RSGDFBuilder.j2c_eig_always = True
    pyscf.pbc.df.rsdf_builder._RSGDFBuilder.eigenvalue_decomposed_metric = eigenvalue_decomposed_metric
    df_eig = True


def get_basis(basis):
    '''
    cc-pvtz -> {'default': cc-pvtz}
    cc-pvtz,H:6-31g -> {'default': 'cc-pvtz', 'H': '6-31g'}
    '''
    basis_list = basis.split(',')
    basis_all = {'default': basis_list[0]}
    for string in basis_list[1:]:
        key, value = string.split(':')
        basis_all[key] = value
    return basis_all


def do_scf(mol_or_cell, df=True, save_df=False, dir=None, type='RHF'):
    is_pbc = (not isinstance(mol_or_cell, pyscf.gto.Mole))
    scf_mod = pyscf.pbc.scf if is_pbc else pyscf.scf
    type = type.upper()
    scf_method = getattr(scf_mod, type)
    mf = scf_method(mol_or_cell)
    if df:
        mf = mf.density_fit()
    mf.verbose = 4
    if save_df:
        assert dir is not None
    cderi_path = f"{dir}/cderi.h5" if save_df else None
    mf.with_df = get_DF(mol_or_cell, save_path=cderi_path)
    if dir is not None:
        chkfile = f"{dir}/{type}.chk"
        if os.path.exists(chkfile):
            _, attrs = pyscf.scf.chkfile.load_scf(chkfile)
            for key, value in attrs.items():
                setattr(mf, key, value)
        else:
            mf.chkfile = chkfile
            mf.scf()
    else:
        mf.scf()
    return mf


def get_DF(mol_or_cell, save_path=None):
    import time
    is_pbc = (not isinstance(mol_or_cell, pyscf.gto.Mole))
    if is_pbc:
        with_df = pyscf.pbc.df.GDF(mol_or_cell)
    else:
        with_df = pyscf.df.DF(mol_or_cell)
    begin = time.time()
    if save_path is not None:
        if os.path.exists(save_path):
            with_df._cderi = save_path
        else:
            with_df._cderi_to_save = save_path
            with_df.build()
    else:
        with_df.build()
    end = time.time()
    print('DF build time', end - begin)
    return with_df


def get_DF_tensor(with_df, transpose=False):
    is_pbc = hasattr(with_df, 'cell')
    if is_pbc:
        generators = (item for item, _, _ in with_df.sr_loop())
    else:
        generators = (item for item in with_df.loop())
    tensor = np.concatenate([pyscf.lib.unpack_tril(item) for item in generators])
    if transpose:
        tensor = np.moveaxis(tensor, 0, -1).copy()
    return tensor


def get_ncore(mol):
    atoms = mol.atom_charges()
    ncores = 0
    for atom in atoms:
        if atom > 36:
            assert False, "cannot deal with large molecule"
        elif atom > 18:
            ncores += 9
        elif atom > 10:
            ncores += 5
        elif atom > 2:
            ncores += 1
    return ncores


def localize_atomic(mol, C, spinor=False):
    '''Atomic-orbital initial guess for localization: a custom reimplementation of
    pyscf.lo.boys.atomic_init_guess. Projects the MOs onto the orthonormal
    (meta-Lowdin) AO basis, keeps the nmo AOs the MOs overlap most, and rotates
    the MOs onto them via the polar factor u @ vh. Uses scipy's gesvd SVD driver
    (QR-based) instead of numpy's default gesdd (divide-and-conquer), which fails
    to converge on clustered/degenerate singular values -- notably the spin-
    doubled singular values of the GHF spinor case. The returned polar factor is
    well defined even under exact degeneracy, so only the driver robustness matters.

    spinor=True: C is a (2*nao, nmo) GHF spinor set (rows [:nao] the alpha-spin AO
    block, [nao:] the beta-spin block). The orthonormal spinor-AO basis is block-
    diagonal blkdiag(c, c), so the MO projection is the vstack of the alpha- and
    beta-block spatial projections, and the AO selection runs over the 2*nao
    (atom, spin) spin-AOs.'''
    import scipy.linalg
    import pyscf.lo.orth
    if getattr(mol, 'pbc_intor', None):
        s = mol.pbc_intor('int1e_ovlp', hermi=1)
    else:
        s = mol.intor_symmetric('int1e_ovlp')
    c = pyscf.lo.orth.orth_ao(mol, s=s)            # (nao, nao) orthonormal AOs
    csC = c.conj().T @ s
    if spinor:
        nao = s.shape[0]
        mo = np.vstack([csC @ C[:nao], csC @ C[nao:]])   # (2*nao, nmo)
    else:
        mo = csC @ C                                     # (nao, nmo)
    nmo = mo.shape[1]
    idx = np.argsort(np.einsum('pi,pi->p', mo.conj(), mo))
    idx = sorted(idx[-nmo:])                             # nmo most-overlapping (atom[,spin]) AOs
    u, w, vh = scipy.linalg.svd(mo[idx], lapack_driver='gesvd')
    U = (u @ vh).conj().T
    return C @ U


def sort_by_ao(C):
    argmax = np.argmax(np.abs(C), axis=0)
    return C[:, np.argsort(argmax)]


class OrbBasis:
    def __init__(self, rhf, locmethod='PM', frozen_core=False):
        import pyscf.lo
        self.frozen_core = frozen_core
        if frozen_core:
            self.ncore = get_ncore(rhf.mol)
        else:
            self.ncore = 0
        self.nocc = np.sum(rhf.mo_occ > 0.5) - self.ncore
        self.nvir = np.sum(rhf.mo_occ < 0.5)
        self.nmo = self.nocc + self.nvir
        self.o = slice(self.ncore, self.ncore + self.nocc)
        self.v = slice(self.ncore + self.nocc, self.ncore + self.nocc + self.nvir)
        self.eocc = rhf.mo_energy[self.o]
        self.evir = rhf.mo_energy[self.v]
        self.Cocc = rhf.mo_coeff[:, self.o]
        self.Cvir = rhf.mo_coeff[:, self.v]
        self.C = np.concatenate([self.Cocc, self.Cvir], axis=1)
        self.locmethod = locmethod
        self.loc_func = getattr(pyscf.lo, locmethod)
        self.Cocc_local = self.loc_func(rhf.mol, self.Cocc).kernel()
        self.Cvir_local = self.loc_func(rhf.mol, self.Cvir).kernel()
        self.Cocc_local = sort_by_ao(self.Cocc_local)
        self.Cvir_local = sort_by_ao(self.Cvir_local)
        self.C_local = np.concatenate([self.Cocc_local, self.Cvir_local], axis=1)
        if hasattr(rhf, 'cell'):
            self.S = rhf.mol.pbc_intor('int1e_ovlp')
        else:
            self.S = rhf.mol.intor('int1e_ovlp')
        self.Uocc = self.Cocc_local.T @ self.S @ self.Cocc
        self.Uvir = self.Cvir_local.T @ self.S @ self.Cvir
        self.U = np.zeros((self.nmo, self.nmo))
        self.U[:self.nocc, :self.nocc] = self.Uocc
        self.U[self.nocc:, self.nocc:] = self.Uvir
        assert np.allclose(self.Uocc @ self.Uocc.T, np.eye(self.nocc))
        assert np.allclose(self.Uvir @ self.Uvir.T, np.eye(self.nvir))
