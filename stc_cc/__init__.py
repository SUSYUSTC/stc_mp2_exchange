# Subset of the stc_cc library needed by the STC-MP2 exchange code: the sampling
# machinery (sample, alias_numba), the tensor helpers (la, utils), the PySCF
# interface (utils_pyscf) and the orbital localizers (orbopt). The full library,
# which additionally implements STC-CCSD(T), is described in Ref. [1] of the paper.
import torch
torch.set_default_dtype(torch.float64)
torch.set_default_device('cpu')
torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.assert_indirect_indexing = False
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024
from .utils import enable_profile, disable_profile, set_num_threads
from . import utils, la, sample, orbopt, utils_pyscf
