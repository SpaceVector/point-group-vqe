from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, down_index, up_index
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import generate_uccsd,uccsd_singlet_generator
import mindspore as ms
from mindquantum.framework import MQAnsatzOnlyLayer
from timeit import default_timer
from pyscf import gto, scf, symm
import numpy as np
from itertools import product, combinations
import copy


ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
geometry = [
    ["Be", [0.0, 0.0, 0.0]],
    ["H", [0.0, 0.0, 1.291]],
    ["H", [0.0, 0.0, -1.291]]
]
basis = "sto3g"
spin = 0

molecule_of = MolecularData(
    geometry,
    basis,
    multiplicity=2 * spin + 1
)
molecule_of = run_pyscf(
    molecule_of
)
mol = gto.Mole()
mol.atom = molecule_of.geometry
mol.basis = molecule_of.basis
mol.spin = molecule_of.multiplicity - 1
mol.charge = molecule_of.charge
mol.symmetry = "D2h"
mol.build()
mf = scf.RHF(mol)
mf.kernel()
print(mf.mo_occ)
print(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff))
# 被占据轨道数
n_occupied = mol.nelec[0]
# 总轨道数
n_virtual = mf.mo_coeff.shape[1]-n_occupied

j = 0


# 计算所有可能的单双电子激发
for i, (p, q) in enumerate(product(range(n_virtual), range(n_occupied))):
    # Get indices of spatial orbitals
    virtual_spatial = n_occupied + p
    occupied_spatial = q
    mo_occ = copy.deepcopy(mf.mo_occ)
    mo_occ[occupied_spatial] -= 1
    mo_occ[virtual_spatial] += 1
    mo_occ = copy.deepcopy(mf.mo_occ)
    mo_occ[occupied_spatial] -= 2
    mo_occ[virtual_spatial] += 2    

#构造所有可能的双电子激发
for i, ((p, q), (r, s)) in enumerate(combinations(product(range(n_virtual), range(n_occupied)), 2)):
    virtual_spatial_1 = n_occupied + p
    occupied_spatial_1 = q
    virtual_spatial_2 = n_occupied + r
    occupied_spatial_2 = s
    mo_occ = copy.deepcopy(mf.mo_occ)
    mo_occ[occupied_spatial_1] -= 1
    mo_occ[occupied_spatial_2] -= 1
    mo_occ[virtual_spatial_1] += 1
    mo_occ[virtual_spatial_2] += 1

print(mf.mo_occ)




