from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, TimeEvolution
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import Transform, get_qubit_hamiltonian
from timeit import default_timer
from pyscf import gto, scf, symm
from SymUCCSD import uccsd_singlet_generator
import mindquantum
from scipy.optimize import minimize
import numpy as np


geometry = [
    # ["C", [0.0, 0.0, 0.653]],
    # ["C", [0.0, 0.0, -0.653]],
    # ["H", [0.0, 0.916, 1.229]],
    # ["H", [0.0, -0.916, 1.229]],
    # ["H", [0.0, -0.916, -1.229]],
    # ["H", [0.0, 0.916, -1.229]]
    # ["O", [0.0, 0.0, 0.127]],
    # ["H", [0.0, 0.758, -0.509]],
    # ["H", [0.0, -0.758, -0.509]]
    # ["Be", [0.0, 0.0, 0.0]],
    # ["H", [0.0, 0.0, 1.291]],
    # ["H", [0.0, 0.0, -1.291]]
    # ["Li",[0.0, 0.0, 0.378]],
    # ["H", [0.0, 0.0, -1.133]]
    # ["F",[0.0, 0.0, 0.096]],
    # ["H", [0.0, 0.0, -0.860]]
    ["C", [0.0, 0.0, 0.0]],
    ["H", [0.625, 0.625, 0.625]],
    ["H", [-0.625, -0.625, 0.625]],
    ["H", [-0.625, 0.625, -0.625]],
    ["H", [0.625, -0.625, -0.625]]
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
mol.symmetry = "D2"
mol.build()
mf = scf.RHF(mol)
mf.kernel()
mo_occ = mf.mo_occ #轨道表示数组
irrep_id = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff)
# 被占据轨道数
n_occupied = mol.nelec[0]
# 总轨道数
n_virtual = mf.mo_coeff.shape[1]-n_occupied

ucc_fermion_ops = uccsd_singlet_generator(n_virtual, n_occupied, irrep_id, mo_occ, anti_hermitian=True)

hamiltonian_QubitOp = get_qubit_hamiltonian(molecule_of)

ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
ansatz_parameter_names = ansatz_circuit.params_name
hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(molecule_of.n_electrons)])
total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()

start_time = default_timer()    # 程序开始时间
sim = Simulator('mqvector', total_circuit.n_qubits)
molecule_pqc = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_QubitOp), total_circuit)

iter_idx = 1
def func(x):
    n,g = molecule_pqc(x)
    global iter_idx
    if iter_idx % 3 == 0:
        end_time = default_timer()    # 程序结束时间
        run_time = end_time - start_time    # 程序的运行时间，单位为秒
        print('第',iter_idx,'次迭代，当前能量为：', np.real(np.squeeze(n)),'当前用时：',run_time)
    iter_idx += 1
    return np.real(np.squeeze(n)), np.real(np.squeeze(g))


init_amp = [0.0]*len(ansatz_parameter_names)
res = minimize(func, init_amp, jac=True, method='TNC', tol=0.0015)

end_time = default_timer()    # 程序结束时间
run_time = end_time - start_time    # 程序的运行时间，单位为秒
print("Molecule: ", molecule_of.name)
print("Number of parameters: %d" % (len(ansatz_parameter_names)))
print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (res.fun))
print("Runtime: ", run_time,'s')

