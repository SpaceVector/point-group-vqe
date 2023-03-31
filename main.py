from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, TimeEvolution
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import Transform, get_qubit_hamiltonian
import mindspore as ms
from mindquantum.framework import MQAnsatzOnlyLayer
from timeit import default_timer
from pyscf import gto, scf, symm
from SymUCCSD import uccsd_singlet_generator


ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
geometry = [
    ["C", [0.0, 0.0, 0.653]],
    ["C", [0.0, 0.0, -0.653]],
    ["H", [0.0, 0.916, 1.229]],
    ["H", [0.0, -0.916, 1.229]],
    ["H", [0.0, -0.916, -1.229]],
    ["H", [0.0, 0.916, -1.229]]
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
molecule_pqcnet = MQAnsatzOnlyLayer(molecule_pqc, 'Zeros')
initial_energy = molecule_pqcnet()

optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

eps = 1.e-8
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff

iter_idx = 0

while abs(energy_diff) > eps:
    energy_i = train_pqcnet().asnumpy()
    # if iter_idx % 5 == 0:
    #     print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1
end_time = default_timer()    # 程序结束时间
run_time = end_time - start_time    # 程序的运行时间，单位为秒
print("Molecule: ", molecule_of.name)
print("Number of parameters: %d" % (len(ansatz_parameter_names)))
print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Runtime: ", run_time,'s')

