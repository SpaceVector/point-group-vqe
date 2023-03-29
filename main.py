from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import generate_uccsd
import mindspore as ms
from mindquantum.framework import MQAnsatzOnlyLayer
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer


ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
geometry = [
    ["O", [0.0, 0.0, 0.137]],
    ["H", [0.0, 0.769, -0.546]],
    ["H", [0.0, -0.769, -0.546]]
]
basis = "sto3g"
spin = 0

molecule_of = MolecularData(
    geometry,
    basis,
    multiplicity=2 * spin + 1
)
molecule_of = run_pyscf(
    molecule_of,
    run_fci=1,
    run_ccsd=1
)

hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(molecule_of.n_electrons)])
ansatz_circuit, init_amplitudes, ansatz_parameter_names, hamiltonian_QubitOp, n_qubits, n_electrons = generate_uccsd(molecule_of, threshold=-1)
total_circuit = hartreefock_wfn_circuit + ansatz_circuit

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

