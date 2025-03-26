import dill
import numpy as np
import pickle
import qiskit as qk
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from typing import Tuple

from example_circuits_to_synthesize.TFIM_ham_gen import construct_hamiltonian

import dreamcoder as dc
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.fragmentGrammar import FragmentGrammar
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.program import Abstraction
from dreamcoder.task import Task
from dreamcoder.utilities import numberOfCPUs
import dreamcoder.domains.quantum_ground_state.primitives as pr
from dreamcoder.domains.quantum_ground_state.primitives import (
    mat_contraction,
    mat_to_tensor,
    circuit_to_mat,
    tensor_contraction,
    execute_program,
    execute_quantum_algorithm,
    normalize_unitary,
    get_qiskit_circuit,
    tcircuit,
    no_op,
    QiskitTester,
)
from dreamcoder.program import Program


def get_energy(psi, H):
    return np.dot(np.dot(psi, H), np.conj(psi))

class GroundStateTask(Task):
    last_circuit = ""
    last_circuit_evaluation = {}

    def __init__(self, name, hamiltonian, arguments,request):
        # NOTE: this is only used for library building, cannot be used for enumeration if arguments are fixed
        self.hamiltonian = hamiltonian
        self.arguments = arguments
        self.n_qubits = int(np.log2(len(hamiltonian)))
        # request = dc.type.arrow(*([dc.type.tint] * self.n_qubits), tcircuit, tcircuit)
        example = (
            *range(self.n_qubits),
            no_op(self.n_qubits),
        ), (self.hamiltonian,)
        super().__init__(
            name=name,
            request=request,
            examples=[example],
            features=[],
        )

    def logLikelihood(self, e: Program, timeout=None):
        if type(e) == str:
            e = Program.parse(e)

        if GroundStateTask.last_circuit is not e:
            GroundStateTask.last_circuit = e
            GroundStateTask.last_circuit_evaluation = None

        if GroundStateTask.last_circuit_evaluation is None:
            GroundStateTask.last_circuit_evaluation = execute_quantum_algorithm(
                e, self.n_qubits, timeout, arguments=self.arguments
            )

        yh = GroundStateTask.last_circuit_evaluation
        # print(e, self.n_qubits, timeout, self.arguments)
        # TO DISABLE CACHING (for testing):
        yh =  execute_quantum_algorithm(e, self.n_qubits, timeout,arguments=self.arguments)


        psi0 = Statevector.from_int(0, 2**self.n_qubits)
        psi1 = np.dot(yh,psi0)

        energy = np.real(get_energy(psi1, self.hamiltonian))


        if yh is None:
            return dc.utilities.NEGATIVEINFINITY

        try:
            # if np.any(np.abs(yh-yh_true) >= 1e-3):
            #     return dc.utilities.NEGATIVEINFINITY
            return -energy  # TODO: very naive (be careful)

            return 0.0  # That's the maximum

        except ValueError:
            return dc.utilities.NEGATIVEINFINITY