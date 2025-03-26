import dreamcoder as dc
from dreamcoder.domains.quantum_circuits.primitives import *
from dreamcoder.domains.quantum_circuits.primitives import circuit_to_mat
import numpy as np
import time

import dill
import numpy as np
import pickle
import qiskit as qk
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from typing import Tuple

import dreamcoder as dc
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.fragmentGrammar import FragmentGrammar
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.program import Abstraction
from dreamcoder.task import Task
from dreamcoder.utilities import numberOfCPUs
import dreamcoder.domains.quantum_circuits.primitives as pr
from dreamcoder.domains.quantum_circuits.primitives import (
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
    QuantumCircuitException
)
from dreamcoder.program import Program

class QuantumTask(dc.task.Task):
    last_circuit = ""
    last_circuit_evaluation = {}

    def __init__(self, name, target_circuit=None, target_unitary=None):
        if target_circuit is not None and target_unitary is not None:
            raise Exception("Either provide target_circuit or target_unitary, not both")
        elif target_circuit is not None:
            self.n_qubits, self.target_circuit = target_circuit
            self.target_circuit_evaluation = circuit_to_mat(target_circuit)
        elif target_unitary is not None:
            self.n_qubits = int(np.log2(len(target_unitary)))
            self.target_circuit_evaluation = target_unitary

        super(QuantumTask, self).__init__(name=name,
                                          request=dc.type.arrow(*([dc.type.tint]*self.n_qubits), tcircuit, tcircuit),
                                          examples=[((*range(self.n_qubits), no_op(self.n_qubits),), (self.target_circuit_evaluation,),)],
                                          features=[])

    def logLikelihood(self, e:dc.program, timeout=None):
        if type(e) == str:
            e = dc.program.Program.parse(e)
            
        if QuantumTask.last_circuit is not e:
            QuantumTask.last_circuit = e
            QuantumTask.last_circuit_evaluation = None

        if QuantumTask.last_circuit_evaluation is None:
            QuantumTask.last_circuit_evaluation = execute_quantum_algorithm(e, self.n_qubits, timeout)

        yh_true = self.target_circuit_evaluation

        yh = QuantumTask.last_circuit_evaluation

        # TO DISABLE CACHING (for testing):
        # yh =  execute_quantum_algorithm(e, self.n_qubits, timeout)

        if yh is None:
            return dc.utilities.NEGATIVEINFINITY

        try:
            if np.any(np.abs(yh-yh_true) >= 1e-3):
                return dc.utilities.NEGATIVEINFINITY
        except ValueError:
            return dc.utilities.NEGATIVEINFINITY
        return 0.





def enumerate_pcfg(pcfg, timeout,
                   observational_equivalence=True,
                   sound=False): 
    enum_dictionary = {}
    t_0 = time.time()
    
    n_qubit = dc.domains.quantum_circuits.primitives.GLOBAL_NQUBIT_TASK
    for code in pcfg.quantized_enumeration(observational_equivalence=observational_equivalence,
                                           inputs=[(*range(n_qubit), no_op(n_qubit),)],
                                           sound=sound):
        if (time.time()>t_0+timeout): break
        # check if it is a valid circuit
        try: 
            arguments = (*range(n_qubit),no_op(n_qubit))
            circuit = execute_program(code, arguments )
            unitary = circuit_to_mat(circuit)
            
            key = dc.domains.quantum_circuits.primitives.hash_complex_array(unitary)
            task = str(code)
            c_time = time.time()
            # if "rep" in str(code):
            #     eprint("YES. There is one!")
            # If multiple programs give the same unitary
            # we want to keep the simplest one
            if key not in enum_dictionary:
                enum_dictionary[key]={"code":code, "circuit":circuit, "time": c_time-t_0}
        except QuantumCircuitException as e:
            ...
    dc.utilities.eprint(f"Enumerated {len(enum_dictionary)} programs")
    return enum_dictionary


# def makeTasks(task_enumeration_timeout=6):
#     pcfg_full = dc.grammar.PCFG.from_grammar(full_grammar, request=dc.type.arrow(
#         *[dc.type.tint]*dc.domains.quantum_circuits.primitives.GLOBAL_NQUBIT_TASK,
#         tcircuit, tcircuit))
#     tasks = dc.enumeration.enumerate_pcfg(pcfg_full,
#                                           timeout=task_enumeration_timeout,
#                                           observational_equivalence=True,
#                                           sound=True)

#     quantumTasks = []
#     for idx, task in enumerate(tasks.values()):
#         quantumTasks.append(QuantumTask(f"t_{idx:03d}_{task['code']}", task["circuit"]))
#     return quantumTasks

# ROTATION TASKS
# TODO: generalize task selection
def makeTasks(task_enumeration_timeout=None):
    quantumTasks = []
    for idx in range(5500):
        x,y,z = np.random.uniform(0,2*np.pi,3)
        with QiskitTester(1) as QT:
            QT.circuit.u(x,y,z,0)

        mat = QT.get_result(QT.circuit)
        mat = normalize_unitary(mat)            
        task = QuantumTask(f"t_{idx:03d}_{x}_{y}_{z}_(())", target_unitary=mat )
            
        quantumTasks.append(task)
    return quantumTasks


def get_task_from_name(name, tasks):
    for task in tasks:
        if task.name == name:
            return task
    else:
        raise Exception("Task not found")
