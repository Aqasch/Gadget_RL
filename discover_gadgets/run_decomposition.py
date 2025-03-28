#! /bin/env python

import warnings
warnings.filterwarnings('ignore')

import argparse
import dill
from datetime import datetime
import pickle
import time

from qiskit import *
from example_circuits_to_synthesize.TFIM_ham_gen import construct_hamiltonian

import dreamcoder as dc
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.fragmentGrammar import FragmentGrammar
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.utilities import numberOfCPUs
import dreamcoder.domains.quantum_ground_state.primitives as pr
from dreamcoder.domains.quantum_ground_state.primitives import (
    get_instructions_from_qiskit,
    get_code_from_instructions,
)
from dreamcoder.domains.quantum_ground_state.tasks import GroundStateTask
from dreamcoder.program import Program
from dreamcoder.utilities import eprint

from qiskit.converters import circuit_to_dag, dag_to_circuit
def trimmed_circuit(qc, max_depth):
    """
    Trim a quantum circuit to a maximum depth using its DAG representation.

    Args:
        qc (QuantumCircuit): The input quantum circuit.
        max_depth (int): The maximum allowed depth.

    Returns:
        QuantumCircuit: The trimmed quantum circuit.
    """
    # Convert the circuit to a DAG representation
    dag = circuit_to_dag(qc)
    
    # Create a new empty DAG to store the trimmed circuit
    trimmed_dag = dag.copy_empty_like()
    
    # Iterate through layers of the DAG
    layers = list(dag.layers())
    for i, layer in enumerate(layers):
        if i >= max_depth:
            break  # Stop adding layers once we reach `max_depth`
        for node in layer['graph'].nodes():
            # Check if the node is an operation node (DAGOpNode)
            if isinstance(node, dag.op_nodes().__iter__().__next__().__class__):  # Dynamically check for DAGOpNode type
                trimmed_dag.apply_operation_back(node.op, node.qargs, node.cargs)
    
    # Convert the trimmed DAG back to a QuantumCircuit
    trimmed_circuit = dag_to_circuit(trimmed_dag)
    
    return trimmed_circuit

# Prompt user for input
print("Do you want to:")
print("1. Load circuits RL cirtuits from library?")
print("2. Test the program synthesis with already saved circuits?")
choice = input("Enter 1 or 2: ")

# Initialize variables based on user choice
if choice == "1":
    # Load circuits from saved library
    print()
    print('-x-x-x-x-x-x-x-x-x-x-x-')
    n_qubits = input("Enter the number of qubits: ")
    J_value = 1
    print()
    print('-x-x-x-x-x-x-x-x-x-x-x-')
    print("Did you already extracted best circuits using GRL?")
    print("1. Yes.")
    print("2. No.")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        path = f"example_circuits_to_synthesize/best_circuits/circ_list_TFIM_qubit{n_qubits}.pickle"
        name = f"ground_{n_qubits}_J{J_value}"
    elif choice == "2":
        print()
        print('-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?')
        print('=>> Please go back to the previous folder and check README.')
        print('=>> There you can find instructions to run the GRL and save the best circuits')
        print('-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?-?')
        exit()
    else:
        print("Invalid choice. Please restart the program and enter either 1 or 2.")
        exit()
elif choice == "2":
    # Test program synthesis
    path = "example_circuits_to_synthesize/best_circuits/circ_list_for_testing_synthesis.pickle"
    name = "the_test_of_program_synthesis"
else:
    print("Invalid choice. Please restart the program and enter either 1 or 2.")
    exit()

class args:
    n_qubits = n_qubits
    J = 1
    hh = 0.001
    decomposed = 1
    arity = 2
    structurePenalty = 1
    pseudoCounts = 10


# Read the command line arguments
parser = argparse.ArgumentParser(
    description="Example implementation of Regularized Mutual Information Feature Selector on a solid drop.",
    epilog="Results will be saved in files with the OUTPUT tag in the 'outputs/' folder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-n_qubits", type=int, default=args.n_qubits, help="Number of qubits"
)
parser.add_argument("-J", type=float, default=args.J, help="Interaction strength")
parser.add_argument("-hh", type=float, default=args.hh, help="External field strength")
parser.add_argument(
    "-decomposed",
    type=int,
    default=args.decomposed,
    help="Either 0=parametrized gates, 1=qiskit hardware basis",
)
parser.add_argument(
    "-arity",
    type=int,
    default=args.arity,
    help="Number of arguments of extracted gates",
)
parser.add_argument(
    "-structurePenalty", type=int, default=args.structurePenalty, help="hyperparameter"
)
parser.add_argument(
    "-pseudoCounts", type=int, default=args.pseudoCounts, help="hyperparameter"
)

try:
    args = parser.parse_args()
except SystemExit as e:
    eprint("Running from interactive session. Loading default parameters")





# Output the selected path and name
print(f"Selected path: {path}")
print(f"Selected name: {name}")


"""
Making large depth circuit smaller.
"""
def calculate_cut_depth(cut_depth):
    # Your function logic here
    print(f"The cut depth is: {cut_depth}")

# Get user input for cut depth
cut_depth = int(input("(THE FINAL QUESTION) Please enter the cut depth (+ve integer only please): "))

# Call the function with user input
calculate_cut_depth(cut_depth)

# print(path)
with open(path, "rb") as handle:
    b = dill.load(handle)


eprint(f"Loading solutions from {path}")
# Unfortunately these flags are set globally
# TODO: remove
dc.domains.quantum_ground_state.primitives.GLOBAL_NQUBIT_TASK = args.n_qubits
dc.domains.quantum_ground_state.primitives.GLOBAL_LIMITED_CONNECTIVITY = False

library_settings = {
    "topK": 2,  # how many solutions to consider
    "arity": args.arity,  # how many arguments
    "structurePenalty": args.structurePenalty,  # increase regularization 3 4 (it was 1), look at a few in [1,15]
    "pseudoCounts": args.pseudoCounts,  # increase to 100, test a few values
}

primitives = [pr.p_x, pr.p_rz, pr.p_sx, pr.p_hadamard, pr.p_cz]
grammar = Grammar.uniform(primitives)
eprint(f"Library building settings: {library_settings}")



# Generate a few example tasks
solutions = {}  # dict of task:solution
# NOTE: we have a task for each decomposition because they have various different real parameters
# We cannot have solutions with different requests for a task,
# and it is not clear how to use real numbers as primitives (just for evaluation, we cannot enumerate them)
for idx, circuit in enumerate(b):

    circuit = trimmed_circuit(circuit, max_depth=cut_depth)

    H = construct_hamiltonian(args.J, args.hh, args.n_qubits)
    instructions = get_instructions_from_qiskit(circuit)
    code, arguments = get_code_from_instructions(instructions)

    program = Program.parse(code)

    
    task = GroundStateTask(
        f"J_{args.J:2.2f}_h_{args.hh:2.2f}_N_{args.n_qubits}_v{idx}",
        hamiltonian=H,
        arguments=arguments,
        request=program.infer(),
    )

    likelihood = task.logLikelihood(program)
    prior = grammar.logLikelihood(program.infer(), program)

    frontier_entry = FrontierEntry(
        program=program, logLikelihood=likelihood, logPrior=prior
    )

    solutions[task] = Frontier(
        frontier=[frontier_entry],  # multiple solutions are allowed
        task=task,
    )
    eprint(f"#{idx:3}, Energy = {likelihood:2.6f}")
tasks = list(solutions.keys())
frontiers = [f for f in solutions.values()]

unique_frontiers_set = set()
unique_frontiers = []
for frontier in frontiers:
    program = frontier.entries[0].program
    if program not in unique_frontiers_set:
        unique_frontiers_set.add(program)
        unique_frontiers.append(frontier)
eprint(
    f"We have {len(unique_frontiers)}/{len(frontiers)} frontiers. The others are duplicate solutions"
)

unique_frontiers
# Run library decomposition
start = time.time()
new_grammar, new_frontiers = FragmentGrammar.induceFromFrontiers(
    g0=grammar,
    frontiers=unique_frontiers[:],
    **library_settings,
    CPUs=1 ##numberOfCPUs() - 2
)
end = time.time()
delta = end-start
eprint(f"Completed gate extraction in {delta} seconds.")
timestamp = datetime.now().isoformat()
time_init = time.time()
with open(f"experimentOutputs/{name}_grammar.pickle", "wb") as f:
    pickle.dump(new_grammar, f)
with open(f"experimentOutputs/{name}_frontiers.pickle", "wb") as f:
    pickle.dump(new_frontiers, f)
print('FINISH TIME (seconds):', time.time()-time_init)
eprint(f"Results saved in experimentOutputs/{timestamp}_{name}_...")