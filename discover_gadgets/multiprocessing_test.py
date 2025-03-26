#! /bin/env python
from multiprocessing import set_start_method

import argparse
import dill
from contextlib import closing
import logging
from multiprocessing import Pool
import time
from memory_profiler import profile
from concurrent.futures import ProcessPoolExecutor, as_completed

from akash.TFIM_ham_gen import construct_hamiltonian

import dreamcoder as dc
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
import dreamcoder.domains.quantum_ground_state.primitives as pr
from dreamcoder.domains.quantum_ground_state.primitives import (
    get_instructions_from_qiskit,
    get_code_from_instructions,
)
from dreamcoder.domains.quantum_ground_state.tasks import GroundStateTask
from dreamcoder.program import Program
from dreamcoder.utilities import eprint

class args:
    n_qubits = 2
    J = 1
    hh = 0.05
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
path = f"akash/solved_RL_circuits/circ_list_TFIM_qubit{args.n_qubits}_J{args.J}_h{args.hh}_decomposed{args.decomposed}.pickle"
name = f"ground_{args.n_qubits}_J{args.J}_h{args.hh}_dec{args.decomposed}"
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

primitives = [pr.p_sx, pr.p_x, pr.p_rz, pr.p_cz]
grammar = Grammar.uniform(primitives)
# eprint(f"Library building settings: {library_settings}")
# Generate a few example tasks
solutions = {}  # dict of task:solution
# NOTE: we have a task for each decomposition because they have various different real parameters
# We cannot have solutions with different requests for a task,
# and it is not clear how to use real numbers as primitives (just for evaluation, we cannot enumerate them)
for idx, circuit in enumerate(b):
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
    # eprint(f"#{idx:3}, Energy = {likelihood:2.6f}")
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


@profile
def parallelMap(numberOfCPUs, f, *xs, chunksize=None, maxtasksperchild=None):
    if numberOfCPUs == 1:
        return list(map(f, *xs))

    n = len(xs[0])
    for x in xs:
        assert len(x) == n

    # Batch size of jobs as they are sent to processes
    if chunksize is None:
        chunksize = max(1, n // (numberOfCPUs * 2))

    with closing(Pool(numberOfCPUs, maxtasksperchild=maxtasksperchild)) as pool:
        # ys = pool.map(parallelMapCallBack, range(n), chunksize=chunksize)
        ys = pool.map(f, xs[0], chunksize=chunksize)
    # with ProcessPoolExecutor(max_workers=numberOfCPUs) as executor:
    #     ys = executor.map(f, xs[0], chunksize=chunksize)

    return ys


arity = 2
CPUs = 1


def fragment(frontier, a=arity):
    return dc.fragmentUtilities._frontier_fragmenter(frontier, a)


if __name__ == "__main__":
    # set_start_method("spawn")
    
    from mem_top import mem_top
    import gc

    start = time.time()
    result = parallelMap(CPUs, fragment, unique_frontiers[1:2])
    end = time.time()
    eprint(f"Completed gate extraction in { end-start} seconds using {CPUs} CPUs.")
    del result
    gc.collect()
    print(mem_top())