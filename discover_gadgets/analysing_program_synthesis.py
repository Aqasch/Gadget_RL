import dill

path_frontier = f'experimentOutputs/ground_2_J1_frontiers.pickle'
path_grammar = f'experimentOutputs/ground_2_J1_grammar.pickle'


with open(path_frontier, "rb") as handle:
    b1 = dill.load(handle)
with open(path_grammar, "rb") as handle:
    b2 = dill.load(handle)


print(b1)