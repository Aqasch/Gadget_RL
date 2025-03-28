#!/bin/env python
import pickle
N = 2
J=1
decomposed_list = [0,1]
mag_field_strength_list = [0.001,0.05,1]
mag_field_strength = [mag_field_strength_list[0]]
for decomposed in decomposed_list:
    print('decomposed?', decomposed)
    
    for h in mag_field_strength:
        # with open(f'example_circuits_to_synthesize/best_circuits/circ_list_TFIM_qubit{N}.pickle', 'rb') as handle:
            # b = pickle.load(handle)

        with open(f'example_circuits_to_synthesize/best_circuits/circ_list_TFIM_qubit{N}.pickle', 'rb') as handle:
            b = pickle.load(handle)
        print(J, len(b))
        exit()
        for circ in b:
            print(circ)
        print('-x-x-x-x-x-')
        print()
