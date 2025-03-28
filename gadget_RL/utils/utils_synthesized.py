import configparser
import json
from itertools import product, combinations
from qiskit import QuantumCircuit
import qiskit


def get_config(config_name,experiment_name, path='configuration_files',
               verbose=True):
    config_dict = {}
    Config = configparser.ConfigParser()
    Config.read('{}/{}{}'.format(path,config_name,experiment_name))
    for sections in Config:
        config_dict[sections] = {}
        for key, val in Config.items(sections):
            
            try:
                config_dict[sections].update({key: int(val)})
            except ValueError:
                config_dict[sections].update({key: val})
            floats = ['learning_rate',  'dropout', 'alpha', 
                      'beta', 'beta_incr', 
                      "shift_threshold_ball","succes_switch","tolearance_to_thresh","memory_reset_threshold",
                      "fake_min_energy","_true_en"]
            strings = ['ham_type', 'fn_type', 'geometry','method','agent_type',
                       "agent_class","init_seed","init_path","init_thresh","method",
                       "mapping","optim_alg", "curriculum_type"]
            lists = ['episodes','neurons', 'accept_err','epsilon_decay',"epsilon_min",
                     "epsilon_decay",'final_gamma','memory_clean',
                     'update_target_net', 'epsilon_restart', "thresholds", "switch_episodes"]
            if key in floats:
                config_dict[sections].update({key: float(val)})
            elif key in strings:
                config_dict[sections].update({key: str(val)})
            elif key in lists:
                config_dict[sections].update({key: json.loads(val)})
    del config_dict['DEFAULT']
    return config_dict




def dictionary_of_actions(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary

def dictionary_of_actions_decomposed(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
         
    for c in range(num_qubits):
        for x in range(c + 1, num_qubits):
            dictionary[i] = [c, x, num_qubits, 0]
            i += 1
   
    """h  denotes which gate. 1, 2, 3 -->  SX, X, RZ axes """
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary


def dictionary_of_actions_nd_synthesized_1(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
    """c is the control and (c+x) % num_qubits is the target of CX"""
    for c in range(num_qubits):
        for x in range(c + 1, num_qubits):
            dictionary[i] =  [c, x, num_qubits, 0, num_qubits, 0]
            i += 1
    
    """c is the control and (c+x) % num_qubits is the target of RYRX"""    
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] = [num_qubits, 0, c, x, num_qubits, 0]
        i += 1
    # print(num_qubits**2-num_qubits, len(dictionary.keys()))
   
    """h  denotes which gate. 1, 2, 3 -->  RX, RY, RZ axes """
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, num_qubits, 0, r, h]
        i += 1
    return dictionary

def dictionary_of_actions_synthesized(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
    """c is the control and (c+x) % num_qubits is the target of CZ"""
    for c in range(num_qubits):
        for x in range(c + 1, num_qubits):
            dictionary[i] =  [c, x, num_qubits, 0, num_qubits, 0]
            i += 1
    
    """c is the control and (c+x) % num_qubits is the target of RZCZ"""    
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] = [num_qubits, 0, c, x, num_qubits, 0]
        i += 1
    # print(num_qubits**2-num_qubits, len(dictionary.keys()))
   
    """h  denotes which gate. 1, 2, 3 -->  SX, X, RZ axes """
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, num_qubits, 0, r, h]
        i += 1
    return dictionary

def dictionary_of_actions_synthesized_2(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
    """c is the control and (c+z) % num_qubits is the target of CZ"""
    for c in range(num_qubits):
        for x in range(c + 1, num_qubits):
            dictionary[i] =  [c, x, num_qubits, 0, num_qubits, 0]
            i += 1
    
    """c is the control and (c+z) % num_qubits is the target of RZCZ"""    
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] = [num_qubits, 0, c, x, num_qubits, 0]
        i += 1
    # print(num_qubits**2-num_qubits, len(dictionary.keys()))
   
    """h  denotes which gate. 1, 2, 3, 4 -->  SX, X, XSX, RZ axes """
    for r, h in product(range(num_qubits),
                           range(1, 5)):
        dictionary[i] = [num_qubits, 0, num_qubits, 0, r, h]
        i += 1
    return dictionary


def dictionary_of_actions_synthesized_3(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
    """c is the control and (c+z) % num_qubits is the target of CZ"""
    for c in range(num_qubits):
        for x in range(c + 1, num_qubits):
            dictionary[i] =  [c, x, num_qubits, 0, num_qubits, 0, num_qubits, 0]
            i += 1
    
    """c is the control and (c+z) % num_qubits is the target of RZ0CZ01"""    
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] = [num_qubits, 0, c, x, num_qubits, 0, num_qubits, 0]
        i += 1
    
    """c is the control and (c+z) % num_qubits is the target of RZ0RZ1CZ01"""    
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] = [num_qubits, 0, num_qubits, 0, c, x, num_qubits, 0]
        i += 1
   
    """h  denotes which gate. 1, 2, 3, 4 -->  SX, X, XSX, RZ axes """
    for r, h in product(range(num_qubits),
                           range(1, 5)):
        dictionary[i] = [num_qubits, 0, num_qubits, 0, num_qubits, 0, r, h]
        i += 1
    return dictionary


def dict_of_actions_revert_q(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations. Systems have reverted order to above dictionary of actions.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits-1,-1,-1),
                        range(num_qubits-1,0,-1)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  SX, X, RZ axes """
    for r, h in product(range(num_qubits-1,-1,-1),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary

def dict_of_actions_revert_q_decomposed(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations. Systems have reverted order to above dictionary of actions.
    """
    dictionary = dict()
    i = 0
         
    for c in range(num_qubits):
        for x in range(c + 1, num_qubits):
            dictionary[i] = [c, x, num_qubits, 0]
            i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits-1,-1,-1),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary

# for q in range(2,10):
#     combo = list(combinations(range(q), 2))
#     print(q, len(dictionary_of_actions_decomposed(q).keys()), len(combo) + q*3, len(dictionary_of_actions(q).keys()))

#     print()
   

   
