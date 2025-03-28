import numpy as np
from typing import List, Callable, Optional, Dict
from scipy.optimize import OptimizeResult
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector





class Parametric_Circuit:
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.ansatz = QuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        # print('--------------')
        # print(state[:3])
        
        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3:]
            rot_pos = (local_state[self.n_qubits: self.n_qubits+3] == 1).nonzero( as_tuple = True )
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.cx([ctrl[r].item()], [targ[r].item()])
            
            rot_direction_list = rot_pos[0]
            rot_qubit_list = rot_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        self.ansatz.rx(thetas[0][rot_qubit].item(), rot_qubit.item())
                    elif r == 1:
                        self.ansatz.ry(thetas[1][rot_qubit].item(), rot_qubit.item())
                    elif r == 2:
                        self.ansatz.rz(thetas[2][rot_qubit].item(), rot_qubit.item())
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2                       
        return self.ansatz
    
    def construct_ansatz_decomposed(self, state):
        # print('--------------')
        # print(state[:3])
        
        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3:]
            rot_pos = (local_state[self.n_qubits: self.n_qubits+3] == 1).nonzero( as_tuple = True )
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            # print(rot_pos, 'this!!!')
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.cz([ctrl[r].item()], [targ[r].item()])
            
            rot_direction_list = rot_pos[0]
            rot_qubit_list = rot_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        self.ansatz.sx(rot_qubit.item())
                    elif r == 1:
                        self.ansatz.x(rot_qubit.item())
                    elif r == 2:
                        self.ansatz.rz(thetas[2][rot_qubit].item(), rot_qubit.item())
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2                       
        return self.ansatz

        

def get_energy_qulacs_(angles, observable, 
                      weights,circuit, n_qubits, 
                      energy_shift, n_shots,
                      noise_value,
                      M=1000,
                      which_angles=[]):
    """"
    Function for Qiskit energy minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    observable            [Observable] : Qulacs observable (Hamiltonian)
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    energy_shift          [float]      : energy shift for Qiskit Hamiltonian after freezing+removing orbitals
    noise_value           [float]      : Physical error probability of the quantum channel; e.g, depolarizing etc
    n_shots               [int]        : Statistical noise, number of samples taken from QC
    M                     [int]        : Physical noise, number of repetitions during expectation value calculation (Q.trajectories)
    
    Output:
    expval [float] : expectation value 
    
    """
    # print(angles)
    print(circuit)
    no = 0
    for i in circuit:
        gate_detail = list(i)[0]
        if gate_detail.name in ['rx', 'ry', 'rz']:
            list(i)[0].params = [angles[no]]
            no+=1
    # print(circuit)
    
    state = np.asmatrix(Statevector.from_instruction(circuit))
# state = state.getH() @ state
    energy = (state @ observable) @ state.getH()

    return float(energy.real)#[0][0]


def get_energy_qiskit(angles, observable,circuit, n_qubits, n_shots,
                      phys_noise = False,
                      which_angles=[]):
    """"
    Function for Qiskit energy minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    observable            [Observable] : Qulacs observable (Hamiltonian)
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    energy_shift          [float]      : energy shift for Qiskit Hamiltonian after freezing+removing orbitals
    n_shots               [int]        : Statistical noise, number of samples taken from QC
    phys_noise            [bool]       : Whether quantum error channels are available (DM simulation) 
    
    Output:
    expval [float] : expectation value 
    
    """
    # print(angles)
    # print(circuit, 'before')

    no = 0
    for i in circuit:
        gate_detail = list(i)[0]
        if gate_detail.name in ['rx', 'ry', 'rz']:
            # print(angles, no)
            list(i)[0].params = [angles[no]]
            no+=1
    # print(circuit, 'after')
    # print('-x-x-x-x-x-')
    # print()
    
    state = np.asmatrix(Statevector(circuit))
    energy = (state @ observable) @ state.getH()

    return float(energy.real)#[0][0]

def get_shot_noise(weights, n_shots):
    
    shot_noise = 0
    
    if n_shots > 0:
        weights1, weights2 = weights[np.abs(weights) > 0.05], weights[np.abs(weights) <= 0.05]
        mu,sigma1,sigma2 =0,(10*n_shots)**(-0.5), (n_shots)**(-0.5)
        
        shot_noise +=(np.array(weights1).real).T@np.random.normal(mu,sigma1,len(weights1))
        shot_noise +=(np.array(weights2).real).T@np.random.normal(mu,sigma2,len(weights2))
        
    return shot_noise
        


def get_exp_val(n_qubits,circuit,op, phys_noise = False, err_mitig = 0):
            
    # state = np.asmatrix(Statevector.from_instruction(circuit))
    state = np.asmatrix(Statevector(circuit))

    # print(state.shape)
    energy = (state @ op) @ state.getH()
    
    return float(energy.real)



if __name__ == "__main__":
    pass


















