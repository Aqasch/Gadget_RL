from utils.synthesized_gates import *

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import random_unitary



class Parametric_Circuit:
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.ansatz = QuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        
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
        """
        THE PREVIOUS ENCODING

        [0., 0.], - The CZ
        [0., 0.], - The CZ
        [0., 0.], - The SX
        [0., 0.], - The X
        [0., 0.], - The RZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZ (angle)

        AFTER MODIFICATION!!

        [0., 0.], - The CZ
        [0., 0.], - The CZ
        [0., 0.], - The SX
        [0., 0.], - The X
        [0., 0.], - The RZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZ (angle)

        """
        
        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3+self.n_qubits:]
            """
            OLDS!

            rot_pos = (local_state[2*self.n_qubits: 2*self.n_qubits+3] == 1).nonzero( as_tuple = True )
            rzcz_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            cz_pos = (local_state[self.n_qubits:2*self.n_qubits] == 1).nonzero( as_tuple = True )
            """

            # mod cz position extraction!
            cz_pos = np.where(local_state[0:self.n_qubits] == 1)
            # rotation position extration
            one_gate_pos = np.where(local_state[self.n_qubits: self.n_qubits+3] == 1)
            # mod rzcz extration
            rzcz_pos = np.where(local_state[self.n_qubits+3: self.n_qubits+3+self.n_qubits] == 1)
            

            targ_cz = cz_pos[0]
            ctrl_cz = cz_pos[1]
            if len(ctrl_cz) != 0:
                for r in range(len(ctrl_cz)):
                    self.ansatz.cz([ctrl_cz[r].item()], [targ_cz[r].item()])
            
            targ_rzcz = rzcz_pos[0]
            ctrl_rzcz = rzcz_pos[1]

            if len(ctrl_rzcz) != 0:
                for r in range(len(ctrl_rzcz)):
                    self.ansatz.append(rzcz(thetas[targ_rzcz[r].item()+1][targ_rzcz[r].item()].item(), ctrl=ctrl_rzcz[r].item(), targ=targ_rzcz[r].item()), [ctrl_rzcz[r].item(), targ_rzcz[r].item()])
                    if random.random() < 0.5:
                        noise_unitary = random_unitary(2, seed=42)
                        self.ansatz.append(noise_unitary, [ctrl_rzcz[r].item()])
                        # print(self.ansatz)
                        # exit()
            rot_direction_list = one_gate_pos[0]
            rot_qubit_list = one_gate_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        self.ansatz.sx(rot_qubit.item())
                    elif r == 1:
                        self.ansatz.x(rot_qubit.item())
                    elif r == 2:
                        self.ansatz.rz(thetas[0][rot_qubit].item(), rot_qubit.item())
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2                       
        return self.ansatz


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
        # print(gate_detail.name)
        if gate_detail.name in ['rx', 'ry', 'rz', 'rzcz01', 'rzcz10']:
            list(i)[0].params = [angles[no]]
            no+=1
    # print(circuit, 'after')
    # print('-x-x-x-x-x-')
    # print()
    
    state = np.asmatrix(Statevector.from_instruction(circuit))
    energy = (state @ observable) @ state.getH()

    return float(energy.real)#[0][0]

def get_shot_noise(weights, n_shots):
    
    return 0
        


def get_exp_val(n_qubits,circuit,op, phys_noise = False, err_mitig = 0):
            
    state = np.asmatrix(Statevector.from_instruction(circuit))
    # print(state.shape)
    energy = (state @ op) @ state.getH()

    return float(energy.real)



if __name__ == "__main__":
    pass


















