import torch
from utils.utils_synthesized import *
from sys import stdout
import scipy
import environments.VQEs.VQE_synthesized_2 as vc
import os
import numpy as np
import copy
from utils import curricula
import copy
from qiskit import QuantumCircuit
from utils.synthesized_gates import *

class CircuitEnv():

    def __init__(self, conf, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        self.random_halt = int(conf['env']['rand_halt'])
        self.decomposed = int(conf['env']['decomposed'])
        
        self.n_shots =   conf['env']['n_shots'] 
        noise_models = ['depolarizing', 'two_depolarizing', 'amplitude_damping']
        noise_values = conf['env']['noise_values'] 
        
        self.ham_type = conf['problem']['ham_type']

        if noise_values !=0:
            indx = conf['env']['noise_values'].index(',')
            self.noise_values = [float(noise_values[1:indx]), float(noise_values[indx+1:-1])]
        else:
            self.noise_values = []
        self.noise_models = noise_models[0:len(self.noise_values)]
        
        if len(self.noise_models) > 0:
            self.phys_noise = True
        else:
            self.phys_noise = False

        self.err_mitig = conf['env']['err_mitig']
        

        self.ham_model = conf['problem']['ham_model']

        self.fake_min_energy = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else None
        self.fn_type = conf['env']['fn_type']
        
        if "cnot_rwd_weight" in conf['env'].keys():
            self.cnot_rwd_weight = conf['env']['cnot_rwd_weight']
        else:
            self.cnot_rwd_weight = 1.
        
        
        self.noise_flag = True
        self.state_with_angles = conf['agent']['angles']
        self.current_number_of_cnots = 0
        # self.synthesized = 1 # MAKE IT A VARIABLE IN CONFIG!!!
        

        self.action_dict = dictionary_of_actions_synthesized_2(self.num_qubits)

        
        print(f"ham_data/{self.ham_type}_{self.ham_model}.npz")
        self.curriculum_dict = {}
        __ham = np.load(f"ham_data/{self.ham_type}_{self.ham_model}.npz")
        # print(f"ham_data/{self.ham_type}_{self.ham_model}.npz")
        # print()
        
        
        self.hamiltonian, eigvals = __ham['hamiltonian'],__ham['eigvals']

        min_eig = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else min(eigvals)

        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals)
        self.max_eig = max(eigvals)

        self.curriculum_dict[self.ham_type] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_energy=min_eig)

        # print(self.curriculum_dict)

        self.device = device
        # self.ket = QuantumState(self.num_qubits)
        self.done_threshold = conf['env']['accept_err']
        

        stdout.flush()
        
        self.state_size = self.num_layers*self.num_qubits*(self.num_qubits+4+self.num_qubits+3)
        # print('state size', self.state_size)
        self.step_counter = -1
        self.prev_energy = None
        self.moments = [0]*self.num_qubits
        self.illegal_actions = [[]]*self.num_qubits
        self.energy = 0
        self.opt_ang_save = 0

        # self.action_size = len(list(combinations(range(self.num_qubits), 2)))+self.num_qubits*2+self.num_qubits**2
        self.action_size = len(self.action_dict.keys())
        # print(self.action_size)
        self.previous_action = [0, 0, 0, 0, 0, 0]
        self.save_circ = 0


        if 'non_local_opt' in conf.keys():
            self.global_iters = conf['non_local_opt']['global_iters']
            self.optim_method = conf['non_local_opt']["method"]
            self.optim_alg = conf['non_local_opt']['optim_alg']
            

            if 'a' in conf['non_local_opt'].keys():
                self.options = {'a': conf['non_local_opt']["a"], 'alpha': conf['non_local_opt']["alpha"],
                            'c': conf['non_local_opt']["c"], 'gamma': conf['non_local_opt']["gamma"],
                            'beta_1': conf['non_local_opt']["beta_1"],
                            'beta_2': conf['non_local_opt']["beta_2"]}

            if 'lamda' in conf['non_local_opt'].keys():
                self.options['lamda'] = conf['non_local_opt']["lamda"]

            if 'maxfev' in conf['non_local_opt'].keys():
                self.maxfev = {}
                self.maxfev['maxfev'] = int(conf['non_local_opt']["maxfev"])

            if 'maxfev1' in conf['non_local_opt'].keys():
                self.maxfevs = {}
                self.maxfevs['maxfev1'] = int( conf['non_local_opt']["maxfev1"] )
                self.maxfevs['maxfev2'] = int( conf['non_local_opt']["maxfev2"] )
                self.maxfevs['maxfev3'] = int( conf['non_local_opt']["maxfev3"] )
                
        else:
            self.global_iters = 0
            self.optim_method = None
        
            



    def step(self, action, train_flag = True) :
        
        """
        Action is performed on the first empty layer.
        Variable 'step_counter' points last non-empty layer.
        """  
        
        next_state = self.state.clone()
        
        self.step_counter += 1

        """
        The state looks like 
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ
        [0., 0.], - The CZ
        [0., 0.], - The CZ
        [0., 0.], - The SX
        [0., 0.], - The X
        [0., 0.], - The RZ
        [0., 0.], - The RZ (angle)
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZCZ (angle)

        AFTER MODIFICATION

        [0., 0.], - The CZ
        [0., 0.], - The CZ
        [0., 0.], - The SX
        [0., 0.], - The X
        [0., 0.], - The RZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZ (angle)
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZCZ (angle)

        AFTER (SECOND GATE INTRODUCTION) MODIFICATION

        [0., 0.], - The CZ
        [0., 0.], - The CZ
        [0., 0.], - The SX
        [0., 0.], - The X
        [0., 0.], - The XSX
        [0., 0.], - The RZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZ (angle)
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZCZ (angle)
        """
        self.action = action


        
        ctrl1 = action[0] # CONTROL FOR CZ
        targ1 = action[1] # TARGET FOR CZ
        ctrl2 = action[2] # CONTROL FOR RZCZ
        targ2 = (action[2] + action[3]) % self.num_qubits # TARGET FOR RZCZ

        rot_qubit = action[4]
        rot_axis = action[5]

        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[ rot_qubit ]
        elif ctrl1 < self.num_qubits:
            gate_tensor = max( self.moments[ctrl1], self.moments[targ1] )
        elif ctrl2 < self.num_qubits:
            gate_tensor = max( self.moments[ctrl2], self.moments[targ2] )

        # print(gate_tensor)

        # print('ROTATION POSITION AND QUBIT')
        # print(self.num_qubits+rot_axis-1, rot_axis, rot_qubit )
        # CZ GATE APPEND
        if ctrl1 < self.num_qubits:
            next_state[gate_tensor][targ1][ctrl1] = 1
        
        # MOD 1q GATE APPEND (THE LAST ROW IS RZ)
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits+rot_axis-1][rot_qubit] = 1
        
        # MOD RZCZ GATE APPEND
        elif ctrl2 < self.num_qubits:
            next_state[gate_tensor][self.num_qubits+4+targ2][ctrl2] = 1


        """
        # RZCZ GATE APPEND
        elif ctrl2 < self.num_qubits:
            next_state[gate_tensor][targ2+self.num_qubits][ctrl2] = 1
        
        # 1q GATES APPEND
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][2*self.num_qubits+rot_axis-1][rot_qubit] = 1
        """

        if rot_qubit < self.num_qubits:
            self.moments[ rot_qubit ] += 1
        elif ctrl1 < self.num_qubits:
            max_of_two_moments = max( self.moments[ctrl1], self.moments[targ1] )
            self.moments[ctrl1] = max_of_two_moments +1
            self.moments[targ1] = max_of_two_moments +1
        elif ctrl2 < self.num_qubits:
            max_of_two_moments = max( self.moments[ctrl2], self.moments[targ2] )
            self.moments[ctrl2] = max_of_two_moments +1
            self.moments[targ2] = max_of_two_moments +1
            
        self.current_action = action
        self.illegal_action_new()
        self.state = next_state.clone()
        if self.optim_method in ["scipy_each_step"]:
            thetas, _, opt_ang = self.scipy_optim(self.optim_alg)
            # print(opt_ang, 'Optimized angles!!!')
            for i in range(self.num_layers):
                for j in range(1+self.num_qubits):
                    next_state[i][self.num_qubits+4+self.num_qubits+j,:] = thetas[i][j,:]
        # print(next_state, 'IN THE STEP!!')
        self.opt_ang_save = opt_ang

        self.state = next_state.clone()
        
        energy,energy_noiseless = self.get_energy()

        if self.noise_flag == False:
            energy = energy_noiseless

        self.energy = energy
        # print(self.energy)

        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)

        # print(energy, self.min_eig)

        self.error = float(abs(self.min_eig-energy))
        self.error_noiseless = float(abs(self.min_eig-energy_noiseless))
        # print(self.error, self.done_threshold)
        rwd = self.reward_fn(energy)
        self.rwd = rwd
        self.prev_energy = np.copy(energy)

        energy_done = int(self.error < self.done_threshold)

        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(energy_done or layers_done)
        
        self.previous_action = copy.deepcopy(action)
        self.save_circ = 0 #self.make_circuit_decomposed()
        # print(thetas)
        # print(self.save_circ)
        # print(self.error, self.done_threshold)
        # print(self.save_circ, 'ARE YOU NOT TAKING ANGLES MY BROTHER???')
        # print('---------')
        # if self.save_circ.depth() >=8:
        #     exit()
        

        if self.random_halt:
            if self.step_counter == self.halting_step:
                done = 1
        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[self.current_prob] = copy.deepcopy(self.curriculum)
        
        if self.state_with_angles:
            return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
        else:
            next_state = next_state[:, :self.num_qubits+4+self.num_qubits]
            return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done

    def reset(self):
        """
        Returns randomly initialized state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with control gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with not gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """

        state = torch.zeros((self.num_layers, self.num_qubits+4+self.num_qubits+1+self.num_qubits, self.num_qubits))

        self.state = state
        
        
        if self.random_halt:
            statistics_generated = np.clip(np.random.negative_binomial(n=70,p=0.573, size=1),25,70)[0]
            
            self.halting_step = statistics_generated

        
        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits]*6
        self.illegal_actions = [[]]*self.num_qubits

        self.make_circuit_decomposed(state)
        self.step_counter = -1
        # print(self.save_circ.data)
        
        self.moments = [0]*self.num_qubits
        self.current_prob = self.ham_type
        self.curriculum = copy.deepcopy(self.curriculum_dict[self.current_prob])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        self.ham_type = self.ham_type
        __ham = np.load(f"ham_data/{self.ham_type}_{self.ham_model}.npz")
        self.hamiltonian,eigvals= __ham['hamiltonian'],__ham['eigvals']
        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals)
        self.prev_energy = self.get_energy(state)[1]

        if self.state_with_angles:
            return state.reshape(-1).to(self.device)
        else:
            state = state[:, :self.num_qubits+4+self.num_qubits]
            return state.reshape(-1).to(self.device)

    def make_circuit_decomposed(self, thetas=None):
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

        AFTER (SECOND GATE INTRODUCTION) MODIFICATION

        [0., 0.], - The CZ
        [0., 0.], - The CZ
        [0., 0.], - The SX
        [0., 0.], - The X
        [0., 0.], - The XSX
        [0., 0.], - The RZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZ (angle)
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZCZ (angle)
        
        3-qubit:
        [0., 0. 0.], - The CZ
        [0., 0. 0.], - The CZ
        [0., 0. 0.], - The CZ

        [0., 0. 0.], - The SX
        [0., 0. 0.], - The X
        [0., 0. 0.], - The XSX
        [0., 0. 0.], - The RZ
        
        [0., 0. 0.], - The RZCZ
        [0., 0. 0.], - The RZCZ
        [0., 0. 0.], - The RZCZ
        
        [0., 0. 0.], - The RZ (angle)
        
        [0., 0. 0.], - The RZCZ (angle)
        [0., 0. 0.], - The RZCZ (angle)
        [0., 0. 0.], - The RZCZ (angle)
        [0., 0. 0.], - The RZCZ (angle)
        [0., 0. 0.], - The RZCZ (angle)
        
        """
        state = self.state.clone()
        if thetas is None:
            thetas = state[:, self.num_qubits+4+self.num_qubits:]
        
        # print(thetas, 'IN THE MAKE CIRCUIT DECOMPOSED!!!!!')
        circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_layers):
            """
            OLDS!
            czrz_pos = np.where(state[i][0:self.num_qubits] == 1)
            cz_pos = np.where(state[i][self.num_qubits:2*self.num_qubits] == 1)
            rot_pos = np.where(state[i][2*self.num_qubits: 2*self.num_qubits+3] == 1)
            """

            # mod cz position extraction!
            cz_pos = np.where(state[i][0:self.num_qubits] == 1)
            # rotation position extration
            oneq_gate_pos = np.where(state[i][self.num_qubits: self.num_qubits+4] == 1)
            rzcz_pos = np.where(state[i][self.num_qubits+4: self.num_qubits+4+self.num_qubits] == 1)

             
            
            # print(thetas)
            # print(rot_pos)
            # print(rzcz_pos)

            # cz append
            targ_cz = cz_pos[0]
            ctrl_cz = cz_pos[1]
            if len(ctrl_cz) != 0:
                for r in range(len(ctrl_cz)):
                    circuit.cz([ctrl_cz[r]], [targ_cz[r]])
            
            targ_rzcz = rzcz_pos[0]
            ctrl_rzcz = rzcz_pos[1]
            if len(ctrl_rzcz) != 0:
                for r in range(len(ctrl_rzcz)):
                    circuit.append(rzcz(thetas[i][targ_rzcz[r]+1][ctrl_rzcz[r]].item(), ctrl=ctrl_rzcz[r], targ=targ_rzcz[r]), [ctrl_rzcz[r], targ_rzcz[r]])
            
            
                  
            rot_direction_list, rot_qubit_list = oneq_gate_pos[0], oneq_gate_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        circuit.sx(rot_qubit)
                    elif r == 1:
                        circuit.x(rot_qubit)
                    elif r == 2:
                        circuit.append(xsx(), [rot_qubit])
                    elif r == 3:
                        # print(thetas, thetas[i][0][rot_qubit].item(), 'THIS IS THE MAIN THING!!!! RZ')
                        circuit.rz(thetas[i][0][rot_qubit].item(), rot_qubit)
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >3
        return circuit


    def get_energy(self, thetas=None):
        
        circ = self.make_circuit_decomposed(thetas)
        # print(circ)
        qiskit_inst = vc.Parametric_Circuit(n_qubits = self.num_qubits, noise_models = self.noise_models, noise_values = self.noise_values)
        
        # print(circ)
        noisy_circ = qiskit_inst.construct_ansatz_decomposed(self.state)
        # print(noisy_circ)
        expval_noisy = vc.get_exp_val(self.num_qubits,noisy_circ,self.hamiltonian,self.phys_noise,self.err_mitig)
        
        expval_noiseless = vc.get_exp_val(self.num_qubits,circ,self.hamiltonian)

        shot_noise = 0       
            
        energy = expval_noisy + shot_noise
        
        energy_noiseless = expval_noiseless
        
        return energy, energy_noiseless
    

        
    def scipy_optim(self, method, which_angles = [] ):
        """
        The state looks like 
        [0., 0.], - The CZ
        [0., 0.], - The CZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ
        [0., 0.], - The SX
        [0., 0.], - The X
        [0., 0.], - The RZ
        [0., 0.], - The RZ (angle)
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZCZ (angle)

        AFTER MODIFICATION

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

        AFTER (SECOND GATE INTRODUCTION) MODIFICATION

        [0., 0.], - The CZ
        [0., 0.], - The CZ
        [0., 0.], - The SX
        [0., 0.], - The X
        [0., 0.], - The XSX
        [0., 0.], - The RZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZCZ
        [0., 0.], - The RZ (angle)
        [0., 0.], - The RZCZ (angle)
        [0., 0.], - The RZCZ (angle)
        """
        state = self.state.clone()
        thetas = state[:, self.num_qubits+4+self.num_qubits:]
        rot_pos = (state[:,self.num_qubits+3: self.num_qubits+3+3] == 1).nonzero( as_tuple = True )
        angles = thetas[rot_pos]
        # print(angles)
        qiskit_inst = vc.Parametric_Circuit(n_qubits=self.num_qubits,noise_models=self.noise_models,noise_values = self.noise_values)
        qiskit_circuit = qiskit_inst.construct_ansatz_decomposed(state)
        # print(qiskit_circuit)
        
        x0 = np.asarray(angles.cpu().detach())
        # print(x0, 'b')
        def cost(x):
            return vc.get_energy_qiskit(x, observable = self.hamiltonian, circuit = qiskit_circuit,
            n_qubits = self.num_qubits,
            n_shots = int(self.n_shots), phys_noise = self.phys_noise,
                      which_angles=[])

        if list(which_angles):
            result_min_qiskit = scipy.optimize.minimize(cost, x0 = x0[which_angles], method = method, options = {'maxiter':self.global_iters})
            # print(result_min_qiskit['x'], which_angles, 1)
            x0[which_angles] = result_min_qiskit['x']
            thetas = state[:, self.num_qubits+4+self.num_qubits:]
            thetas[rot_pos] = torch.tensor(x0, dtype=torch.float)
        else:
            result_min_qiskit = scipy.optimize.minimize(cost, x0 = x0, method = method, options = {'maxiter':self.global_iters})
            # print(result_min_qiskit['x'], which_angles, 2)
            thetas = state[:, self.num_qubits+4+self.num_qubits:]
            # print(thetas, 'in scipy_optim before')
            thetas[rot_pos] = torch.tensor(result_min_qiskit['x'], dtype=torch.float)
            # print(thetas, 'in scipy_optim after')
        # print(result_min_qiskit['nfev'], 'after')

        return thetas, result_min_qiskit['nfev'], result_min_qiskit['x']

    def reward_fn(self, energy):
        
        
        if self.fn_type == "staircase":
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "two_step":
            return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
        elif self.fn_type == "two_step_end":

            max_depth = self.step_counter == (self.num_layers - 1)
            if ((self.error < self.done_threshold) or max_depth):
                return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
            else:
                return 0.0
        elif self.fn_type == "naive":
            return 0. + 1.*(self.error < self.done_threshold)
        elif self.fn_type == "incremental":
            return (self.prev_energy - energy)/abs(self.prev_energy - self.min_eig)
        elif self.fn_type == "incremental_clipped":
            return np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
        elif self.fn_type == "nive_fives":

            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = 0.
            return rwd
        
        elif self.fn_type == "incremental_with_fixed_ends":
            

            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            return rwd
        
        elif self.fn_type == "log":
            return -np.log(1-(energy/self.min_eig))
        
        elif self.fn_type == "log_to_ground":
            
            return -np.log(self.error)
        
        elif self.fn_type == "log_to_threshold":
            if self.error < self.done_threshold + 1e-5:
                rwd = 11
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_0_end":
            rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_50_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 50
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_100_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 100
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_500_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 100
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
	
        elif self.fn_type == "log_to_threshold_500_end":
                if self.error < self.done_threshold + 1e-5:
                    rwd = 500
                else:
                    rwd = -np.log(abs(self.error - self.done_threshold))
                return rwd
        
        elif self.fn_type == "log_to_threshold_1000_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 1000
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_non_repeat_energy":
            if self.error < self.done_threshold + 1e-5:
                rwd = 30
            elif np.abs(self.energy-self.prev_energy) <= 1e-3:
                rwd = -30
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_no_repeat_actions":
            if self.current_action == self.previous_action:
                return -1 
            elif self.error < self.done_threshold + 1e-5:
                rwd = 20
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_neg_punish":
            return -np.log(1-(energy/self.min_eig)) - 5
        
        elif self.fn_type == "end_energy":

            max_depth = self.step_counter == (self.num_layers - 1)
            
            if ((self.error < self.done_threshold) or max_depth):
                rwd = (self.max_eig - energy) / (abs(self.min_eig) + abs(self.max_eig))
            else:
                rwd = 0.0

        elif self.fn_type == "hybrid_reward":
            path = 'threshold_crossed.npy'
            if os.path.exists(path):
                
                threshold_pass_info = np.load(path)
                if threshold_pass_info > 8:

                    max_depth = self.step_counter == (self.num_layers - 1)
                    if (self.error < self.done_threshold):
                        rwd = 5.
                    elif max_depth:
                        rwd = -5.
                    else:
                        rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
                    return rwd
                else:
                    if self.error < self.done_threshold + 1e-5:
                        rwd = 11
                    else:
                        rwd = -np.log(abs(self.error - self.done_threshold))
                    return rwd
            else:
                np.save('threshold_crossed.npy', 0)
        
        elif self.fn_type == 'negative_above_chem_acc':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = 1000*(self.done_threshold/self.error)
            return rwd
        
        elif self.fn_type == 'negative_above_chem_acc_non_increment':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = self.done_threshold/self.error
            return rwd
        
        elif self.fn_type == 'negative_above_chem_acc_slight_increment':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = 100*(self.done_threshold/self.error)
            return rwd


        elif self.fn_type == "cnot_reduce":

            max_depth = self.step_counter == (self.num_layers - 1)
            
            
            if (self.error < self.done_threshold):
                rwd = self.num_layers - self.cnot_rwd_weight*self.current_number_of_cnots
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            return 

        
    def illegal_action_new(self):
        action = self.current_action
        illegal_action = self.illegal_actions
        
        ctrl_cz, targ_cz = action[0], action[1] #CZ
        ctrl_rzcz, targ_rzcz = action[2], (action[2] + action[3]) % self.num_qubits # RZCZ
        rot_qubit, rot_axis = action[4], action[5]
        
        #CZ
        if ctrl_cz < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        # ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        ill_ac_targ = ill_ac[1]
                        
                        if ill_ac[2] == self.num_qubits:
                        
                            if ctrl_cz == ill_ac[0] or ctrl_cz == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ_cz == ill_ac[0] or targ_cz == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl_cz == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ_cz == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break                          
            else:
                illegal_action[0] = action
        
        # RZCZ
        if ctrl_rzcz < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[2] + ill_ac[3] ) % self.num_qubits
                        
                        if ill_ac[2] == self.num_qubits:
                        
                            if ctrl_rzcz == ill_ac[0] or ctrl_rzcz == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ_rzcz == ill_ac[0] or targ_rzcz == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl_rzcz == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ_rzcz == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break                          
            else:
                illegal_action[0] = action

                            
        if rot_qubit < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ_cz = ill_ac[1]
                        ill_ac_targ_rzcz = ( ill_ac[2] + ill_ac[3] ) % self.num_qubits
                        
                        if ill_ac[0] == self.num_qubits:
                            
                            if rot_qubit == ill_ac[4] and rot_axis != ill_ac[5]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            elif rot_qubit != ill_ac[4]:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if rot_qubit == ill_ac[0]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                                        
                            elif rot_qubit == ill_ac_targ_cz:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            elif rot_qubit == ill_ac_targ_rzcz:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break 
            else:
                illegal_action[0] = action
        
        for indx in range(self.num_qubits):
            for jndx in range(indx+1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx +1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break
        
        for indx in range(self.num_qubits-1):
            if len(illegal_action[indx])==0:
                illegal_action[indx] = illegal_action[indx+1]
                illegal_action[indx+1] = []
        
        illegal_action_decode = []
        for key, contain in self.action_dict.items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        return illegal_action_decode
        # return []




if __name__ == "__main__":
    pass
