import numpy as np
import random
import torch
import sys
import os
import argparse
import pathlib
import copy
from utils.utils_synthesized import get_config
from environments.environment_nd_synthesized_1 import CircuitEnv
import agents
import time
torch.set_num_threads(1)
import pickle

class Saver:
    def __init__(self, results_path, experiment_seed):
        self.stats_file = {'train': {}, 'test': {}}
        self.exp_seed = experiment_seed
        self.rpath = results_path

    def get_new_episode(self, mode, episode_no):
        if mode == 'train':
            self.stats_file[mode][episode_no] = {'loss': [],
                                                 'actions': [],
                                                 'errors': [],
                                                 'errors_noiseless':[],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [], 
                                                 'opt_ang': [],
                                                 'time' : [],
                                                'save_circ' : [],
                                                'rewards' : []
                                                 }
        elif mode == 'test':
            self.stats_file[mode][episode_no] = {'actions': [],
                                                 'errors': [],
                                                 'errors_noiseless':[],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [],
                                                 'opt_ang': [],
                                                 'time' : []
                                                 }

    def save_file(self):
        # PREVIOUSLY
        # np.save(f'{self.rpath}/summary_{self.exp_seed}.npy', self.stats_file)
        
        # NOW!
        with open(f'{self.rpath}/summary_{self.exp_seed}.pickle', 'wb') as f:
            pickle.dump(self.stats_file, f, pickle.HIGHEST_PROTOCOL)

    def validate_stats(self, episode, mode):
        assert len(self.stats_file[mode][episode]['actions']) == len(self.stats_file[mode][episode]['errors'])

    
def modify_state(state,env):
    
        
    if conf['agent']['en_state']:
        
        state = torch.cat((state, torch.tensor(env.prev_energy,dtype=torch.float,device=device).view(1)))
        
    if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
        state = torch.cat((state, torch.tensor(env.done_threshold,dtype=torch.float,device=device).view(1)))
    # print('state size in modified state')
    # print(state.shape)  
    return state


def agent_test(env, agent, episode_no, seed, output_path,threshold):
    """ Testing function of the trained agent. """    
    agent.saver.get_new_episode('test', episode_no)
    state = env.reset()
    state = modify_state(state, env)
    current_epsilon = copy.copy(agent.epsilon)
    agent.policy_net.eval()

    for t in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        
        agent.epsilon = 0
        with torch.no_grad():
            action, _ = agent.act(state, ill_action_from_env)
            assert type(action) == int
            agent.saver.stats_file['test'][episode_no]['actions'].append(action)
        next_state, reward, done = env.step(agent.translate[action],train_flag=False)
        next_state = modify_state(next_state, env)
        state = next_state.clone()
        assert type(env.error) == float 
        agent.saver.stats_file['test'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['test'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['test'][episode_no]['opt_ang'].append(env.opt_ang_save)
        
        if done:
            
            agent.saver.stats_file['test'][episode_no]['done_threshold'] = env.done_threshold
            agent.saver.stats_file['test'][episode_no]['bond_distance'] = env.current_bond_distance
            errors_current_bond = [val['errors'][-1] for val in agent.saver.stats_file['test'].values()
                                   if val['done_threshold'] == env.done_threshold]
            if len(errors_current_bond) > 0 and min(errors_current_bond) > env.error:
                torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_model.pth")
                torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_optim.pth")
            agent.epsilon = current_epsilon
            agent.saver.validate_stats(episode_no, 'test')
            
            return reward, t
        

def one_episode(episode_no, env, agent, episodes):
    """ Function preforming full trainig episode."""
    t0 = time.time()
    agent.saver.get_new_episode('train', episode_no)
    state = env.reset()
    agent.saver.stats_file['train'][episode_no]['bond_distance'] = env.current_prob
    agent.saver.stats_file['train'][episode_no]['done_threshold'] = env.done_threshold
    
    state = modify_state(state, env)
    agent.policy_net.train()
    rewards4return = []
    
    for itr in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        
        action, _ = agent.act(state, ill_action_from_env)
        assert type(action) == int
        agent.saver.stats_file['train'][episode_no]['actions'].append(action)
        
        next_state, reward, done = env.step(agent.translate[action])
        
        next_state = modify_state(next_state, env)
        agent.remember(state, 
                       torch.tensor(action, device=device), 
                       reward,
                       next_state,
                       torch.tensor(done, device=device))
        state = next_state.clone()
        rewards4return.append(float(reward.clone()))

        assert type(env.error) == float
        agent.saver.stats_file['train'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['train'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['train'][episode_no]['opt_ang'].append(env.opt_ang_save)
        # agent.saver.stats_file['train'][episode_no]['save_circ'].append(env.save_circ)
        agent.saver.stats_file['train'][episode_no]['rewards'].append(env.rwd)
        
        agent.saver.stats_file['train'][episode_no]['time'].append(time.time()-t0)

        if agent.memory_reset_switch:            
           if env.error < agent.memory_reset_threshold:
               agent.memory_reset_counter += 1
           if agent.memory_reset_counter == agent.memory_reset_switch:
               agent.memory.clean_memory()
               agent.memory_reset_switch = False
               agent.memory_reset_counter = False
               
  
        if done:
            # print('time:', time.time()-t0)
            if episode_no%20==0:
                print("episode: {}/{}, score: {}, e: {:.2}, rwd: {} \n"
                        .format(episode_no, episodes, itr+1, agent.epsilon, reward),flush=True)
            break 
        
        if len(agent.memory) > conf['agent']['batch_size']:
            if "replay_ratio" in conf['agent'].keys():
                if  itr % conf['agent']["replay_ratio"]==0:
                    loss = agent.replay(conf['agent']['batch_size'])
            else:
                loss = agent.replay(conf['agent']['batch_size'])         
            assert type(loss) == float
            agent.saver.stats_file['train'][episode_no]['loss'].append(loss)
            agent.saver.validate_stats(episode_no, 'train')
            
            

def train(agent, env, episodes, seed, output_path,threshold):
    """Training loop"""
    threshold_crossed = 0
    for e in range(episodes):
        
        one_episode(e, env, agent, episodes)
        
        if e %50==0 and e > 0:
            agent.saver.save_file()
            torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_model.pth")
            torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_optim.pth")
            torch.save( {i: a._asdict() for i,a in enumerate(agent.memory.memory)}, f"{output_path}/thresh_{threshold}_{seed}_replay_buffer.pth")
        if env.error <= 0.0016:
            threshold_crossed += 1
            np.save( f'threshold_crossed', threshold_crossed )

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproduction')
    parser.add_argument('--config', type=str, default='h_s_2', help='Name of configuration file')
    parser.add_argument('--experiment_name', type=str, default='lower_bound_energy/', help='Name of experiment')
    parser.add_argument('--gpu_id', type=int, default=0, help='Set specific GPU to run experiment [0, 1, ...]')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':

    args = get_args(sys.argv[1:])


    results_path ="results/"
    pathlib.Path(f"{results_path}{args.experiment_name}{args.config}").mkdir(parents=True, exist_ok=True)
    # device = torch.device(f"cuda:{args.gpu_id}")
    device = torch.device(f"cpu:0")
    
    
    conf = get_config(args.experiment_name, f'{args.config}.cfg')

    loss_dict, scores_dict, test_scores_dict, actions_dict = dict(), dict(), dict(), dict()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    

    actions_test = []
    action_test_dict = dict()
    error_test_dict = dict()
    error_noiseless_test_dict=dict()

    
    """ Environment and Agent initialization"""
    environment = CircuitEnv(conf, device=device)
    agent = agents.__dict__[conf['agent']['agent_type']].__dict__[conf['agent']['agent_class']](conf, environment.action_size, environment.state_size, device)
    agent.saver = Saver(f"{results_path}{args.experiment_name}{args.config}", args.seed)

    if conf['agent']['init_net']: 
        PATH = f"{results_path}{conf['agent']['init_net']}{args.seed}"
        agent.policy_net.load_state_dict(torch.load(PATH+f"_model.pth"))
        agent.target_net.load_state_dict(torch.load(PATH+f"_model.pth"))
        agent.optim.load_state_dict(torch.load(PATH+f"_optim.pth"))
        agent.policy_net.eval()
        agent.target_net.eval()

        replay_buffer_load = torch.load(f"{PATH}_replay_buffer.pth")
        for i in replay_buffer_load.keys():
            agent.remember(**replay_buffer_load[i])

        if not conf['agent']['epsilon_restart']:
            agent.epsilon = agent.epsilon_min

    train(agent, environment, conf['general']['episodes'], args.seed, f"{results_path}{args.experiment_name}{args.config}",conf['env']['accept_err'])
    agent.saver.save_file()
            
    torch.save(agent.policy_net.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_model.pth")
    torch.save(agent.optim.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_optim.pth")
