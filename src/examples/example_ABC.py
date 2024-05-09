'''
© Edgar Andrade 2018
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Examples with the ABC environment
-----------------------------------------------
'''

from utils.performer import Performer
from utils.interpreters import id_state

agent_parameters = {'nA':2,
                    'nS':3,
                    'gamma':0.8,
                    'epsilon':0.01,
                    'alpha':0.1}
perf = Performer(env_name='ABC',
                 env_parameters=None,
                 state_interpreter=id_state,
                 agent_name='SARSA',
                 agent_parameters=agent_parameters)

def check_run():
    print('Running check on ABC environment...')
    perf.run(visual=False)
    print('Done!')

def train():
    print('Training agent on ABC...')
    perf.train()
    print('Done!')

def test():
    print('Testing agent on ABC...')
    perf.test()
    print('Done!')

def sweep_epsilon():
    print('Running parameter sweep on ABC...')
    perf.sweep(parameter='epsilon',
               values=[0, 0.01, 0.1],
               num_simulations=100)
    print('Done!')
