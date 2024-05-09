import numpy as np
from typing import Dict
import json

class Agent :
    '''
    Defines the basic methods for all RL agents.
    '''

    def __init__(self, parameters:Dict[any, any]):
        self.parameters = parameters
        self.nS = self.parameters['nS']
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.epsilon = self.parameters['epsilon']
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))
        self.seed = None

    def make_decision(self):
        '''
        Agent makes a decision according to its policy.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        state = self.states[-1]
        weights = [self.policy[state, action] for action in range(self.nA)]
        action = np.random.choice(range(self.nA), p=weights)
        return action

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))

    def max_Q(self, s):
        '''
        Determines the max Q value in state s
        '''
        return max([self.Q[s, a] for a in range(self.nA)])

    def argmaxQ(self, s):
        '''
        Determines the action with max Q value in state s
        '''
        maxQ = self.max_Q(s)
        opt_acts = [a for a in range(self.nA) if self.Q[s, a] == maxQ]
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.choice(opt_acts)

    def update_policy(self, s):
        opt_act = self.argmaxQ(s)
        prob_epsilon = lambda action: 1 - self.epsilon if action == opt_act else self.epsilon/(self.nA-1)
        self.policy[s] = [prob_epsilon(a) for a in range(self.nA)]

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        TO BE DEFINED BY SUBCLASS
        '''
        pass

    def save(self, file:str) -> None:
        # Serializing json
        dictionary = {'policy':self.policy.tolist(),
                      'Q':self.Q.tolist()}
        json_object = json.dumps(dictionary, indent=4)
        # Writing to file
        with open(file, "w") as outfile:
            outfile.write(json_object)
        outfile.close()

    def load(self, file:str):
        # Opening JSON file
        with open(file, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        self.reset()
        self.policy = np.array(json_object['policy'])
        self.Q = np.array(json_object['Q'])
        openfile.close()