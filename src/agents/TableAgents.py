'''
© Edgar Andrade 2018
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Reinforce-learning agents

Includes:
    - MC, a learning rule with Monte Carlo optimization.
    - SARSA, a learning rule 
    - Q_learning, a learning rule
-----------------------------------------------
'''
import numpy as np
from agents.BaseAgent import Agent

dash_line = '-'*20

class SARSA(Agent) :
    '''
    Implements a SARSA learning rule.
    '''

    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.alpha = self.parameters['alpha']
        self.debug = False
   
    def update(self, next_state, reward, done, gamma:float=1):
        '''
        Agent updates its model.
        '''
        # obtain previous state
        state = self.states[-1]
        # obtain previous action
        action = self.actions[-1]
        # Get next_action
        next_action = self.make_decision()
        # Find bootstrap
        estimate = reward + gamma * self.Q[next_state, next_action]  # recompensa más descuento por valor del siguiente estado
        # Obtain delta
        delta = estimate - self.Q[state, action] # Diferencia temporal: estimado menos valor del estado actual
        # Update Q value
        prev_Q = self.Q[state, action]
        self.Q[state, action] = prev_Q + self.alpha * delta # Actualizar en la dirección de delta por una fracción alfa
        # Update policy
        self.update_policy(state)
        if self.debug:
            print('')
            print(dash_line)
            print(f'Learning log:')
            print(f'state:{state}')
            print(f'action:{action}')
            print(f'reward:{reward}')
            print(f'estimate:{estimate}')
            print(f'Previous Q:{prev_Q}')
            print(f'delta:{delta}')
            print(f'New Q:{self.Q[state, action]}')

class Q_learning(Agent) :
    '''
    Implements a Q-learning rule.
    '''

    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.alpha = self.parameters['alpha']
        self.debug = False
   
    def update(self, next_state, reward, done, gamma:float=1):
        '''
        Agent updates its model.
        '''
        # obtain previous state
        state = self.states[-1] # Aquí estado previo
        # obtain previous action
        action = self.actions[-1]
        # Find bootstrap
        maxQ = self.max_Q(next_state) 
        
        estimate = reward + (gamma * maxQ) # Calcula el estimado
        # Obtain delta
        delta = estimate - self.Q[state, action] # Calcula el delta
        # Update Q value
        prev_Q = self.Q[state, action]
        self.Q[state, action] = prev_Q + self.alpha * delta # Actualiza el valor
        # Update policy
        self.update_policy(state) # Actualizar la política en el estado        
        if self.debug:
            print('')
            print(dash_line)
            print(f'Learning log:')
            print(f'state:{state}')
            print(f'action:{action}')
            print(f'reward:{reward}')
            print(f'estimate:{estimate}')
            print(f'Previous Q:{prev_Q}')
            print(f'delta:{delta}')
            print(f'New Q:{self.Q[state, action]}') 