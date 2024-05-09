'''
© Edgar Andrade 2023
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Class to run, renderize, train and test agents
over environments.
-----------------------------------------------
'''
import gymnasium as gym
import environments as E
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from typing import Dict
from utils.interaction import Episode, Experiment
import agents.TableAgents as agents
import environments.envs as E
from os import path
from pathlib import Path
from utils.plot_utils import Plot

gym_env_list = ['FrozenLake-v1', 'Blackjack-v1', 'Taxi-v3']
own_env_list = ['ABC', 'TriquiEnv']


class Performer :
    '''
    Class to train and run an agent in an environment.
    '''
    def __init__(self,\
                env_name:str,\
                env_parameters:Dict[str, any],\
                state_interpreter:any,\
                agent_name:str,\
                agent_parameters:Dict[str,any],\
                deep:bool=False
                ) -> None:
        self.env_name = env_name
        self.env_parameters = env_parameters
        self.state_interpreter = state_interpreter
        self.agent_name = agent_name
        self.agent_parameters = agent_parameters
        self.deep = deep
        self.consolidate_folders()

    def consolidate_folders(self):
        self.file_name = f'{self.agent_name}_in_{self.env_name}'
        self.image_folder = Path.cwd() / Path('..').resolve() / Path('images', self.file_name)
        self.image_folder.mkdir(parents=True, exist_ok=True)
        self.data_folder = Path.cwd() / Path('..').resolve() / Path('data', self.file_name)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.model_folder = Path.cwd() / Path('..').resolve() / Path('models', self.file_name)
        self.model_folder.mkdir(parents=True, exist_ok=True)
        self.video_folder = Path.cwd() / Path('..').resolve() / Path('videos', self.file_name)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.extension = '.pt' if self.deep else '.json'
        self.file_model = path.join(self.model_folder, f'{self.file_name}{self.extension}')
        self.file_csv = path.join(self.data_folder, f'{self.file_name}.csv')
        self.file_png = path.join(self.image_folder, f'{self.file_name}.png')
        self.file_losses = path.join(self.image_folder, f'{self.file_name}_losses.png')
        self.file_test = path.join(self.image_folder, f'{self.file_name}_test.png')
        self.file_compare_hist = path.join(self.image_folder, f'comparison_hist.png')
        self.file_compare_rew = path.join(self.image_folder, f'comparison_rew.png')

    def load_env(self, render_mode):
        '''
        Load environment. If using gymnasium environments, render mode
        is different for training (None) than for running (rgb_array). Render
        mode can only be set when instantiating the environment.
        '''
        if self.env_name in gym_env_list:
            self.environment = gym.make(self.env_name, 
                                        render_mode=render_mode,
                                        **self.env_parameters)
        elif self.env_name in own_env_list:
            if self.env_parameters is not None:
              exec(f'self.environment = E.{self.env_name}(self.env_parameters)')
            else:
              exec(f'self.environment = E.{self.env_name}()')
        else:
            raise Exception(f'Environment {self.env_name} unknown. Please modify Performer.load_env() to include it.')

    def load_agent(self, from_file:bool=False):
        '''
        Load agent from name
        '''
        exec(f'self.agent = agents.{self.agent_name}(self.agent_parameters)')
        if from_file:
          print(f'Loading agent from {self.file_model}')
          self.agent.load(file=self.file_model)
    
    def save_agent(self):
        try:
            self.agent.save(file=self.file_model)
        except Exception as e:
            print('\n\tAn error occurred:\n\t', e,'\n')
            pass

    def shutdown_agent_exploration(self) -> (float, np.ndarray):
        backup_epsilon = deepcopy(self.agent.epsilon)
        backup_policy = deepcopy(self.agent.policy)
        self.agent.epsilon = 0
        for s in range(self.agent.nS):
           self.agent.update_policy(s)
        return backup_epsilon, backup_policy

    def run(self, 
            from_file:bool=False,
            no_exploration:bool=False,
            visual:bool=True, 
            to_video:bool=False,
            num_rounds:int=200):
        '''
        Run the agent on the environment and displays the behavior.
        Agent does not learn.
        Input:
          - from_file (bool), if true, attemts to load the agent from file
          - no_exploration (bool), if true, makes epsilon = 0
          - visual (bool),
            True: displays the environment as in a video using environment render
            False: displays the behavioral data in the terminal step by step
          - num_rounds (int), number of rounds to display
        '''
        # Load agent from name
        self.load_agent(from_file=from_file)
        # self.agent.debug = True # Uncomment for debugging
        if no_exploration:
          backup_epsilon, backup_policy = self.shutdown_agent_exploration()
        if visual:
          '''
          To display the environment as in a video
          '''
          # Create environment
          self.load_env(render_mode='rgb_array')
          try:
            self.environment._max_episode_steps = num_rounds
          except:
             pass
          # Create episode
          episode = Episode(environment=self.environment,\
                            env_name=self.env_name,\
                            agent=self.agent,\
                            model_name=self.agent_name,\
                            num_rounds=num_rounds,\
                            state_interpreter=self.state_interpreter)
          episode.renderize(to_video=to_video,
                            file=self.video_folder)
        else:
          '''
          To display data information in the terminal
          '''
          # Create environment
          self.load_env(render_mode=None)
          try:
            self.environment._max_episode_steps = num_rounds
          except:
             pass
          # Create episode
          episode = Episode(environment=self.environment,\
                            env_name=self.env_name,\
                            agent=self.agent,\
                            model_name=self.agent_name,\
                            num_rounds=num_rounds,\
                            state_interpreter=self.state_interpreter
                            )
          df = episode.run(verbose=4, learn=False)
          self.data = df
        print('Number of rounds:', len(episode.agent.rewards) - 1)
        print('Total reward:', np.nansum(episode.agent.rewards))
        if no_exploration:
          self.agent.epsilon = backup_epsilon
          self.agent.policy = backup_policy
            
    def train(self, 
              num_rounds:int=200, 
              num_episodes:int=500, 
              from_file:bool=False):
        '''
        Trains agent.
        '''
        # Load agent from name
        self.load_agent(from_file=from_file)
        # Create environment
        self.load_env(render_mode=None)
        try:
          self.environment._max_episode_steps = num_rounds
        except:
            pass
        # Create episode
        episode = Episode(environment=self.environment,\
                          env_name=self.env_name,\
                          agent=self.agent,\
                          model_name=self.agent_name,\
                          num_rounds=num_rounds,\
                          state_interpreter=self.state_interpreter
                          )
        # Train agent
        df = episode.simulate(num_episodes=num_episodes, file=self.file_csv)
        print(f'Data saved to {self.file_csv}')
        # Save agent to file
        self.save_agent()
        print(f'Agent saved to {self.file_model}')
        # Plot results
        p =  Plot(df)
        if num_episodes == 1:
          p.plot_round_reward(file=self.file_png)    
        else:
          p.plot_rewards(file=self.file_png) 
        print(f'Plot saved to {self.file_png}')
        # Save losses if agent uses NN
        if hasattr(self.agent.Q, 'losses'):
          losses = self.agent.Q.losses
          fig, ax = plt.subplots(figsize=(4,3.5))
          ax = sns.lineplot(x=range(len(losses)), y=losses)
          ax.set_xlabel("Epoch",fontsize=14)
          ax.set_ylabel("Loss",fontsize=14)
          plt.savefig(self.file_losses, dpi=300, bbox_inches="tight")

    def test(self, 
             from_file:bool=True, 
             num_rounds:int=200, 
             num_episodes:int=100):
        '''
        Test the trained agent.
        '''
        # Load agent from name
        self.load_agent(from_file=from_file)
        # Shutdown exploration
        backup_epsilon, backup_policy = self.shutdown_agent_exploration()
        # Create environment
        self.load_env(render_mode=None)
        try:
          self.environment._max_episode_steps = num_rounds
        except:
            pass
        # Create episode
        episode = Episode(environment=self.environment,\
                          env_name=self.env_name,\
                          agent=self.agent,\
                          model_name=self.agent_name,\
                          num_rounds=num_rounds,\
                          state_interpreter=self.state_interpreter
                          )
        # Run simulation
        df = episode.simulate(num_episodes=num_episodes, learn=False)
        # Plot results
        p = Plot(df)
        p.plot_histogram_rewards(self.file_test)
        print(f'Plot saved to {self.file_test}')
        self.agent.epsilon = backup_epsilon
        self.agent.policy = backup_policy
         
    def sweep(self, 
              parameter:str, 
              values:list, 
              num_rounds:int=200, 
              num_episodes:int=100,
              num_simulations:int=10):
        '''
        Runs a sweep over the specified parameter 
        with the specified values.
        '''
        # Load agent from name
        self.load_agent()
        # Creates environment
        self.load_env(render_mode=None)
        # Creates experiment
        experiment = Experiment(environment=self.environment,\
                                env_name=self.env_name,\
                                num_rounds=num_rounds,\
                                num_episodes=num_episodes,\
                                num_simulations=num_simulations,\
                                state_interpreter=self.state_interpreter
                  )
        # Run sweep
        experiment.run_sweep1(agent=self.agent, \
                       name=self.agent_name, \
                       parameter=parameter, \
                       values=values, \
                       measures=['reward'])
        # Plot results
        p = Plot(experiment.data)
        print('Plotting...')
        p.plot_rewards(self.file_compare_rew)
        print(f'Plot saved to {self.file_compare_rew}')
        
    def compare_test(self, 
                     agent_vs_name:str,
                     agent_vs_parameters:Dict,
                     num_rounds:int=200, 
                     num_episodes:int=100):
        '''
        Runs a comparison of two agents
        over an environment.
        Agents are loaded from file.
        '''
        # Load agent 1
        self.load_agent(from_file=True)
        self.shutdown_agent_exploration()
        agent1 = deepcopy(self.agent)
        # Load vs agent
        backup_agent_name = self.agent_name
        backup_agent_parameters = deepcopy(self.agent_parameters)
        self.agent_name = agent_vs_name
        self.agent_parameters = agent_vs_parameters
        self.consolidate_folders()
        try:
          self.load_agent(from_file=True)
        except Exception as e:
          print(e)
          print(f'An agent of class {agent_vs_name} is required.\nRun a performer on such an agent first.') 
        self.shutdown_agent_exploration()
        agent2 = deepcopy(self.agent)
        self.agent_name = backup_agent_name
        self.agent_parameters = backup_agent_parameters
        self.consolidate_folders()
        # Create environment
        self.load_env(render_mode=None)
        # Create experiment
        experiment = Experiment(environment=self.environment,\
                                env_name=self.env_name,\
                                num_rounds=num_rounds,\
                                num_episodes=num_episodes,\
                                num_simulations=1,\
                                state_interpreter=self.state_interpreter
                  )
        # Run sweep
        experiment.run_experiment(agents=[agent1, agent2], \
                                  names=[self.agent_name, agent_vs_name], \
                                  measures=['hist_reward'],\
                                  learn=False)
        self.data = experiment.data
        # Plot results
        p = Plot(experiment.data)
        print('Plotting...')
        p.plot_histogram_rewards(self.file_compare_hist)
        print(f'Plot saved to {self.file_compare_hist}')
