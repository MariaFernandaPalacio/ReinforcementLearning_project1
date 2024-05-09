from seaborn import lineplot, histplot, heatmap, color_palette
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PlotGridValues :
    
    def __init__(self, shape:tuple, action_dict:dict):
        assert(len(shape) == 2)
        self.shape = shape
        self.action_dict = action_dict
        self.nA = len(action_dict.keys())
        
    def plot_policy(self, policy, V=None, ax=None):
        try:
            policy = np.flipud(np.array(policy).reshape(self.shape))
        except:
            raise Exception('¡Política no legible!')
        annotations = np.vectorize(self.action_dict.get)(policy)
        if V is None:
            values = np.zeros(self.shape)
        else:
            try:
                values = np.flipud(np.array(V).reshape(self.shape))
            except:
                raise Exception('Valores no legibles!')
        if ax is None:
            heatmap(
                values,
                annot=annotations,
                cbar=False,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=1.7,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                annot_kws={"fontsize": "xx-large"},
            ).set(title="Policy: action per state")
            plt.plot()
        else:
            heatmap(
                values,
                annot=annotations,
                cbar=False,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=1.7,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                annot_kws={"fontsize": "xx-large"},
                ax = ax
            ).set(title="Policy: action per state")

    def plot_V_values(self, V, ax=None):
        try:
            V = np.flipud(np.array(V).reshape(self.shape))
        except:
            raise Exception('Valores no legibles!')
        if ax is None:
            heatmap(
                V,
                annot=True,
                cbar=False,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=0.7,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                annot_kws={"fontsize": "x-large"},
            ).set(title="V-values")
        else:
            heatmap(
                V,
                annot=True,
                cbar=False,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=0.7,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                annot_kws={"fontsize": "x-large"},
                ax = ax
            ).set(title="V-values")

    def plot_policy_and_values(self, policy, V):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        self.plot_policy(policy, V, ax=ax[0])
        self.plot_V_values(V, ax=ax[1])
        ax[0].set_title('Policy', fontsize='18')
        ax[1].set_title('Values', fontsize='18')


class Plot :
    '''
    Gathers a number of frequently used visualizations.
    '''

    def __init__(self, data:pd.DataFrame):
        self.data = data

    def plot_rewards(self, file:str=None) -> plt.axis:
        '''
        Plots the reward per episode.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        assert('model' in self.data.columns)
        assert('environment' in self.data.columns)
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        data = self.data.copy()
        if 'simulation' in self.data.columns:
            data = data.groupby(['model', 'environment', 'simulation', 'episode'])["reward"].sum().reset_index()
        else:
            data = data.groupby(['model', 'environment', 'episode'])["reward"].sum().reset_index()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total reward')
        ax.grid()
        if vs_models:
            ax = lineplot(x='episode', y='reward', hue='model', data=data)
        else:
            ax = lineplot(x='episode', y='reward', data=data)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        return ax

    def plot_histogram_rewards(self, file:str=None) -> plt.axis:
        '''
        Plots a histogram with the sum of rewards per episode.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        assert('model' in self.data.columns)
        assert('environment' in self.data.columns)
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Sum of rewards')
        ax.set_ylabel('Frequency')
        ax.grid()
        if vs_models:
            df = self.data.groupby(['model','environment','episode']).reward.sum().reset_index()
            ax = histplot(x='reward', hue='model', data=df)
        else:
            df = self.data.groupby(['environment','episode']).reward.sum().reset_index()
            ax = histplot(x='reward', data=df)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        df = self.data.groupby(['environment','model','episode']).reward.sum().reset_index()
        total_reward = df.groupby('model').reward.mean()
        print('Average sum of rewards:\n', total_reward)
        df = self.data.groupby(['environment','model','episode']).done.sum().reset_index()
        df["done"] = df["done"].astype(int)
        termination = df.groupby('model').done.mean()*100
        print('\nEpisode termination percentage:\n', termination)
        return ax