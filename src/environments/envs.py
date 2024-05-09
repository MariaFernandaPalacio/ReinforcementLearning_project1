from environments.games import Triqui
from agents.MBagents import AlphaBetaAgent
from utils.interaction import EnvfromGameAndPl2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from os import path
from pathlib import Path

image_folder = Path.cwd() / Path('environments', 'imagenes')
image_folder.mkdir(parents=True, exist_ok=True)
to_array_folder = Path.cwd() / Path('environments', 'to_array')
to_array_folder.mkdir(parents=True, exist_ok=True)

class ABC():
    
    def __init__(self):
        self.nA = 2
        self.action_space = [0,1]
        self.nS = 3
        self.A = 0
        self.B = 1
        self.C = 2
        self.LEFT = 0
        self.RIGHT = 1
        P = {}
        P[self.A] = {a:[] for a in range(self.nA)}
        P[self.A][self.LEFT] = [(1, self.A, -1, False)]
        P[self.A][self.RIGHT] = [(0.1, self.A, -1, False), (0.9, self.B, -1, False)]
        P[self.B] = {a:[] for a in range(self.nA)}
        P[self.B][self.LEFT] = [(1, self.A, -1, False)]
        P[self.B][self.RIGHT] = [(0.1, self.B, -1, False), (0.9, self.C, 10, True)]
        P[self.C] = {a:[] for a in range(self.nA)}
        self.P = P
        self.dict_acciones = {self.LEFT:'LEFT', self.RIGHT:'RIGHT'}
        self.dict_states = {self.A:'A', self.B:'B', self.C:'C'}
        self.p_right = 0.9
        self.state = self.A
        
    def reset(self):
        self.state = self.A
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def render(self):
        print(f'Estado: {self.state}')

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {self.dict_states[s]}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    string += f'new_state:{self.dict_states[x[1]]}, '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string
    

class TriquiEnv(EnvfromGameAndPl2) :
    '''
    Environment for playing triqui against a minimax player.
    '''

    def __init__(self):
        triqui_base = Triqui()
        player2 = AlphaBetaAgent(game=triqui_base, 
                                 player=2, 
                                 max_lim=100)
        super().__init__(game=triqui_base, other_player=player2)
        self.list_acciones = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]



class GridworldEnv():
    """
    A 4x4 Grid World environment from Sutton's Reinforcement 
    Learning book chapter 4. Termial states are top left and
    the bottom right corner.
    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step until agent reaches a terminal state.
    """

    def __init__(self, shape=(4,4)):
        assert(shape[0] == shape[1])
        self.shape = shape
        self.nS = np.prod(self.shape)
        self.nA = 4
        self.action_space = list(range(self.nA))
        self.state = np.random.randint(1, self.nS - 2)
        self.NORTH = 0
        self.WEST = 1
        self.SOUTH = 2
        self.EAST = 3
        P = {}
        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}
            # Per state and action provide list as follows
            # P[state][action] = [(probability, next_state, reward, done)]
            # Assignment is obtained by means of method _transition_prob
            position = self._State2Car(s)
            P[s][self.NORTH] = self._transition_prob(position, [0, 1])
            P[s][self.WEST] = self._transition_prob(position, [-1, 0])
            P[s][self.SOUTH] = self._transition_prob(position, [0, -1])
            P[s][self.EAST] = self._transition_prob(position, [1, 0])
        # We expose the model of the environment for dynamic programming
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.dict_acciones = {0:"⬆", 1:"⬅", 2:"⬇", 3:"➡"}
        self.proportion = 5
        self.render_mode = None

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = np.clip(coord[0], 0, self.shape[0] - 1)
        coord[1] = np.clip(coord[1], 0, self.shape[1] - 1)
        return coord

    def _Car2State(self, casilla:tuple) -> int:
        X, Y = casilla
        return np.ravel_multi_index((Y, X), self.shape)

    def _State2Car(self, index:int) -> tuple:
        Y, X = np.unravel_index(index, self.shape)
        return (X, Y)

    def _transition_prob(self, current, delta):
        """
        Model Transitions. Prob is always 1.0.
        :param current: Current position on the grid as (x, y)
        :param delta: Change in position for transition
        :return: [(1.0, new_state, reward, done)]
        """
        # if stuck in terminal state
        current_state = self._Car2State(current)
        if current_state == self._Car2State((self.shape[0] - 1, 0)) or current_state == self._Car2State((0, self.shape[1] - 1)):
            return [(1.0, current_state, 0, True)]
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = self._Car2State(new_position)
        is_done = new_state == self._Car2State((self.shape[0] - 1, 0)) or new_state == self._Car2State((0, self.shape[1] - 1))
        return [(1.0, new_state, -1, is_done)]

    def reset(self):
        self.state = np.random.randint(1, self.nS - 2)
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def _find_figsize(self):
        x, y = self.shape
        if x == y:
            return (self.proportion,self.proportion)
        elif x > y:
            return (int(self.proportion*(x/y)),self.proportion)
        else:
            return (self.proportion,int(self.proportion*(y/x)))
        
    def _find_offset(self):
        return 1/(self.shape[0]*2), 1/(self.shape[1]*2)

    def render(self):
        # Dibuja el laberinto
        fig, axes = plt.subplots(figsize=self._find_figsize())
        # Dibujo el tablero
        step_x = 1./self.shape[0]
        step_y = 1./self.shape[1]
        tangulos = []
        # Borde del tablero
        tangulos.append(patches.Rectangle((0,0),0.998,0.998,\
                                            facecolor='xkcd:sky blue',\
                                            edgecolor='black',\
                                            linewidth=1))
        offsetX, offsetY = self._find_offset()
        #Poniendo las salidas
        for casilla in [(0,self.shape[1]-1), (self.shape[0]-1,0)]:
            X, Y = casilla
            file_salida = Path(image_folder, 'salida.png')
            arr_img = plt.imread(file_salida, format='png')
            image_salida = OffsetImage(arr_img, zoom=0.05)
            image_salida.image.axes = axes
            ab = AnnotationBbox(
                image_salida,
                [(X*step_x) + offsetX, (Y*step_y) + offsetY],
                frameon=False)
            axes.add_artist(ab)
		# Creo las líneas del tablero
        for j in range(self.shape[1]):
            # Crea linea horizontal en el rectangulo
            tangulos.append(patches.Rectangle(*[(0, j * step_y), 1, 0.008],\
            facecolor='black'))
        for j in range(self.shape[0]):
            # Crea linea vertical en el rectangulo
            tangulos.append(patches.Rectangle(*[(j * step_x, 0), 0.008, 1],\
            facecolor='black'))
        for t in tangulos:
            axes.add_patch(t)
        #Poniendo agente
        Y, X = np.unravel_index(self.state, self.shape)
        imagen_robot = Path(image_folder, 'robot.png')
        arr_img = plt.imread(imagen_robot, format='png')
        image_robot = OffsetImage(arr_img, zoom=0.125)
        image_robot.image.axes = axes
        ab = AnnotationBbox(
            image_robot,
            [(X*step_x) + offsetX, (Y*step_y) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        axes.axis('off')
        if self.render_mode == 'rgb_array':
            to_array_file = Path(to_array_folder, f'to_array.png')
            plt.savefig(to_array_file)
            return plt.imread(to_array_file)
        else:
            plt.show()
            return axes

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {s} at {np.unravel_index(s, self.shape)}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    Y, X = np.unravel_index(x[1], self.shape)
                    string += f'new_state:{x[1]} at ({X}, {Y}), '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string