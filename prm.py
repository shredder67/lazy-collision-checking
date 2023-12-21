import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEFAULT_START_ST = (np.array([10, 10]), 0) # x, y, angle
DEFAULT_GOAL_ST = (np.array([90, 90]), -90) 
DEFAULT_CONFIG = {
    'env_size': (100, 100),
    'obst_positions': [[50, 50]],
    'obst_radi': [1],
}

SIMPLE_CONFIG = {
    'env_size': (100, 100),
    'obst_positions': [[30, 25], [30, 75], [70, 25], [70, 75]],
    'obst_radi': [15, 15, 15, 15]
}


class State:
    RECT_WIDTH=5
    RECT_HEIGHT=9

    def __init__(self, pos: np.ndarray, angle: float):
        """
        Represents a state of rectangle agent

        ### Parameters:
        pos - coordinates of rectangle center
        angle - rotation angle in range (-180, 180)
        """
        self._center_coors = pos
        self._angle = np.deg2rad(angle)
        self._vertices = State._calculate_vertices_coordinates(self._center_coors, self._angle)

    def to_list(self):
        return [self._center_coors[0],
                self._center_coors[1],
                self._angle]


    @staticmethod
    def _calculate_vertices_coordinates(center_coors, angle):
        """Translates state into coordinates of 4 rectangle vertices coordinates"""
        translated_vertices = np.array([
            [-State.RECT_WIDTH / 2, -State.RECT_HEIGHT / 2],
            [-State.RECT_WIDTH  / 2, State.RECT_HEIGHT / 2],
            [State.RECT_WIDTH  / 2, -State.RECT_HEIGHT / 2],
            [State.RECT_WIDTH  / 2, State.RECT_HEIGHT / 2],
        ], dtype=np.float32)

        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=np.float32)
        
        rotated_v = np.array([rot @ translated_vertices[idx] for idx in range(4)], dtype=np.float32)
        return rotated_v + center_coors
    
    @staticmethod
    def generate_lin_space(start_state: 'State', end_state: 'State', n: int):
        dist_step = (end_state - start_state) / n
        res = [start_state]
        for _ in range(n):
            res.append(res[-1] + dist_step)
        return res

    def __sub__(self, other: 'State'):
        pos_delta = self._center_coors - other._center_coors
        angle_delta = self._angle - other._angle
        angle_delta = (angle_delta + np.pi) % (2 * np.pi) - np.pi
        return State(pos_delta, angle_delta)
    
    def __add__(self, other: 'State'):
        new_pos = self._center_coors + other._center_coors
        new_angle = self._angle + other._angle
        new_angle = (new_angle + np.pi) % (2 * np.pi) - np.pi
        return State(new_pos, new_angle)
    
    def __mul__(self, scalar: float):
        if type(scalar) is not float:
            raise ValueError('Scalar should be float')
        new_pos = self._center_coors * scalar
        new_angle = self._angle * scalar
        new_angle = (new_angle + np.pi) % (2 * np.pi) - np.pi
        return State(new_pos, new_angle)
    
    def __eq__(self, other: 'State'):
        return np.allclose(self._center_coors, other._center_coors) and \
            np.allclose(self._angle, other._angle)
            
    def __hash__(self):
        return hash((self._center_coors.tobytes(), self._angle))
    

def distance(state1: State, state2: State) -> float:
    """Returns distance between two states"""
    state_dif = state1 - state2
    return np.linalg.norm(state_dif._center_coors) + \
        np.abs(state_dif._angle)
        

class Environment:
    """
    Continious environment with certain amount of circle obstacles with set
    coordinates and radiuses
    """
    def __init__(self, state, env_size, obst_positions, obst_radi):
        self._state = state
        self._env_size = env_size
        self._obstacles = np.array(obst_positions)
        self._radiuses = np.array(obst_radi)

    @classmethod
    def from_config(cls, st_state, config=DEFAULT_CONFIG):
        return cls(state=st_state, **config)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state:State) -> bool:
        self._state = new_state

    def check_collision(self, state_to_check: State) -> bool:
        for v in state_to_check._vertices:
            for idx, obst_coors in enumerate(self._obstacles):
                if np.linalg.norm(v - obst_coors) < self._radiuses[idx]:
                    return True
        return False

    def render(self, goal_state=None, path=None) -> None:
        plt.figure(figsize=(7, 7))
        plt.xlim(0, self._env_size[0])
        plt.ylim(0, self._env_size[1])

        # render current state
        plt.gca().add_patch(
            Rectangle(
                self._state._vertices[0], 
                State.RECT_WIDTH, 
                State.RECT_HEIGHT,
                np.rad2deg(self._state._angle),
                facecolor='red',
                fill=True,
            )
        )

        # additionally render goal state
        if goal_state:
            plt.gca().add_patch(
            Rectangle(
                goal_state._vertices[0], 
                State.RECT_WIDTH, 
                State.RECT_HEIGHT,
                np.rad2deg(goal_state._angle),
                facecolor='green',
                fill=True,
                alpha=0.3,
                label='goal state'
            )
        )
            
        plt.scatter(self._state._center_coors[0], self._state._center_coors[0], c='orange')

        # render obstacles
        for idx, o in enumerate(self._obstacles):
            plt.gca().add_patch(
                plt.Circle((o[0], o[1]), self._radiuses[idx], fill=True)
            )
            
        if path:
            for i in range(len(path) - 1):
                plt.plot([path[i]._center_coors[0], path[i+1]._center_coors[0]], 
                         [path[i]._center_coors[1], path[i+1]._center_coors[1]], 
                         color='blue')                
        plt.legend()
        plt.grid()
        plt.show()
        
class PRMStar:
    def __init__(self, environment, num_samples, k):
        self.environment = environment
        self.num_samples = num_samples
        self.k = k
        self.G = nx.Graph()

    def sample_state(self):
        x = np.random.uniform(0, self.environment._env_size[0])
        y = np.random.uniform(0, self.environment._env_size[1])
        angle = np.random.uniform(-180, 180)
        return State(np.array([x, y]), angle)

    def build_roadmap(self, start_state, goal_state):
        points = [self.sample_state() for _ in range(self.num_samples)] + [start_state, goal_state]
        tree = KDTree([point._center_coors for point in points])

        for i, point in enumerate(points):
            distances, indices = tree.query(point._center_coors, self.k + 1)
            for dist, idx in zip(distances, indices):
                if i != idx and dist > 0:
                    neighbor = points[idx]
                    if self.is_path_free(point, neighbor):
                        self.G.add_edge(point, neighbor, weight=distance(point, neighbor))

    def is_path_free(self, state1, state2):
        num_checks = 10
        for i in range(num_checks + 1):
            t = i / float(num_checks)
            interpolated_state = self.interpolate_state(state1, state2, t)
            if self.environment.check_collision(interpolated_state):
                return False
        return True

    def interpolate_state(self, state1, state2, t):
        new_pos = (1 - t) * state1._center_coors + t * state2._center_coors
        new_angle = (1 - t) * state1._angle + t * state2._angle
        return State(new_pos, new_angle)

    def find_path(self, start_state, goal_state):
        if self.is_path_free(start_state, goal_state):
            self.G.add_edge(start_state, goal_state, weight=distance(start_state, goal_state))
        try:
            path = nx.shortest_path(self.G, source=start_state, target=goal_state, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None

    def render(self, path):
        self.environment.render(goal_state=path[-1], path=path)
        plt.show()
        
env = Environment.from_config(State(*DEFAULT_START_ST), SIMPLE_CONFIG)
prm_star = PRMStar(env, num_samples=500, k=15)

start_state = State(*DEFAULT_START_ST)
goal_state = State(*DEFAULT_GOAL_ST)
prm_star.build_roadmap(start_state, goal_state)

path = prm_star.find_path(start_state, goal_state)

# Visualization
prm_star.render(path)                