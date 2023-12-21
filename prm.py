import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


from environment import State, Environment, distance


        
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