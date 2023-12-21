import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from utils import sampler, collision_checker, line_intersects_circle, edge_collision_checker, euclidean_distance, plot

class LazyPRMStar:
    def __init__(self, sampler, collision_checker, distance_fn, x_range, y_range, num_samples, k, obstacles):
        self.sampler = sampler
        self.collision_checker = collision_checker
        self.distance_fn = distance_fn
        self.x_range = x_range
        self.y_range = y_range
        self.num_samples = num_samples
        self.k = k
        self.obstacles = obstacles
        self.G = nx.Graph()
        self.Glazy = nx.Graph()
        self.samples = []
        self.kdtree = None

    def sample_free(self):
        while True:
            sample = self.sampler(self.x_range, self.y_range)
  
            for obstacle in self.obstacles:
                if np.linalg.norm(np.array(sample) - np.array(obstacle['center'])) <= obstacle['radius']:
                    break
            return sample
          
    def add_sample(self, sample):
        self.samples.append(sample)
        self.G.add_node(sample)
        self.Glazy.add_node(sample)
        self.build_kdtree()

    def build_kdtree(self):
        self.kdtree = KDTree(self.samples)

    def nearest_neighbors(self, sample):
        if len(self.samples) <= self.k:
            return self.samples
        else:
            distances, indices = self.kdtree.query(sample, self.k)
            return [self.samples[i] for i in indices]

    def lazy_expand(self):
        new_sample = self.sample_free()
        self.add_sample(new_sample)
        for neighbor in self.nearest_neighbors(new_sample):
            self.Glazy.add_edge(new_sample, neighbor, weight=self.distance_fn(new_sample, neighbor))

   
    def lazy_update(self, start, goal):
        Cbest = float('inf')
        path = []
        while True:
            try:
                path = nx.shortest_path(self.Glazy, source=start, target=goal, weight='weight')
                Cpath = sum(self.Glazy[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                if Cpath < Cbest:
                    for u, v in zip(path[:-1], path[1:]):
                        if not self.G.has_edge(u, v):
                            if self.collision_checker(u, v, self.obstacles):
                                self.G.add_edge(u, v, weight=self.distance_fn(u, v))
                            else:
                                self.Glazy.remove_edge(u, v)
                                break
                    else:
                        Cbest = Cpath
                else:
                    break
            except nx.NetworkXNoPath:
                break
        return path if all(self.G.has_edge(u, v) for u, v in zip(path[:-1], path[1:])) else []

    def build_roadmap(self):
        for _ in range(self.num_samples):
            self.lazy_expand()
        self.build_kdtree()


    def query(self, start, goal):
        self.add_sample(start)
        self.add_sample(goal)
        self.build_roadmap()
        return self.lazy_update(start, goal)