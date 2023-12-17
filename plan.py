from typing import Callable, List
from queue import PriorityQueue
from collections import defaultdict

import numpy as np

from environment import State, Environment


class Planner:
    """Basic planner class"""
    def __init__(self,
                 env: Environment,
                 distance_fn: Callable,
                 max_angle_step: float=10.0,
                 max_move_step: float=3.0):
        """
        :param env: manipulator environment
        :param distance_fn: function distance_fn(state1, state2) -> float
        :param max_angle_step: max allowed step for each joint in degrees
        """
        self._env = env
        self._distance_fn = distance_fn
        self._max_angle_step = max_angle_step

    def plan(self,
             start_state: State,
             goal_state: State) -> List[State]:
        raise NotImplementedError


class LazyRRGPlanner(Planner):
    def __init__(self, env: Environment,
                 distance_fn: Callable,
                 env_sampler: Callable,
                 max_angle_step: float=10.0,
                 max_move_step: float=3.0):
        super().__init__(env, distance_fn, max_angle_step, max_move_step)
        self.env_sampler = env_sampler
        self.G = {}
        self.G_lazy = {}
        self.costs = {}
        self.predecessors = {}
        

    def _construct_plan(self, start_state, goal_state):
        path = self._path_to(goal_state)
        assert path[0] == start_state
        return path # ??? seems to be correct, but need to test

    def _find_nearest(self, state, G):
        mindist, argmin = np.inf, None
        for st_cand in G.keys():
            d = self._distance_fn(state, st_cand) 
            if d < mindist:
                mindist = d
                argmin = st_cand
        return argmin

    def _lazy_expand(self):
        q_rand = self.env_sampler()
        q_near = self._find_nearest(q_rand, self.G_lazy)
        q_delta = q_rand - q_near # think about this as vector
        
        # ensure constraints
        q_delta._center_coors = min(self._max_move_step, q_delta._center_coors)
        q_delta._angle = min(self._max_angle_step, q_delta._angle)

        # TODO: consider r(i) "r-ball strategy"

        q = q_near + q_delta
        if self._env.check_collision(q): # if state is collision free
            self.G_lazy[q_near].append(q_near)
            self.costs[q] = np.inf
            self.predecessors[q] = None
            self.G_lazy[q] = [q_near]
            for v in self.G_lazy[q_near]: # add to all neighbors
                self.G_lazy[q].append(v)
                self.G[v].append(q)
                if self.costs[v] + self._distance_fn(v, q) < self.costs[q]:
                    self.costs[q] = self.costs[v] + self._distance_fn(v, q)
                    self.predecessors[q] = v
                    # TODO: backprop cost update (two-way)    

    def _path_to(self, state):
        """backtrack path through self.predecessors"""
        path = [state]
        while self.predecessors[path[-1]] is not None:
            path.append(self.predecessors[path[-1]])
        return path[::-1]
    
    def _is_visible(self, u, v):
        inter_states = State.generate_lin_space(u, v, 10)
        for st in inter_states:
            if self._env.check_collision(st):
                return False
        return True

    def _lazy_update(self, c_best):
        q_g = None
        all_edges_in_G = True
        while True:
            q_g = min(self.G.keys(), key=lambda st: self.costs[st])
            if self.costs[q_g] < c_best:
                p = self._path_to(q_g)
                for (u, v) in zip(p[:-1], p[1:]):
                    if v not in self.G[u] and self._is_visible(u, v):
                        self.G[u].append(v)
                        self.G[v].append(u)
                    else:
                        self.G_lazy[u].remove(v)
                        self.G_lazy[v].remove(u)
                        if v in self.predecessors:
                            # search for least cost parent
                            min_cost = np.inf
                            min_parent = None
                            for parent in self.G_lazy[v]:
                                if self.costs[parent] + self._distance_fn(parent, v) < min_cost:
                                    min_cost = self.costs[parent] + self._distance_fn(parent, v)
                                    min_parent = parent
                            self.predecessors[v] = min_parent
                            self.costs[v] = min_cost
                        all_edges_in_G = False
                        break
                if all_edges_in_G:
                    c_best = self.costs[q_g]
            if q_g is None or c_best == self.costs[q_g]:
                break
                
        

    def plan(self, start_state, goal_state, N=1000):
        self.G[start_state] = []
        self.G_lazy[start_state] = []
        self.G[goal_state] = []
        self.G_lazy[goal_state] = []

        self.costs[start_state] = 0
        self.predecessors[start_state] = None


        # TODO: consider prunning (would require to add it to every algorithm)
        for idx in range(N):
            self._lazy_expand() # expands G_lazy
            self._lazy_update() # update G with cost-improving candidates from G_lazy
        return self._construct_plan(self.G, start_state, goal_state)

