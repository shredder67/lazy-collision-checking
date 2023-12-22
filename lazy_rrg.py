from typing import Callable, List
from queue import PriorityQueue
from collections import defaultdict

import numpy as np
#from scipy.spatial import KDTree

from environment import State, Environment


def construct_plan_from_a_star(parent_table, goal_state, start_state):
        plan = []
        cur = goal_state
        while cur != start_state:
            plan.append(cur)
            cur = parent_table[cur][0]
            plan.append(cur) # starting position
        return plan[::-1]


def a_star(G, start_state, goal_state):
    p_queue = PriorityQueue()
    visited = defaultdict(int) # 1 - visited, 0 - not visited
    visited[start_state] = 1
    parent_table = {start_state: (start_state, 0)} # state: (parent_state, C(state))
    p_queue.put((0, start_state))
    while p_queue.qsize() != 0:
        cur_state = p_queue.get()[1]
        if cur_state == goal_state:
            return 'success', (
                self._construct_plan_from_a_star(parent_table, goal_state, start_state),
                sum(visited.values()),
                parent_table[cur_state][1]
            )

        for st_next in G[cur_state]:
            a_cost = self._distance_fn(cur_state, st_next)
            next_st_cost = a_cost + parent_table[cur_state][1] # l(x, u) + C(x)
            if visited[st_next] == 0:
                visited[st_next] = 1
                parent_table[st_next] = (cur_state, next_st_cost)
                p_queue.put((next_st_cost, st_next))
            else:
                if parent_table[st_next][1] > next_st_cost:
                    parent_table[st_next] = (cur_state, next_st_cost)

    return "failure", (None, sum(visited.values()), -1)


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
        self._max_move_step = max_move_step

    def plan(self,
             start_state: State,
             goal_state: State) -> List[State]:
        raise NotImplementedError


class LazyRRGPlanner(Planner):
    def __init__(self, env: Environment,
                 distance_fn: Callable,
                 env_sampler: Callable,
                 k: int,
                 max_angle_step: float=10.0,
                 max_move_step: float=3.0,
        ):
        super().__init__(env, distance_fn, max_angle_step, max_move_step)
        self.env_sampler = env_sampler
        self.G = {}
        self.G_lazy = {}
        self.costs = {}
        self.predecessors = {}

        self.k=k # Neighbors(R,q,i) under Algorithm 2
        self.cur_iter = 0
        
    def _get_config_points_from_G(self,  G):
        states = list(G.keys())
        return [st.to_list() for st in states]

    def _construct_plan(self, start_state, goal_state):
        path = self._path_to(goal_state)
        #assert path[0] == start_state and path[-1] == goal_state
        return path

    def _find_nearest(self, state, G):
        mindist, argmin = np.inf, None
        for st_cand in G.keys():
            d = self._distance_fn(state, st_cand) 
            if d < mindist:
                mindist = d
                argmin = st_cand
        return argmin
    
    def _find_k_nearest(self, state, G, k):
        min_cand = PriorityQueue()
        for st_cand in G.keys():
            d = self._distance_fn(state, st_cand) 
            min_cand.put((d, st_cand))
            if min_cand.qsize() > self.k:
                min_cand.get()
        return [min_cand.get()[1] for _ in range(min_cand.qsize())]

    def _lazy_expand(self, q_goal):
        q_rand = self.env_sampler()

        # small chance to sample q_goal
        if np.random.uniform() < 0.1:
            q_rand = q_goal

        q_near = self._find_nearest(q_rand, self.G_lazy)
        q_delta = q_rand - q_near # think about this as vector
        
        # ensure steering constraints
        q_delta._center_coors = np.minimum(self._max_move_step, q_delta._center_coors)
        q_delta._angle = np.minimum(self._max_angle_step, q_delta._angle)
        
        #d = 3 # config space dimension
        #r_i = (np.log(self.cur_iter) / self.cur_iter)**(1/d)
        #q_delta = q_delta * max(r_i/q_delta.norm(), 1)

        q = q_near + q_delta
        if self._env.check_collision(q): # if state is collision free
            self.G_lazy[q_near].append(q)
            self.G_lazy[q] = [q_near]

            self.costs[q] = np.inf
            self.predecessors[q] = [None, None]

            # k-nearest strategy
            v_neighbor = self._find_k_nearest(q, self.G, self.k)
            for v in v_neighbor:
                if v != q_near:
                    self.G_lazy[q].append(v)
                    self.G_lazy[v].append(q)

                # cost and predecessor update
                if self.costs[v] + self._distance_fn(v, q) < self.costs[q]:
                    self.costs[q] = self.costs[v] + self._distance_fn(v, q)
                    self.predecessors[q][0] = v
                    # propogate cost update FORWARD from v to q (along path)
                    nxt = self.predecessors[q][1]
                    while nxt is not None:
                        cur = self.predecessors[nxt][0]
                        if self.costs[nxt] > self.costs[cur] + self._distance_fn(cur, nxt):
                            self.costs[nxt] = self.costs[cur] + self._distance_fn(cur, nxt)
                            nxt = self.predecessors[nxt][1]
                        else:
                            break

                if self.costs[q] + self._distance_fn(q, v) < self.costs[v]:
                    self.costs[v] = self.costs[q] + self._distance_fn(q, v)
                    self.predecessors[v][1] = q
                    # propagate cost update BACKWARD from q to v (along path)
                    prev = self.predecessors[v][0]
                    while prev is not None:
                        cur = self.predecessors[prev][1]
                        if self.costs[prev] > self.costs[cur] + self._distance_fn(prev, cur):
                            self.costs[prev] = self.costs[q] + self._distance_fn(prev, cur)
                            prev = self.predecessors[prev][0]
                        else:
                            break


    def _path_to(self, state):
        """backtrack path through self.predecessors from q_start to state"""
        path = [state]
        while self.predecessors[path[-1]][0] is not None:
            path.append(self.predecessors[path[-1]][0])
        return path[::-1]
    
    def _is_visible(self, u, v):
        inter_states = State.generate_lin_space(u, v, 10)
        for st in inter_states:
            if self._env.check_collision(st):
                return False
        return True

    def _lazy_update(self, c_best, goal_state):
        all_edges_in_G = True
        while True:
            q_g = goal_state
            if self.costs[q_g] < c_best:
                print(1)
                p = self._path_to(q_g)
                for (u, v) in zip(p[:-1], p[1:]):
                    if v in self.G[u]: continue
                    if self._is_visible(u, v):
                        self.G[u].append(v)
                        self.G[v].append(u)
                    else:
                        self.G_lazy[u].remove(v)
                        self.G_lazy[v].remove(u)

                        if self.predecessors[v][0] == u:
                            # search for least cost parent
                            min_cost = np.inf
                            min_parent = None
                            for parent in self.G_lazy[v]:
                                if self.costs[parent] + self._distance_fn(parent, v) < min_cost:
                                    min_cost = self.costs[parent] + self._distance_fn(parent, v)
                                    min_parent = parent
                            self.predecessors[u][1] = None 
                            self.predecessors[v][0] = min_parent
                            self.costs[v] = min_cost

                            # need to propogate cost update for all vetices
                            # involved into shortest path to q_g from v
                            nxt = self.predecessors[v][1]
                            while nxt is not None:
                                cur = self.predecessors[nxt][0]
                                self.costs[nxt] = self.costs[cur] + self._distance_fn(cur, nxt)
                                nxt = self.predecessors[nxt][1]
                    
                        all_edges_in_G = False
                        break
                if all_edges_in_G:
                    return self.costs[q_g]
            if q_g is None or c_best == self.costs[q_g]:
                return c_best

    def naive_update(self, c_best, start_state, goal_state):
        all_edges_in_G = True
        while True:
            status, (path, cost, _) = self._a_star(self.G_lazy, start_state, goal_state)
            if status == 'failure': break
            if cost < c_best:
                for (u, v) in zip(path[:-1], path[1:]):
                    if v in self.G[u]: continue
                    if v not in self.G[u]:
                        self.G[u].append(v)
                        self.G[v] = [u]
                    else:
                        self.G_lazy[u].remove(v)
                        self.G_lazy[v].remove(u)
                        all_edges_in_G = False
                        break
                if all_edges_in_G:
                    c_best = cost
                    break
        return c_best
        

    def plan(self, start_state, goal_state, N=1000, naive=True):
        self.G[start_state] = []
        self.G_lazy[start_state] = []
        self.G_lazy[goal_state] = []

        self.costs[start_state] = 0
        self.costs[goal_state] = np.inf
        self.predecessors[start_state] = [None, None] # store both previous and next state in path
        self.predecessors[goal_state] = [None, None]

        c_best = np.inf
        for idx in range(1, N):
            self.cur_iter = idx
            self._lazy_expand(goal_state) # expands G_lazy
            if naive:
                c_best = self.naive_update(c_best, start_state, goal_state)
            else:
                # update G with cost-improving candidates from G_lazy
                c_best = self._lazy_update(c_best, goal_state)
            c_best = self.naive_update(c_best, start_state, goal_state)
        return self._construct_plan(start_state, goal_state)

