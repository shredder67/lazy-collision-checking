import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environment import State, distance 

def compute_path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        cost += distance(path[i], path[i + 1])
    return cost

def visualize_plan_execution(env, plan, start_st, goal_st, saveas='visalization.mp4'):
    raise NotImplementedError()