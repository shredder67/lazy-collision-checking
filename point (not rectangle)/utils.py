import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def sampler(x_range, y_range):
    return (random.uniform(*x_range), random.uniform(*y_range))

def collision_checker(u, v, obstacles):
    for obstacle in obstacles:
        if line_intersects_circle(u, v, obstacle['center'], obstacle['radius']):
            return False
    return True
    
def line_intersects_circle(p1, p2, center, radius):
    p1, p2, center = np.array(p1), np.array(p2), np.array(center)
    d = p2 - p1
    f = p1 - center
    a = d.dot(d)
    b = 2*f.dot(d)
    c = f.dot(f) - radius**2
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True
    return False

def edge_collision_checker(p1, p2, obstacles):
    for obstacle in obstacles:
        if line_intersects_circle(p1, p2, obstacle['center'], obstacle['radius']):
            return True
    return False

def euclidean_distance(u, v):
    return np.linalg.norm(np.array(u) - np.array(v))

def plot(lazy_prm_star, path, start, goal, obstacles, x_range, y_range):
    plt.figure(figsize=(8, 8))
    plt.scatter(*zip(*lazy_prm_star.G.nodes), s=10, label='Sampled Points')
    if path:
        plt.plot(*zip(*path), c='r', lw=2, label='Path')
    plt.scatter(*start, c='green', s=50, label='Start')
    plt.scatter(*goal, c='blue', s=50, label='Goal')

    for obstacle in obstacles:
        circle = plt.Circle(obstacle['center'], obstacle['radius'], color='gray', alpha=0.5)
        plt.gca().add_patch(circle)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Lazy PRM* Path Planning with Obstacles')
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.show()