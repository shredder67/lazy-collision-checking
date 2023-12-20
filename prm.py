import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def sampler(x_range, y_range):
    return (random.uniform(*x_range), random.uniform(*y_range))

# Define the collision checker function
def line_intersects_circle(p1, p2, center, radius):
    p1, p2, center = np.array(p1), np.array(p2), np.array(center)
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return False
    else:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def collision_checker(u, v, obstacles):
    for obstacle in obstacles:
        if line_intersects_circle(u, v, obstacle['center'], obstacle['radius']):
            return False
    return True

def euclidean_distance(u, v):
    return np.linalg.norm(np.array(u) - np.array(v))

class PRMStar:
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

    def build_roadmap(self, start, goal):
        points = [self.sampler(self.x_range, self.y_range) for _ in range(self.num_samples)] + [start, goal]
        tree = KDTree(points)
        for point in points:
            distances, indices = tree.query(point, self.k + 1)
            for dist, idx in zip(distances, indices):
                if dist > 0 and self.collision_checker(point, points[idx], self.obstacles):
                    self.G.add_edge(point, points[idx], weight=dist)

    def find_path(self, start, goal):
        if self.collision_checker(start, goal, self.obstacles):
            self.G.add_edge(start, goal, weight=self.distance_fn(start, goal))
        path = nx.shortest_path(self.G, source=start, target=goal, weight='weight')
        return path


#Implementation
# Initialize PRM* with obstacles
obstacles = [{'center': (50, 50), 'radius': 20}, {'center': (30, 30), 'radius': 5}]
x_range = (0, 100)
y_range = (0, 100)
num_samples = 500
k = 15
prm_star = PRMStar(
    lambda x_range, y_range: sampler(x_range, y_range), 
    lambda u, v, obstacles: collision_checker(u, v, obstacles), 
    euclidean_distance, 
    x_range, 
    y_range, 
    num_samples, 
    k, 
    obstacles
)
start = (10, 10)
goal = (90, 90)
# Build the roadmap
prm_star.build_roadmap(start, goal)

# Define start and goal


# Find path
path = prm_star.find_path(start, goal)

# Visualization
plt.figure(figsize=(8, 8))
plt.scatter(*zip(*prm_star.G.nodes), s=10, label='Sampled Points')
if path:
    plt.plot(*zip(*path), c='r', lw=2, label='Path')
plt.scatter(*start, c='green', s=50, label='Start')
plt.scatter(*goal, c='blue', s=50, label='Goal')

# Draw obstacles
for obstacle in obstacles:
    circle = plt.Circle(obstacle['center'], obstacle['radius'], color='gray', alpha=0.5)
    plt.gca().add_patch(circle)

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PRM* Path Planning with Obstacles')
plt.xlim(*x_range)
plt.ylim(*y_range)
plt.show()