import numpy as np
from collections import defaultdict

import sys
sys.path.append(".")
from ex2_utils import generate_responses_1

def initialize_grid_cells(X, h):
    grid_cells = defaultdict(lambda: {"centroid": np.zeros(X.shape[1]), "count": 0, "points": []})
    for i, x in enumerate(X):
        index = tuple(np.floor(x / h).astype(int))
        grid_cells[index]["centroid"] += x
        grid_cells[index]["count"] += 1
        grid_cells[index]["points"].append(i)
    for cell in grid_cells.values():
        cell["centroid"] /= cell["count"]
    return grid_cells

def update_centroids(grid_cells):
    for index, cell in grid_cells.items():
        neighbors = [(index[0] + v[0], index[1] + v[1]) for v in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]]
        neighbor_centroids = [grid_cells[n]["centroid"] for n in neighbors if n in grid_cells]
        if neighbor_centroids:
            cell["centroid"] = np.mean(neighbor_centroids, axis=0)

def merge_cells(grid_cells):
    merged_cells = {}
    for index, cell in grid_cells.items():
        if tuple(cell["centroid"]) in merged_cells:
            merged_cells[tuple(cell["centroid"])]["points"] += cell["points"]
            merged_cells[tuple(cell["centroid"])]["centroid"] += cell["centroid"] * cell["count"]
            merged_cells[tuple(cell["centroid"])]["count"] += cell["count"]
        else:
            merged_cells[tuple(cell["centroid"])] = cell.copy()
    for cell in merged_cells.values():
        cell["centroid"] /= cell["count"]
    return merged_cells

def grid_shift(X, h, max_iterations=100):
    grid_cells = initialize_grid_cells(X, h)
    for _ in range(max_iterations):
        prev_grid_cells = grid_cells.copy()
        update_centroids(grid_cells)
        grid_cells = merge_cells(grid_cells)
        if prev_grid_cells == grid_cells:
            break
    return grid_cells

# Example usage:
# X = np.random.rand(100, 2)  # Sample data points
X = generate_responses_1()
h = 0.1  # Grid cell size
clusters = grid_shift(X, h)
for i, cluster in enumerate(clusters.values()):
    print(f"Cluster {i+1}: Centroid={cluster['centroid']}, Points={cluster['points']}")
