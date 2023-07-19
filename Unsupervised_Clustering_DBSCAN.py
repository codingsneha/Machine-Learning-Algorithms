import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def dbscan(X, eps, min_samples):
    # Step 1: Compute the neighborhood of each data point
    neighbors = compute_neighbors(X, eps)

    # Step 2: Assign core and noise points
    labels = np.zeros(len(X), dtype=int)
    cluster_id = 0
    for i in range(len(X)):
        if labels[i] != 0:
            continue

        if len(neighbors[i]) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_id += 1
            labels[i] = cluster_id
            expand_cluster(X, neighbors, labels, i, cluster_id, eps, min_samples)

    return labels


def compute_neighbors(X, eps):
    nbrs = NearestNeighbors(n_neighbors=len(X), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    neighbors = [indices[i, distances[i] <= eps].tolist() for i in range(len(X))]
    return neighbors


def expand_cluster(X, neighbors, labels, point_idx, cluster_id, eps, min_samples):
    cluster_points = [point_idx]
    i = 0
    while i < len(cluster_points):
        point = cluster_points[i]
        neighbors_point = neighbors[point]
        if len(neighbors_point) >= min_samples:
            for neighbor in neighbors_point:
                if labels[neighbor] == 0:
                    cluster_points.append(neighbor)
                    labels[neighbor] = cluster_id
                elif labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
        i += 1


# Example usage
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
eps = 2.0
min_samples = 2
labels = dbscan(X, eps, min_samples)

# Visualization
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black color for noise points

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('DBSCAN Clustering')
plt.show()
