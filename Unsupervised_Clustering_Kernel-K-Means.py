import numpy as np
import matplotlib.pyplot as plt


def kernel_kmeans(X, kernel_matrix, num_clusters, max_iterations=100):
    n_samples = X.shape[0]
    labels = np.zeros(n_samples)
    centroids = X[np.random.choice(n_samples, num_clusters, replace=False)]

    for _ in range(max_iterations):
        # Step 1: Assign samples to the nearest centroid
        distance_matrix = compute_distance_matrix(kernel_matrix, centroids)
        new_labels = np.argmin(distance_matrix, axis=1)

        # Step 2: Update centroids
        for i in range(num_clusters):
            cluster_points = X[new_labels == i]
            centroids[i] = compute_kernel_mean(cluster_points, kernel_matrix)

        # Check convergence
        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

    return labels


def compute_distance_matrix(kernel_matrix, centroids):
    n_samples = kernel_matrix.shape[0]
    n_centroids = centroids.shape[0]
    distance_matrix = np.zeros((n_samples, n_centroids))
    for i in range(n_centroids):
        centroid = centroids[i]
        centroid_reshaped = centroid.reshape((1, -1))
        distance_matrix[:, i] = np.diagonal(kernel_matrix) - 2 * np.sum(kernel_matrix * centroid_reshaped, axis=1) \
                                 + np.sum(centroid_reshaped * kernel_matrix.dot(centroid_reshaped.T), axis=1)
    return distance_matrix


def compute_kernel_mean(cluster_points, kernel_matrix):
    weights = np.sum(kernel_matrix[cluster_points], axis=0)
    normalized_weights = weights / np.sum(weights)
    return np.sum(normalized_weights[:, np.newaxis] * cluster_points, axis=0)


# Example usage
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
num_clusters = 2

# Compute the kernel matrix
kernel_matrix = np.exp(-np.linalg.norm(X[:, np.newaxis] - X, axis=2) ** 2)

# Perform Kernel K-means clustering
labels = kernel_kmeans(X, kernel_matrix, num_clusters)

# Visualization
colors = ['r', 'g', 'b', 'c', 'm', 'y']

plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    plt.scatter(X[i, 0], X[i, 1], color=colors[label])

plt.title('Kernel K-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
