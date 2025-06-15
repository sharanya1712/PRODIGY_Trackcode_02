import numpy as np
import random
data = [
    [15, 39], [15, 81], [16, 6], [16, 77], [17, 40], [17, 76],
    [18, 6], [18, 94], [19, 3], [19, 72], [19, 14], [19, 99],
    [20, 15], [20, 77], [20, 13], [20, 79], [21, 35], [21, 66],
    [23, 29], [23, 98], [24, 35], [24, 73], [25, 5], [25, 73]
]  
X = np.array(data)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_scaled = (X - mean) / std
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def initialize_centroids(X, k):
    indices = random.sample(range(len(X)), k)
    return X[indices]

def assign_clusters(X, centroids):
    labels = []
    for point in X:
        distances = [euclidean_distance(point, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)

def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            
            new_centroids.append(X[random.randint(0, len(X)-1)])
    return np.array(new_centroids)

def compute_inertia(X, centroids, labels):
    inertia = 0
    for i, point in enumerate(X):
        inertia += euclidean_distance(point, centroids[labels[i]]) ** 2
    return inertia
inertia_values = []
for k in range(1, 11):
    centroids = initialize_centroids(X_scaled, k)
    for _ in range(100):  # max iterations
        labels = assign_clusters(X_scaled, centroids)
        new_centroids = update_centroids(X_scaled, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    inertia = compute_inertia(X_scaled, centroids, labels)
    inertia_values.append(inertia)
    print(f"k={k}, Inertia={inertia:.2f}")

k = 5
centroids = initialize_centroids(X_scaled, k)
for _ in range(100):
    labels = assign_clusters(X_scaled, centroids)
    new_centroids = update_centroids(X_scaled, labels, k)
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids
for i, (income, score) in enumerate(X):
    print(f"Customer {i+1}: Income = {income}, Score = {score}, Cluster = {labels[i]}")

