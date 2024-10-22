import numpy as np


def k_means(points, K, max_iters=20, tol=1e-4):
    # initialize the cluster assignments
    cluster_centers = np.random.randint(0, 10, size=K)
    cluster_ind = np.zeros(
        N_points, dtype=int
    )  # 0 means the ith point belongs to the cluster 0

    for iter in range(max_iters):
        # Expectation-step: Assign points to the nearest cluster center
        for i in range(N_points):
            # compute the distance for every point to every K cluster
            distances = np.array(
                [np.linalg.norm(points[i] - cluster_centers[k]) for k in range(K)]
            )
            cluster_ind[i] = np.argmin(distances)

        # Maximization-step: Update cluster centers
        new_cluster_centers = np.zeros(K)
        for k in range(K):
            cluster_points = points[cluster_ind == k]
            # If there are some points assigned to the cluster, then we compute the mean of the cluster
            if len(cluster_points) > 0:
                new_cluster_centers[k] = np.mean(cluster_points)
            # If a cluster has no points assigned, we randomly reinitialize the centers
            else:
                new_cluster_centers[k] = np.random.randint(0, 10)

        loss = sum((new_cluster_centers - cluster_centers) ** 2) ** 0.5
        if iter % 10 == 0:
            print(f"[{iter}] Loss: {loss}")
        if loss < tol:
            print(f"Converged after {iter} iterations.")
            break

        cluster_centers[:] = new_cluster_centers

    return cluster_ind, cluster_centers


if __name__ == "__main__":
    K = 10  # number of clusters
    N_points = 100

    cluster_ind = np.zeros(N_points, dtype=int)
    cluster_centers = np.random.randint(0, 10, size=K)
    points = np.random.randint(0, 10, size=N_points)

    # Compute a sample loss
    print(
        "Initial sum of squared distances:",
        sum((points[i] - cluster_centers[cluster_ind[i]]) ** 2 for i in range(N_points))
        ** 0.5,
    )

    cluster_ind, cluster_centers = k_means(points, K, max_iters=100)

    sum_squared_distances = (
        sum((points[i] - cluster_centers[cluster_ind[i]]) ** 2 for i in range(N_points))
        ** 0.5
    )
    print("Final sum of squared distances:", sum_squared_distances)

    # We expect the final loss to be zero
    assert sum_squared_distances == 0

    # We verify that the cluster assignments are correct
    for i in range(N_points):
        for k in range(K):
            if points[i] != cluster_centers[cluster_ind[i]]:
                print(f"Point {points[i]} is not assigned to cluster {cluster_ind[i]}")
                raise Exception(
                    "Cluster assignments are incorrect, please double-check your code."
                )
            else:
                if i % 100 == 0:
                    print(
                        f"Point {points[i]} is correctly assigned to cluster {cluster_ind[i]} with center mean as {cluster_centers[cluster_ind[i]]}"
                    )
