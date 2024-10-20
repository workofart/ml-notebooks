from collections import Counter
import numpy as np

def k_nearest_neighbors(X: np.array, y: np.array, idx: int, K: int):
    """
    Computes the k-nearest neighbors of a given point (specified by the index) in a dataset.
    Every time this function gets called, we will traverse all the points in the entire dataset.

    Args:
        X (np.array): The full array of input points
        y (np.array): The label of each of the points (which groundtruth class it belongs to)
        idx (int): The index of the point we're currently interested in finding the neighbors for
        K (int): The number of neighbors to find

    Returns:
        tuple: majority_class(list), neighbors_indices(list)
    """
    # Compute the Euclidean distance from point X[frame] to all other points
    distances = np.sqrt(np.sum((X[idx] - X) ** 2, axis=1))
    
    # Sort the distances and get the indices of the sorted distances
    indices = np.argsort(distances)
    
    # Get the top K neighbors (excluding the point itself)
    neighbors_indices = indices[1:K+1]
    
    # Determine the majority class among the neighbors
    neighbor_labels = y[neighbors_indices]
    majority_class = Counter(neighbor_labels).most_common(1)[0][0]
    
    return majority_class, neighbors_indices
