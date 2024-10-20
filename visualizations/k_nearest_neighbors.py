import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from knn import k_nearest_neighbors

def initialize_plot(X, K):
    """Set up the plot and initialize plot components."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'{K}-Nearest Neighbors Animation')
    
    # Initialize colors with the same length as X
    initial_colors = np.array(['gray'] * X.shape[0])
    
    # Create scatter plot with individual colors for each point
    scatter = ax.scatter(X[:, 0], X[:, 1], c=initial_colors, marker='o', edgecolor='none', s=100, alpha=0.6)
    
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    return fig, ax, scatter, text

def update_plot(ax, scatter, text, idx, majority_class_arr, neighbors_indices, final_colors, gray_rgba):
    """Update the plot for each frame of the animation."""
    colors = scatter.get_facecolors()
    
    for neighbor_index in neighbors_indices[idx]:
        ax.plot([X[idx, 0], X[neighbor_index, 0]], 
                [X[idx, 1], X[neighbor_index, 1]], 
                'r--', linewidth=1)
        if not final_colors[neighbor_index]:
            colors[neighbor_index] = plt.cm.viridis(y[neighbor_index] / (max(y) + 1))

    if idx < colors.shape[0] and not final_colors[idx]:
        colors[idx] = plt.cm.viridis(majority_class_arr[idx] / (max(y) + 1))
        final_colors[idx] = True

    for neighbor_index in neighbors_indices[idx]:
        if neighbor_index < colors.shape[0] and not final_colors[neighbor_index]:
            colors[neighbor_index] = gray_rgba

    scatter.set_facecolors(colors)
    text.set_text(f'Point {idx}: Majority class = {majority_class_arr[idx]}')

def animate(frame):
    """Animation update function."""
    update_plot(ax, scatter, text, frame, majority_class_arr, neighbors_indices, final_colors, gray_rgba)
    return scatter, text

if __name__ == "__main__":
    K = 3
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
    
    majority_class_arr, neighbors_indices = zip(*[k_nearest_neighbors(X, y, i, K) for i in range(X.shape[0])])
    majority_class_arr, neighbors_indices = list(majority_class_arr), list(neighbors_indices)

    fig, ax, scatter, text = initialize_plot(X, K)
    gray_rgba = plt.cm.gray(0.5)
    final_colors = np.zeros(X.shape[0], dtype=bool)

    ani = FuncAnimation(fig, animate, frames=range(X.shape[0]), blit=False, interval=300, repeat=False)
    plt.show()