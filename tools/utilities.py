import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def plot_classification(y, y_pred, custom_title=""):
    mid_point = (min(y) + max(y)) // 2
    plt.figure(figsize=(4, 3))
    plt.scatter(range(len(y)), y, color="red", marker="o", label="True Labels (y)")
    plt.scatter(
        range(len(y_pred)),
        y_pred,
        color="blue",
        marker="x",
        label="Predicted Values (y_pred)",
    )
    plt.axhline(
        mid_point,
        color="gray",
        linestyle="--",
        label=f"Decision Boundary ({mid_point})",
    )
    plt.title(f"True Labels vs Predictions {custom_title}")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_regression(y, y_pred, custom_title=""):
    plt.figure(figsize=(4, 3))
    plt.scatter(range(len(y)), y, color="red", marker="o", label="True Labels (y)")
    plt.scatter(
        range(len(y_pred)),
        y_pred,
        color="blue",
        marker="x",
        label="Predicted Values (y_pred)",
    )
    plt.title(f"True Labels vs Predictions {custom_title}")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_losses(losses, custom_title=""):
    plt.figure(figsize=(4, 3))
    plt.plot(losses, marker="o", linestyle="-", color="b")
    plt.title(f"Loss Values {custom_title}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def create_knn_animation(X, y, K, majority_class_arr, neighbors_indices, interval=300):
    """
    Create and display KNN classification animation.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, 2)
    y : numpy.ndarray
        Target labels
    K : int
        Number of neighbors
    interval : int, optional
        Animation interval in milliseconds (default: 300)
        
    Returns:
    --------
    animation : matplotlib.animation.FuncAnimation
        The animation object
    """
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{K}-Nearest Neighbors Animation")

    # Initialize scatter plot
    initial_colors = np.array(["gray"] * X.shape[0])
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=initial_colors,
        marker="o",
        edgecolor="none",
        s=100,
        alpha=0.6,
    )

    # Initialize text
    text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top"
    )
    
    # Initialize animation variables
    gray_rgba = plt.cm.gray(0.5)
    final_colors = np.zeros(X.shape[0], dtype=bool)
    
    def update(frame):
        """Frame update function for animation"""
        colors = scatter.get_facecolors()
        
        # Draw connections to neighbors
        for neighbor_index in neighbors_indices[frame]:
            ax.plot(
                [X[frame, 0], X[neighbor_index, 0]],
                [X[frame, 1], X[neighbor_index, 1]],
                "r--",
                linewidth=1,
            )
            if not final_colors[neighbor_index]:
                colors[neighbor_index] = plt.cm.viridis(y[neighbor_index] / (max(y) + 1))
        
        # Update current point
        if frame < colors.shape[0] and not final_colors[frame]:
            colors[frame] = plt.cm.viridis(majority_class_arr[frame] / (max(y) + 1))
            final_colors[frame] = True
        
        # Update neighbor colors
        for neighbor_index in neighbors_indices[frame]:
            if neighbor_index < colors.shape[0] and not final_colors[neighbor_index]:
                colors[neighbor_index] = gray_rgba
        
        scatter.set_facecolors(colors)
        text.set_text(f"Point {frame}: Majority class = {majority_class_arr[frame]}")
        return scatter, text
    
    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=range(X.shape[0]),
        blit=False,
        interval=interval,
        repeat=False
    )
    
    return ani