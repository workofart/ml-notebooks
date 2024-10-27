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
        X[:, 0],
        X[:, 1],
        c=initial_colors,
        marker="o",
        edgecolor="none",
        s=100,
        alpha=0.6,
    )

    # Initialize text
    text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
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
                colors[neighbor_index] = plt.cm.viridis(
                    y[neighbor_index] / (max(y) + 1)
                )

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
        repeat=False,
    )

    return ani


def create_bayes_animation(N, parameter_sets, all_p_cancer_values, all_parameters):
    """
    Create an animation of diagnosis probabilities over time.

    Parameters:
    -----------
    N : int
        Number of diagnoses per parameter set
    parameter_sets : list
        List of tuples containing (p_cancer_initial, accuracy_range, false_positive_range, title)
    all_p_cancer_values : list
        List of lists containing p_cancer values for each parameter set
    all_parameters : list
        List of lists containing diagnosis parameters for each parameter set

    Returns:
    --------
    animation : matplotlib.animation.FuncAnimation
        The animation object
    """

    # Set up the figure and animation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Diagnosis Number")
    ax.set_ylabel("Probability of Cancer")
    ax.set_title("Evolution of p(cancer) Over Diagnoses with Varying Parameters")

    # Initialize plot elements
    (line,) = ax.plot([], [], lw=2)
    text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment="top"
    )

    def update(frame):
        """Update function for animation that handles both initialization and frame updates"""
        # Initialize if frame is None
        if frame is None:
            line.set_data([], [])
            text.set_text("")
            return line, text

        # Regular frame update
        param_set_index, diagnosis_index = divmod(frame, N)

        # Get current data
        p_cancer_values = all_p_cancer_values[param_set_index]
        parameters = all_parameters[param_set_index]
        p_cancer_initial, accuracy_range, false_positive_range, title = parameter_sets[
            param_set_index
        ]

        # Update line data
        x = np.arange(diagnosis_index + 1)
        y = p_cancer_values[: diagnosis_index + 1]
        line.set_data(x, y)

        # Update text
        (
            p_test_positive_given_cancer,
            p_test_positive_given_no_cancer,
            p_cancer,
            diagnosis,
        ) = parameters[diagnosis_index]
        text.set_text(
            f"Scenario: {title}:\n"
            f"  P_CANCER_INITIAL = {p_cancer_initial}\n"
            f"  ACCURACY_RANGE = {accuracy_range}\n"
            f"  FALSE_POSITIVE_RANGE = {false_positive_range}\n\n"
            f"Diagnosis {diagnosis_index + 1}:\n"
            f"  p(test positive | cancer) = {p_test_positive_given_cancer}\n"
            f"  p(test positive | no cancer) = {p_test_positive_given_no_cancer}\n"
            f"  Diagnosis result: {diagnosis}\n"
            f"  Updated p(cancer) = {p_cancer:.4f}"
        )

        return line, text

    # Create animation
    animation = FuncAnimation(
        fig=fig,
        func=update,
        frames=N * len(parameter_sets),
        init_func=lambda: update(None),
        interval=150,
        blit=True,
    )

    return animation
