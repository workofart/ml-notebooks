import matplotlib.pyplot as plt


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
