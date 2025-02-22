import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def compare_gradient_weight_update_schedule(mini_batch_size=4, accum_batch_size=4, total_samples=12):
    def plot_grad_and_update(ax, method_name, grad_steps, update_steps):
        """
        Plots a timeline showing sample indices (1..total_samples),
        highlighting:
        - where we compute gradients (green)
        - where we perform parameter updates (red)
        """
        # Draw a horizontal timeline
        ax.hlines(y=0, xmin=1, xmax=total_samples, color='gray', linewidth=2)
        
        # Mark all sample positions in black
        for i in range(1, total_samples + 1):
            ax.plot(i, 0, 'o', color='black', markersize=5)
        
        # Mark gradient-computation points in green
        for g in grad_steps:
            ax.plot(g, 0, '^', color='green', markersize=8)  # '^' is a triangle marker
            ax.text(g, -0.15, f"Grad@{g}", ha='center', color='green', fontsize=6)
        
        # Mark update points in red
        for u in update_steps:
            ax.plot(u, 0, 'o', color='red', markersize=10)
            ax.text(u, 0.15, f"Update@{u}", ha='center', color='red', fontsize=6)
        
        ax.set_xlim(0.5, total_samples + 0.5)
        ax.set_ylim(-0.5, 1.0)
        ax.set_yticks([])
        ax.set_title(method_name, fontsize=10)
        ax.set_xlabel("Sample Index")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # ----------------------------------------------------
    # Example parameters
    N = 12

    # FULL-BATCH:
    # - Grad compute after all N=12 samples
    # - Single update at sample 12
    full_batch_grads = [N]
    full_batch_updates = [N]

    # MINI-BATCH (size=4):
    # - Grad compute after each batch: 4, 8, 12
    # - Update immediately at the same step
    mini_batch_grads = list(range(mini_batch_size, N+1, mini_batch_size))
    mini_batch_updates = mini_batch_grads  # same as grad steps

    # STOCHASTIC (size=1):
    # - Grad compute at every sample: 1..12
    # - Update at every sample: 1..12
    stochastic_grads = list(range(1, N+1))
    stochastic_updates = stochastic_grads

    # GRADIENT ACCUMULATION:
    # - Suppose each mini-batch is size 3 => 4 mini-batches total (ends at sample 3,6,9,12)
    # - We accumulate for 2 mini-batches => effectively update at sample 6 and 12
    accum_grads = list(range(accum_batch_size, N+1, accum_batch_size))  # [3, 6, 9, 12]
    accum_updates = [6, 12]  # after every 2 mini-batches

    # ----------------------------------------------------
    # PLOT
    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

    plot_grad_and_update(
        axes[0],
        "(Full) Batch Gradient Descent (Size=12)",
        grad_steps=full_batch_grads,
        update_steps=full_batch_updates,
    )
    plot_grad_and_update(
        axes[1],
        "Mini-Batch Gradient Descent (Size=4)",
        grad_steps=mini_batch_grads,
        update_steps=mini_batch_updates,
    )
    plot_grad_and_update(
        axes[2],
        "Stochastic Gradient Descent (Size=1)",
        grad_steps=stochastic_grads,
        update_steps=stochastic_updates,
    )
    plot_grad_and_update(
        axes[3],
        "Gradient Accumulation (Batch=3, Accum=2 -> Update@6,12)",
        grad_steps=accum_grads,
        update_steps=accum_updates,
    )

    plt.tight_layout()
    plt.show()
    


# This is just a simple loss function that can illustrate the differences of various GD methods
def mse_loss_and_grad(w, X, y):
    """Returns (loss, grad) for MSE = mean((w*x - y)^2)."""
    preds = w * X
    residuals = preds - y
    loss = np.mean(residuals ** 2)
    grad = 2.0 / len(X) * np.sum(residuals * X)
    return loss, grad

def run_gradient_descent(
    X, y, 
    method="full-batch", 
    batch_size=1, 
    lr=0.01, 
    epochs=5, 
    accum_steps=1
):
    """
    method: 'full-batch', 'mini-batch' or 'accum'
    batch_size: used for mini-batch or accum (size of each mini-batch)
    lr: learning rate
    epochs: number of passes through the dataset
    accum_steps: number of mini-batches to accumulate before updating (used if method='accum')
    """
    w = 0.0  # initialize parameter
    N = len(X)

    w_history = []
    loss_history = []

    for epoch in range(epochs):
        # Shuffle each epoch (we do the shuffling for all methods)
        indices = np.arange(N)
        np.random.shuffle(indices)

        if method == "full-batch":
            # One update per epoch over the entire dataset
            loss, grad = mse_loss_and_grad(w, X, y)
            w = w - lr * grad
            w_history.append(w)
            loss_history.append(loss)

        elif method in ["mini-batch"]:

            # Go through data in mini-batches
            for start_idx in range(0, N, batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                loss, grad = mse_loss_and_grad(w, X_batch, y_batch)
                w = w - lr * grad

                w_history.append(w)
                loss_history.append(loss)

        elif method == "accum":
            # Gradient accumulation:
            #  - break data into mini-batches of size "batch_size"
            #  - accumulate over "accum_steps" mini-batches, then update once
            grad_acc = 0.0
            accum_count = 0
            for start_idx in range(0, N, batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                loss, grad = mse_loss_and_grad(w, X_batch, y_batch)
                grad_acc += grad
                accum_count += 1

                # Once we've accumulated enough mini-batches, do a single update
                if accum_count == accum_steps:
                    grad_acc /= accum_steps
                    w = w - lr * grad_acc
                    w_history.append(w)
                    loss_history.append(loss)
                    grad_acc = 0.0
                    accum_count = 0

            # If leftover mini-batches remain
            if accum_count > 0:
                grad_acc /= accum_count
                w = w - lr * grad_acc
                w_history.append(w)
                loss_history.append(loss)
                grad_acc = 0.0
                accum_count = 0

    return np.array(w_history), np.array(loss_history)


def compare_loss_parameter_trajectory(epochs=20, lr=0.1, mini_batch_size=16, grad_accumulate_size=4, grad_accumulate_every_n_steps=4):
    N = 1000  # number of data points
    X = np.random.randn(N)
    true_w = 3.0
    y = true_w * X + 0.5 * np.random.randn(N)  # y = 3*x + noise
    
    w_full, loss_full = run_gradient_descent(X, y, method="full-batch", lr=lr, epochs=epochs)
    w_mini, loss_mini = run_gradient_descent(X, y, method="mini-batch", batch_size=mini_batch_size, lr=lr, epochs=epochs)
    w_accum, loss_accum = run_gradient_descent(X, y, method="accum", batch_size=grad_accumulate_size, accum_steps=grad_accumulate_every_n_steps, lr=lr, epochs=epochs)

    def plot_normalized_methods(w_dict, loss_dict, method_list):
        """
        Each method may have a different number of updates.
        We'll map each method's update steps to [0,1] for plotting,
        so that the lines don't overlap or "disappear."
        """
        plt.figure(figsize=(10, 4))

        # Subplot (1,2,1): Parameter Trajectories
        plt.subplot(1, 2, 1)
        for method_name in method_list:
            w_vals = w_dict[method_name]
            # Normalized x from 0 to 1
            x_vals = np.linspace(0, 1, len(w_vals))
            plt.plot(x_vals, w_vals, label=method_name, marker='o', markersize=3)
        plt.axhline(true_w, color='gray', linestyle='--', label=f"True w = {true_w}")
        plt.xlabel("Normalized Update Step (0=begin, 1=end)")
        plt.ylabel("Parameter w")
        plt.title("Parameter Trajectory")
        plt.legend()

        # Subplot (1,2,2): Loss vs. Update Step
        plt.subplot(1, 2, 2)
        for method_name in method_list:
            loss_vals = loss_dict[method_name]
            x_vals = np.linspace(0, 1, len(loss_vals))
            plt.plot(x_vals, loss_vals, label=method_name, marker='.', markersize=1)
        plt.yscale('log')
        plt.xlabel("Normalized Update Step")
        plt.ylabel("Loss (MSE, log scale)")
        plt.title("Loss vs. Update Step")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Organize results
    w_dict = {
        "Full-Batch": w_full,
        "Mini-Batch (B=16)": w_mini,
        "Accum (B=4x4)": w_accum
    }
    loss_dict = {
        "Full-Batch": loss_full,
        "Mini-Batch (B=16)": loss_mini,
        "Accum (B=4x4)": loss_accum
    }

    method_list = list(w_dict.keys())

    plot_normalized_methods(w_dict, loss_dict, method_list)