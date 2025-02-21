import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, Video
import ipywidgets as widgets

def nonconvex_func(x):
    """
    A function with multiple local minima and one global minimum at x=5.
    """
    return 0.01*(x-5)**2 + 0.3*np.sin(2*x) + 0.2*np.sin(4*x) - 0.2


def nonconvex_grad(x):
    """
    Gradient of the above nonconvex function.
    """
    return 0.02*(x-5) + 0.6*np.cos(2*x) + 0.8*np.cos(4*x)


def noisy_grad(x, noise_scale=0.2, **grad_kwargs):
    """
    Adds Gaussian noise to the gradient to simulate 'stochastic' updates.
    """
    return nonconvex_grad(x, **grad_kwargs) + noise_scale * np.random.randn()


def run_optimizer(
    update_fn,
    func,
    func_grad,
    x_init,
    lr,
    num_iters,
    init_state_fn=None,
    track_grad=False,
    track_lr_per_param=False,
):
    """
    Runs one optimizer (with a single learning rate) for a given number of iterations,
    and returns the resulting trajectory data:
      - x-values at each iteration
      - loss at each iteration
      - effective learning rate at each iteration
      - (optionally) gradient norms at each iteration

    Parameters
    ----------
    update_fn : callable
        The optimizer update function, e.g. sgd(x, grad, lr) or sgd_with_clip(x, grad, lr, state).
        Expected returns can be:
          - new_x
          - (new_x, effective_lr)
          - (new_x, new_state, effective_lr)
    func : callable
        The objective function f(x).
    func_grad : callable
        The gradient function g(x). Could be noisy or deterministic.
    x_init : float
        Initial value of x.
    lr : float
        Learning rate (base LR to pass into the update function).
    num_iters : int
        Number of iterations.
    init_state_fn : callable or None
        Function to create an initial state (dictionary or other structure). If None, no state is used.
    track_grad : bool
        If True, record the (absolute) gradient or "effective" gradient each iteration.
    track_lr_per_param : bool
        If True, record the effective learning rate for each parameter (as returned by update_fn).

    Returns
    -------
    dict with keys:
      "x_vals": 1D np.array of shape (num_iters,)
      "loss_vals": 1D np.array of shape (num_iters,)
      "lr_vals": 1D np.array of shape (num_iters,)
      "lr_vals_vec": list of arrays (one per iteration) or None
      "grad_vals": 1D np.array of shape (num_iters,) or None
    """

    if init_state_fn is None:
        init_state_fn = lambda: None

    state = init_state_fn()
    x = x_init

    # --- Pre-allocate arrays ---
    if np.ndim(x_init) == 0:
        x_vals = np.zeros(num_iters)
    else:
        x_vals = np.zeros((num_iters, len(x_init)))
    loss_vals = np.zeros(num_iters)
    lr_vals = np.zeros(num_iters)
    grad_vals = np.zeros(num_iters) if track_grad else None
    lr_vals_vec = [] if track_lr_per_param else None  # still a list if each iteration can have a variable-size array

    for i in range(num_iters):
        g = func_grad(x)

        # Call the optimizer's update function (with or without state).
        if state is not None:
            result = update_fn(x, g, lr, state)
        else:
            result = update_fn(x, g, lr)

        # Unpack
        if isinstance(result, tuple):
            if len(result) == 2:
                x_new, eff_lr = result
            else:
                x_new, state, eff_lr = result
        else:
            x_new, eff_lr = result, lr

        # Store
        x_vals[i] = x_new
        loss_vals[i] = np.mean(func(x_new))

        # If eff_lr is a vector, store its mean for a scalar summary.
        if isinstance(eff_lr, (list, np.ndarray)):
            lr_vals[i] = np.mean(eff_lr)
        else:
            lr_vals[i] = eff_lr

        if track_lr_per_param:
            lr_vals_vec.append(eff_lr)  # variable-length arrays can't easily be pre-allocated

        if track_grad:
            # If the state has "grad" explicitly, use that. Otherwise, use g.
            if state and isinstance(state, dict) and "grad" in state:
                grad_vals[i] = np.abs(state["grad"])
            else:
                grad_vals[i] = np.abs(g)

        x = x_new

    return {
        "x_vals": x_vals,
        "loss_vals": loss_vals,
        "lr_vals": lr_vals,
        "lr_vals_vec": lr_vals_vec,
        "grad_vals": grad_vals,
    }


def collect_trajectories(
    optimizers,
    num_iters,
    x_init,
    track_grad=False,
    track_lr_per_param=False,
):
    """
    Helper function that runs each optimizer in `optimizers` for all specified
    learning rates, and returns a dictionary keyed by (opt_name, lr_value).

    Parameters
    ----------
    optimizers : dict
        E.g. {
           "SGD": {
              "update_fn": <function>,
              "func": <function>,
              "func_grad": <function>,
              "init_state_fn": <function or None>,
              "lrs": [0.01, 0.1]
           },
           ...
        }
    num_iters : int
        Number of iterations to run.
    x_init : float or array
        Initial parameter(s).
    track_grad : bool
        If True, record gradient norms in the results.
    track_lr_per_param : bool
        If True, record per-parameter LR in the results.
    noise_scale : float or None
        If not None, a Gaussian noise of this scale is added to the gradient.

    Returns
    -------
    dict
        {(opt_name, lr_val): output_dict_from_run_optimizer, ...}
    """
    results = {}
    for opt_name, opt_info in optimizers.items():
        func = opt_info["func"]
        raw_grad_fn = opt_info["func_grad"]
        init_state_fn = opt_info.get("init_state_fn", lambda: None)

        for lr_val in opt_info["lrs"]:
            out = run_optimizer(
                update_fn=opt_info["update_fn"],
                func=func,
                func_grad=raw_grad_fn,
                x_init=x_init,
                lr=lr_val,
                num_iters=num_iters,
                init_state_fn=init_state_fn,
                track_grad=track_grad,
                track_lr_per_param=track_lr_per_param
            )
            results[(opt_name, lr_val)] = out

    return results


def plot_convergence_speed(
    optimizers: dict,
    num_iters: int,
    x_init: float = 0.0
):
    """
    Plots "convergence speed" (loss vs. iteration) for multiple optimizers.
    Each optimizer may have multiple learning rates in its config.
    """
    # Collect trajectories (no gradient tracking needed).
    results = collect_trajectories(
        optimizers, num_iters, x_init,
        track_grad=False, track_lr_per_param=False
    )

    plt.figure(figsize=(7, 5))

    for (opt_name, lr_val), output in results.items():
        label = f"{opt_name} (initial_lr={lr_val})"
        plt.plot(output["loss_vals"], label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Convergence Speed & Stability")
    plt.legend()
    plt.show()


def simulate_and_plot(optimizers_config, n_iterations=100, x_init=0.0, noise_scale=1.2):
    """
    Simulates training on a nonconvex function using the provided optimizer configurations,
    and plots both the gradient norms and loss convergence.
    """
    # Collect trajectories (with gradient tracking).
    results = collect_trajectories(
        optimizers_config,
        n_iterations,
        x_init,
        track_grad=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_grad, ax_loss = axes

    # 1) Gradient Norms
    for (opt_name, lr), res in results.items():
        label = f"{opt_name} (lr={lr})"
        ax_grad.plot(res["grad_vals"], label=label, marker="o")
    ax_grad.set_title("Gradient Norms Over Iterations")
    ax_grad.set_xlabel("Iteration")
    ax_grad.set_ylabel("Gradient Norm")
    ax_grad.legend()
    ax_grad.grid(True)

    # 2) Loss Convergence
    for (opt_name, lr), res in results.items():
        label = f"{opt_name} (lr={lr})"
        ax_loss.plot(res["loss_vals"], label=label, marker="o")
    ax_loss.set_title("Loss Convergence Over Iterations")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    plt.tight_layout()
    plt.show()


def animate(
    filename: str,
    optimizers: dict,
    num_iters: int,
    x_init: float = 0.0,
    domain: tuple = (0, 10),
    n_points: int = 500
):
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    if os.path.exists(f"artifacts/{filename}.mp4"):
        return Video(f"artifacts/{filename}.mp4")
    
    """
    Creates a matplotlib animation visualizing the parameter path x over iterations
    for multiple optimizers, each with possibly multiple LRs (1D).
    """
    # Collect trajectories (no gradient tracking).
    results = collect_trajectories(
        optimizers, num_iters, x_init,
        track_grad=False, track_lr_per_param=False
    )

    # Precompute the function plot over domain for the background curve
    first_opt = next(iter(optimizers.values()))
    plot_func = first_opt["func"]
    x_plot = np.linspace(domain[0], domain[1], n_points)
    y_plot = plot_func(x_plot)  # fully vectorized call, if your function is vector-safe

    # Precompute f(x) for each trajectory and iteration
    # shapes: (len(results), num_iters)
    keys_list = list(results.keys())
    f_trajectories = {}
    for key in keys_list:
        xs = results[key]["x_vals"]  # shape: (num_iters,)
        # Vectorized call if possible:
        f_trajectories[key] = plot_func(xs)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_plot, y_plot, color='gray', label='Function')

    # Markers for each (optimizer, lr)
    markers = {}
    text_labels = {}
    color_cycle = ["r", "g", "b", "m", "c", "y", "k"]

    base_y = 0.95
    spacing = 0.05
    for i, key in enumerate(keys_list):
        opt_name, lr_val = key
        color = color_cycle[i % len(color_cycle)]
        marker, = ax.plot([], [], color + "o", markersize=6)
        markers[key] = marker

        txt = ax.text(
            0.05, base_y - i*spacing, '',
            transform=ax.transAxes, color=color, fontsize=9, verticalalignment='top'
        )
        text_labels[key] = txt

    ax.set_xlim(domain[0], domain[1])
    ax.set_ylim(min(y_plot) - 0.5, max(y_plot) + 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Convergence Progression")
    ax.legend()

    # Pull out the x-vals and LR-vals from each run
    trajectories = {k: v["x_vals"] for k, v in results.items()}
    lrs_trajectories = {k: v["lr_vals"] for k, v in results.items()}

    def init_anim():
        for marker in markers.values():
            marker.set_data([], [])
        for txt in text_labels.values():
            txt.set_text('')
        return list(markers.values()) + list(text_labels.values())

    def update(frame):
        for key in keys_list:
            xs = trajectories[key]      # shape: (num_iters,)
            ys = f_trajectories[key]    # shape: (num_iters,) precomputed
            lr_array = lrs_trajectories[key]

            cur_x = xs[frame]
            cur_y = ys[frame]          # <--- we don’t call plot_func() here anymore
            eff_lr = lr_array[frame]

            markers[key].set_data([cur_x], [cur_y])
            opt_name, base_lr = key
            text_labels[key].set_text(f"{opt_name} LR: {eff_lr:.4f}")
        return list(markers.values()) + list(text_labels.values())

    ani = FuncAnimation(
        fig,
        update,
        frames=num_iters,
        init_func=init_anim,
        interval=20,
        blit=True
    )
    plt.close(fig)
    
    ani.save(f"artifacts/{filename}.mp4", writer="ffmpeg")
    return HTML(ani.to_jshtml())


def animate_heatmap(filename, optimizers, num_iters, x_init, grid_shape=None):
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    if os.path.exists(f"artifacts/{filename}.mp4"):
        return Video(f"artifacts/{filename}.mp4")
    
    """
    Creates an animation comparing effective learning rate heatmaps for multiple optimizers.
    
    Parameters
    ----------
    optimizers : dict
        Dictionary where each key is an optimizer name and the value is a dictionary with keys:
          - "lrs": list of learning rates (each will yield one trajectory)
          - "update_fn": the optimizer update function
          - "func": the objective function f(x)
          - "func_grad": the gradient function g(x)
          - "init_state_fn": function to initialize optimizer state (if any)
    num_iters : int
        Number of iterations to run each optimizer.
    x_init : array-like
        Initial parameter values (should be a vector; its length determines the heatmap grid).
    grid_shape : tuple or None, optional
        The shape (rows, cols) to reshape the effective learning rate vector into for visualization.
        If None, it will be inferred from the length of x_init.
    """
    # Collect full per-parameter LR
    results = collect_trajectories(
        optimizers, num_iters, x_init,
        track_grad=False, track_lr_per_param=True
    )

    # Extract LR arrays
    traj_data = {
        k: np.array(v["lr_vals_vec"])  # shape: (num_iters, ?)
        for k, v in results.items()
    }

    try:
        n_params = len(x_init)
    except TypeError:
        n_params = 1

    if grid_shape is None:
        rows = int(np.floor(np.sqrt(n_params)))
        cols = int(np.ceil(n_params / rows))
        grid_shape = (rows, cols)

    total_cells = grid_shape[0] * grid_shape[1]

    def reshape_lr_array(lr_array_1d):
        """
        Convert a 1D array of LRs into the (rows, cols) 2D array for heatmap.
        If it's a single scalar, replicate; if fewer than needed, pad with NaN; 
        if more, truncate.
        """
        arr = np.array(lr_array_1d).flatten()
        size = arr.size
        if size == 1:
            filled = np.full(total_cells, arr.item())
        elif size < total_cells:
            # Repeat the entire array enough times to fill the grid
            repeats = (total_cells + size - 1) // size  # number of times to tile
            tiled = np.tile(arr, repeats)               # e.g. [p1,p2,p3,...p1,p2,p3,...]
            filled = tiled[:total_cells]                # truncate to exact grid size
        else:
            filled = arr[:total_cells]
        return filled.reshape(grid_shape)

    # One subplot per (opt_name, lr)
    keys = list(traj_data.keys())
    total_traj = len(keys)
    fig, axes = plt.subplots(1, total_traj, figsize=(4 * total_traj, 4))
    if total_traj == 1:
        axes = [axes]

    heatmaps = {}
    texts = {}
    for ax, key in zip(axes, keys):
        first_lr = traj_data[key][0]
        data0 = reshape_lr_array(first_lr)

        hm = ax.imshow(data0, cmap='viridis', aspect='auto')
        ax.set_title(f"{key[0]} (lr={key[1]})", pad=20)
        txt = ax.text(
            0.05, 0.95, "Iteration: 0",
            ha="left", va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.6)
        )
        heatmaps[key] = hm
        texts[key] = txt

        plt.colorbar(hm, ax=ax, label="Effective LR")

    fig.tight_layout()
    
    def init_anim():
        artists = []
        for key in keys:
            data0 = reshape_lr_array(traj_data[key][0])
            heatmaps[key].set_array(data0)
            texts[key].set_text("Iteration: 0")
            artists.append(heatmaps[key])
            artists.append(texts[key])
        return artists

    def update(frame):
        artists = []
        for key in keys:
            lr_array_1d = traj_data[key][frame]
            data = reshape_lr_array(lr_array_1d)
            heatmaps[key].set_array(data)
            texts[key].set_text(f"Iteration: {frame}")
            artists.append(heatmaps[key])
            artists.append(texts[key])
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=num_iters,
        init_func=init_anim,
        interval=20,
        blit=True
    )

    plt.close(fig)
    ani.save(f"artifacts/{filename}.mp4", writer="ffmpeg")
    return HTML(ani.to_jshtml())
