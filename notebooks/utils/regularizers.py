import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, Video
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

np.random.seed(1337)
EPSILON = 1e-5

def prepare_video(filename):
    """
    1) Ensures the 'artifacts/' directory exists.
    2) Checks if the MP4 already exists; if yes, returns a Video object immediately.
       Otherwise, returns (video_path, None).
    """
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    video_path = f"artifacts/{filename}.mp4"
    if os.path.exists(video_path):
        # MP4 already exists, so return a Video object immediately.
        return video_path, Video(video_path)
    else:
        # Need to create it; return the intended path and None.
        return video_path, None

def finalize_animation(fig, ani, video_path):
    """
    Closes the figure, saves the animation to the given path, and returns
    an HTML-wrapped animation for inline display.
    """
    plt.close(fig)
    ani.save(video_path, writer="ffmpeg")
    return HTML(ani.to_jshtml())


def plot_regularization(filename, reg_type="L1", reg_strength=10):
    # Step A: Prepare video (ensure artifacts directory exists and check for existing MP4)
    video_path, existing_video = prepare_video(filename)
    if existing_video is not None:
        return existing_video

    # Select the regularization model based on reg_type.
    if reg_type == "L1":
        from sklearn.linear_model import Lasso
        reg_model = Lasso
        reg_label = "Lasso"
    elif reg_type == "L2":
        from sklearn.linear_model import Ridge
        reg_model = Ridge
        reg_label = "Ridge"
    else:
        raise ValueError("reg_type must be 'L1' or 'L2'")

    # 1. CREATE SYNTHETIC DATA
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=3,
        noise=10,
        coef=False,
        random_state=1337
    )

    # Fit Ordinary Least Squares (OLS) for reference.
    lr = LinearRegression()
    lr.fit(X, y)
    coef_ols = lr.coef_
    y_pred_ols = lr.predict(X)

    # 2. SETUP THE FIGURE AND SUBPLOTS
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5), tight_layout=True)
    feature_indices = np.arange(len(coef_ols))
    bar_width = 0.4

    # Left subplot: Bar plots for OLS vs. Regularized coefficients.
    ax1.bar(
        feature_indices - bar_width / 2,
        coef_ols,
        width=bar_width,
        label="OLS (alpha=0)",
        color="C2",
        alpha=0.7
    )
    bars_reg = ax1.bar(
        feature_indices + bar_width / 2,
        np.zeros_like(coef_ols),  # initial regularized coefficients (zeros)
        width=bar_width,
        label=f"{reg_label} (alpha varies)",
        color="C3",
        alpha=0.7
    )
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xticks(feature_indices)
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Coefficient Value")
    ax1.set_title(f"Coefficients: OLS vs. {reg_label} (alpha = 0.00)")
    ax1.legend()

    # Right subplot: Scatter plot for predicted vs. actual values.
    ols_scatter = ax2.scatter(y, y_pred_ols, alpha=0.6, label="OLS", color="C2")
    reg_scatter = ax2.scatter([], [], alpha=0.6, label=reg_label, color="C3")
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=1)
    ax2.set_xlabel("True y")
    ax2.set_ylabel("Predicted y")
    ax2.set_title("Predicted vs. Actual")
    ax2.legend()

    # 3. DEFINE THE ANIMATION LOGIC
    alphas = np.linspace(0.01, reg_strength, 100)  # Range of regularization strengths

    def update(frame):
        alpha_val = alphas[frame]
        # Instantiate and fit the appropriate regularization model.
        if reg_type == "L1":
            model = reg_model(alpha=alpha_val, max_iter=10000, random_state=1337)
        else:  # L2
            model = reg_model(alpha=alpha_val, random_state=1337)
        model.fit(X, y)
        coef_reg = model.coef_
        y_pred_reg = model.predict(X)

        # Update the bar heights for the regularized coefficients.
        for bar, new_height in zip(bars_reg, coef_reg):
            bar.set_height(new_height)

        # Update the title with the current regularization strength.
        ax1.set_title(f"Coefficients: OLS vs. {reg_label} (alpha = {alpha_val:.2f})")

        # Update the scatter plot for predicted values.
        xy = np.column_stack((y, y_pred_reg))
        reg_scatter.set_offsets(xy)
        return bars_reg, reg_scatter

    # 4. CREATE THE ANIMATION
    ani = FuncAnimation(fig, update, frames=len(alphas), interval=20, blit=False)

    # Step B: Finalize the animation (close figure, save MP4, and return HTML-wrapped video)
    return finalize_animation(fig, ani, video_path)

def plot_layernorm(filename, gamma=1.5, beta=0.5, n_frames=100, num_features=20):
    # Ensure the artifacts directory exists.
    video_path, existing_video = prepare_video(filename)
    if existing_video is not None:
        return existing_video

    # 1. Generate a single input with multiple features.
    # Simulate activations with non-zero mean and non-unit variance.
    x = np.random.randn(num_features) * 2 + 3  # For example: mean ~3, std ~2

    # 2. Compute Layer Norm Statistics and Transformations.
    mu = np.mean(x)
    sigma2 = np.var(x)
    sigma = np.sqrt(sigma2 + EPSILON)
    # Normalized activations: each feature centered and scaled.
    x_hat = (x - mu) / sigma
    # Final output after applying learnable scale (γ) and shift (β).
    y_final = gamma * x_hat + beta

    # 3. Set up the figure.
    fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter plot: x-axis represents feature index, y-axis represents activation value.
    scatter = ax.scatter(np.arange(num_features), x, c='C0', s=50)
    # Horizontal line representing the current mean.
    mean_line = ax.axhline(mu, color='gray', linestyle='--', label="Mean")
    # Determine y-axis limits to cover all stages of transformation.
    y_all = np.concatenate([x, x_hat, y_final])
    y_min, y_max = y_all.min() - 1, y_all.max() + 1
    ax.set_xlim(-1, num_features)
    ax.set_ylim(y_min, y_max)
    title_text = ax.set_title("Layer Normalization: Original Activations")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Activation Value")
    ax.legend()

    # 4. Define the animation update function.
    def update(frame):
        half_frames = n_frames // 2
        if frame <= half_frames:
            # Stage 1: Transition from original activations (x) to normalized activations (x_hat).
            t = frame / half_frames  # t goes from 0 to 1.
            current_values = (1 - t) * x + t * x_hat
            stage = "Normalization"
            # Interpolate the mean from the original mean to 0.
            current_mean = (1 - t) * mu + t * 0
        else:
            # Stage 2: Transition from normalized activations (x_hat) to scaled & shifted outputs (y_final).
            t = (frame - half_frames) / half_frames  # t goes from 0 to 1.
            current_values = (1 - t) * x_hat + t * y_final
            stage = "Scale and Shift"
            # Interpolate the mean from 0 to β.
            current_mean = (1 - t) * 0 + t * beta

        # Update scatter plot with new activation values.
        scatter.set_offsets(np.column_stack((np.arange(num_features), current_values)))
        # Update the horizontal line representing the mean (as a two-point sequence).
        mean_line.set_ydata([current_mean, current_mean])
        # Update the title to indicate the current transformation stage.
        if frame <= half_frames:
            title_text.set_text(f"Layer Norm: {stage} (Centering activations)")
        else:
            title_text.set_text(f"Layer Norm: {stage} (γ={gamma}, β={beta})")
        return scatter, mean_line, title_text

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    return finalize_animation(fig, ani, video_path)


def plot_batchnorm(
    filename,
    gamma=1.0,
    beta=0.0,
    n_frames=100,
    batch_size=32,
    num_features=3
):
    """
    Demonstrates Batch Normalization on a dataset of shape (batch_size, num_features).
    The dataset is chosen to highlight when batch normalization is more appropriate
    (larger batch, fewer features).
    """
    # Ensure the artifacts directory exists
    video_path, existing_video = prepare_video(filename)
    if existing_video is not None:
        return existing_video

    # 1. Generate random data (batch_size x num_features)
    #    Larger batch_size to illustrate BN over the batch dimension.
    x = np.random.randn(batch_size, num_features) * 2.0 + 3.0  # Shifted and scaled random data

    # 2. Compute batch normalization statistics & transform
    #    BN is computed per feature across all samples in the batch.
    mu = np.mean(x, axis=0)        # shape: (num_features,)
    var = np.var(x, axis=0)        # shape: (num_features,)
    x_hat = (x - mu) / np.sqrt(var + EPSILON)  # normalized across batch
    y_bn = gamma * x_hat + beta            # apply learnable scale & shift

    # 3. Set up figure & animation
    #    We'll visualize each feature's activation across the batch dimension.
    fig, ax = plt.subplots(figsize=(7, 5))
    # We plot each feature as a separate scatter series
    feature_scatters = []
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for f_idx in range(num_features):
        scat = ax.scatter(np.arange(batch_size), x[:, f_idx], 
                          color=colors[f_idx % len(colors)], label=f"Feature {f_idx}")
        feature_scatters.append(scat)

    # Lines to show the mean for each feature
    mean_lines = []
    for f_idx in range(num_features):
        line = ax.axhline(mu[f_idx], color=colors[f_idx % len(colors)], linestyle="--", alpha=0.5)
        mean_lines.append(line)

    ax.set_title("Batch Normalization (Raw Data)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Activation Value")
    ax.set_xlim([-1, batch_size])
    # Determine suitable y-limits
    all_vals = np.concatenate([x.flatten(), x_hat.flatten(), y_bn.flatten()])
    y_min, y_max = all_vals.min() - 1, all_vals.max() + 1
    ax.set_ylim([y_min, y_max])
    ax.legend()

    # Animation update function
    def update(frame):
        half = n_frames // 2
        if frame <= half:
            # Interpolate from raw data x to normalized x_hat
            t = frame / half
            current_data = (1 - t) * x + t * x_hat
            current_mu = (1 - t) * mu + t * np.zeros_like(mu)  # mean transitions to 0
            stage_text = "Batch Norm: Centering & Scaling (x → x_hat)"
        else:
            # Interpolate from x_hat to final BN output (y_bn)
            t = (frame - half) / half
            current_data = (1 - t) * x_hat + t * y_bn
            # Mean transitions from 0 to beta for each feature
            current_mu = (1 - t) * np.zeros_like(mu) + t * (beta * np.ones_like(mu))
            stage_text = "Batch Norm: Applying γ, β (x_hat → y_bn)"

        # Update scatter points
        for f_idx, scat in enumerate(feature_scatters):
            new_points = np.column_stack((np.arange(batch_size), current_data[:, f_idx]))
            scat.set_offsets(new_points)
            # Update mean line
            mean_lines[f_idx].set_ydata([current_mu[f_idx], current_mu[f_idx]])

        ax.set_title(stage_text)
        return feature_scatters + mean_lines

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    return finalize_animation(fig, ani, video_path)


def plot_layernorm_on_large_batch(
    filename="layernorm_on_large_batch",
    gamma=1.0,
    beta=0.0,
    n_frames=100,
    batch_size=32,   # Large batch
    num_features=3   # Few features
):
    """
    Demonstrates Layer Normalization on a dataset of shape (batch_size, num_features),
    where the batch is large but each sample has only a few features. This is
    typically not where LayerNorm excels, so it serves as an illustration
    of the 'wrong' scenario for LN.
    """
    video_path, existing_video = prepare_video(filename)
    if existing_video is not None:
        return existing_video

    # 1. Generate data (large batch of 32 samples, each with only 3 features)
    x = np.random.randn(batch_size, num_features) * 2 + 3  # shape (32, 3)

    # 2. Compute LN across features for each sample
    #    That means for each sample i, we do mean/var across the feature dimension
    mu = np.mean(x, axis=1, keepdims=True)  # shape (32, 1)
    var = np.var(x, axis=1, keepdims=True)  # shape (32, 1)
    x_hat = (x - mu) / np.sqrt(var + EPSILON)    # shape (32, 3)
    y_ln = gamma * x_hat + beta

    # 3. Set up figure
    fig, ax = plt.subplots(figsize=(7, 5))
    # We'll visualize each feature dimension with a different color,
    # plotting all 32 samples along the x-axis.
    colors = ["C0", "C1", "C2", "C3", "C4"]
    feature_scatters = []
    for f_idx in range(num_features):
        scat = ax.scatter(
            np.arange(batch_size),
            x[:, f_idx],
            color=colors[f_idx % len(colors)],
            label=f"Feature {f_idx}"
        )
        feature_scatters.append(scat)

    ax.set_title("LayerNorm on Large Batch (Raw Data)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Activation Value")
    ax.set_xlim([-1, batch_size])
    # We’ll combine data to find min/max for y-limits
    all_vals = np.concatenate([x.flatten(), x_hat.flatten(), y_ln.flatten()])
    y_min, y_max = all_vals.min() - 1, all_vals.max() + 1
    ax.set_ylim([y_min, y_max])
    ax.legend()

    # We won't draw one single "mean line" here because LN is done per-sample,
    # meaning there's a separate mean for each sample. Instead, we just show the
    # transitions of the data points. If desired, one could draw 32 horizontal
    # lines, each for the sample's mean—but that might get cluttered.

    def update(frame):
        half = n_frames // 2
        if frame <= half:
            # Interpolate from x to x_hat
            t = frame / half
            current_data = (1 - t) * x + t * x_hat
            stage_text = "LayerNorm: (x → x_hat)"
        else:
            # Interpolate from x_hat to y_ln
            t = (frame - half) / half
            current_data = (1 - t) * x_hat + t * y_ln
            stage_text = "LayerNorm: (x_hat → y_ln)"

        # Update scatter positions
        for f_idx, scat in enumerate(feature_scatters):
            new_points = np.column_stack((np.arange(batch_size), current_data[:, f_idx]))
            scat.set_offsets(new_points)

        ax.set_title(stage_text)
        return feature_scatters

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    return finalize_animation(fig, ani, video_path)


def plot_batchnorm_on_small_batch(
    filename="batchnorm_on_small_batch",
    gamma=1.0,
    beta=0.0,
    n_frames=100,
    batch_size=4,    # Very small batch
    num_features=20  # Many features
):
    """
    Demonstrates Batch Normalization on a dataset of shape (batch_size, num_features),
    where the batch size is quite small (4) but the number of features is large (20).
    This is generally less stable for BN, since it relies on robust statistics across
    the batch dimension.
    """
    video_path, existing_video = prepare_video(filename)
    if existing_video is not None:
        return existing_video

    # 1. Generate data (small batch, large number of features)
    x = np.random.randn(batch_size, num_features) * 2 + 3  # shape (4, 20)

    # 2. Compute BN across the batch dimension for each feature
    #    BN means we compute mean/var for each feature across all samples in the batch.
    mu = np.mean(x, axis=0)    # shape (20,)
    var = np.var(x, axis=0)    # shape (20,)
    x_hat = (x - mu) / np.sqrt(var + EPSILON)  # shape (4, 20)
    y_bn = gamma * x_hat + beta            # shape (4, 20)

    # 3. Set up figure
    fig, ax = plt.subplots(figsize=(7, 5))
    # We'll show the distribution of each feature across the 4 samples (x-axis: sample idx)
    # Because there are 20 features, let's try to color by feature index. This can get busy,
    # but it illustrates the concept.
    colors = [f"C{i%10}" for i in range(num_features)]
    feature_scatters = []
    for f_idx in range(num_features):
        scat = ax.scatter(
            np.arange(batch_size),
            x[:, f_idx],
            color=colors[f_idx],
            s=25,
            alpha=0.8
        )
        feature_scatters.append(scat)

    ax.set_title("BatchNorm on Small Batch (Raw Data)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Activation Value")
    ax.set_xlim([-1, batch_size])
    # Combine data to find min/max for y-limits
    all_vals = np.concatenate([x.flatten(), x_hat.flatten(), y_bn.flatten()])
    y_min, y_max = all_vals.min() - 1, all_vals.max() + 1
    ax.set_ylim([y_min, y_max])

    def update(frame):
        half = n_frames // 2
        if frame <= half:
            # Interpolate from x to x_hat
            t = frame / half
            current_data = (1 - t) * x + t * x_hat
            stage_text = "BatchNorm: (x → x_hat)"
        else:
            # Interpolate from x_hat to y_bn
            t = (frame - half) / half
            current_data = (1 - t) * x_hat + t * y_bn
            stage_text = "BatchNorm: (x_hat → y_bn)"

        for f_idx, scat in enumerate(feature_scatters):
            new_points = np.column_stack((np.arange(batch_size), current_data[:, f_idx]))
            scat.set_offsets(new_points)

        ax.set_title(stage_text)
        return feature_scatters

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    return finalize_animation(fig, ani, video_path)


def plot_dropout(filename, p=0.5, n_frames=100, num_features=20):
    # Ensure the artifacts directory exists.
    video_path, existing_video = prepare_video(filename)
    if existing_video is not None:
        return existing_video

    # 1. Generate a single input with multiple neurons.
    # Simulate activations with a non-zero mean and non-unit variance.
    x = np.random.randn(num_features) * 2 + 3  # For example: mean ~3, std ~2

    # 2. Generate dropout mask and compute dropout outcomes.
    # Each neuron is kept (mask=1) with probability (1-p) and dropped (mask=0) with probability p.
    m = (np.random.rand(num_features) < (1-p)).astype(float)
    # Apply dropout: dropped neurons become 0.
    x_dropped = x * m
    # Scale the retained neurons by 1/(1-p) to maintain the expected activation value.
    x_scaled = x_dropped / (1-p)

    # Compute the linear combination (sum) of the original activations.
    original_output = np.sum(x)

    # 3. Set up the figure.
    fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter plot: x-axis represents neuron index, y-axis represents activation value.
    scatter = ax.scatter(np.arange(num_features), x, c='C1', s=50)
    ax.set_xlim(-1, num_features)
    # Determine y-axis limits based on original, dropped, and scaled activations.
    y_all = np.concatenate([x, x_dropped, x_scaled])
    y_min, y_max = y_all.min() - 1, y_all.max() + 1
    ax.set_ylim(y_min, y_max)
    title_text = ax.set_title("Dropout: Original Activations")
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Activation Value")

    # Add a text label to compare the final output (linear combination) of activations.
    comparison_text = ax.text(0.5, 0.85, '', transform=ax.transAxes, fontsize=12,
                              color='red', ha='center')

    # 4. Define the animation update function.
    def update(frame):
        half_frames = n_frames // 2
        if frame <= half_frames:
            # Stage 1: Transition from original activations (x) to dropout-applied activations (x_dropped).
            t = frame / half_frames  # t goes from 0 to 1.
            current_values = (1 - t) * x + t * x_dropped
            stage = "Applying Dropout Mask"
        else:
            # Stage 2: Transition from dropout-applied activations (x_dropped) to scaled activations (x_scaled).
            t = (frame - half_frames) / half_frames  # t goes from 0 to 1.
            current_values = (1 - t) * x_dropped + t * x_scaled
            stage = "Scaling Retained Neurons"

        # Update scatter plot with new activation values.
        scatter.set_offsets(np.column_stack((np.arange(num_features), current_values)))
        # Update the title to indicate the current transformation stage.
        title_text.set_text(f"Dropout: {stage}")
        
        # Compute the linear combination (sum) of the current dropout activations.
        dropout_output = np.sum(current_values)
        # Update the comparison label with the current output and the original activations.
        comparison_text.set_text(
            f"Output with Dropout: {dropout_output:.2f} | Output without Dropout: {original_output:.2f}"
        )
        return scatter, title_text, comparison_text

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    return finalize_animation(fig, ani, video_path)

def plot_label_smoothing(filename, K=5, n_frames=100):
    # Ensure the artifacts directory exists.
    video_path, existing_video = prepare_video(filename)
    if existing_video is not None:
        return existing_video

    # 1. Set up the original one-hot label for a given class.
    correct_class = 2  # For example, assume the correct class is index 2 (0-based).
    y_onehot = np.zeros(K)
    y_onehot[correct_class] = 1.0

    # 2. Define the smoothed label distribution.
    #    Correct class: 1 - epsilon + epsilon/K
    #    Other classes: epsilon/K
    y_smooth = np.full(K, 0.2 / K)
    y_smooth[correct_class] = 1.0 - 0.2 + (0.2 / K)

    # 3. Set up the figure for animation.
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(np.arange(K), y_onehot, color='C0')
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Class Index")
    ax.set_ylabel("Label Probability")
    title_text = ax.set_title("Label Smoothing: Original One-Hot Label")

    # 4. Define the animation update function to transition from one-hot to smoothed labels.
    def update(frame):
        # t goes from 0 to 1 as frames progress.
        t = frame / (n_frames - 1)
        current_values = (1 - t) * y_onehot + t * y_smooth

        # Update the bar heights to reflect the current label distribution.
        for bar, val in zip(bars, current_values):
            bar.set_height(val)

        title_text.set_text(f"Label Smoothing: Transition (t={t:.2f})")
        return bars, title_text

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    return finalize_animation(fig, ani, video_path)



def plot_label_smoothing_comparison(
    filename,
    noise_rate=0.1,
    n_frames=50,
    test_size=0.2  # Fraction of data reserved for testing
):
    video_path, existing_video = prepare_video(filename)
    if existing_video is not None:
        return existing_video

    # -------------------------------------------------------------------------
    # 1. Data Generation and Train-Test Split
    # -------------------------------------------------------------------------
    N = 100
    mean0, cov0 = [2, 2], [[0.5, 0.0], [0.0, 0.5]]
    mean1, cov1 = [4, 4], [[0.5, 0.0], [0.0, 0.5]]
    X0 = np.random.multivariate_normal(mean0, cov0, N//2)
    X1 = np.random.multivariate_normal(mean1, cov1, N//2)
    X = np.vstack((X0, X1))
    y = np.array([0]*(N//2) + [1]*(N//2))
    
    # Shuffle the data
    perm = np.random.permutation(N)
    X = X[perm]
    y = y[perm]

    # Split into training and test sets
    split_idx = int(N * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Apply label noise only to the training set
    n_flip = int(noise_rate * len(y_train))
    flip_indices = np.random.choice(len(y_train), size=n_flip, replace=False)
    y_train[flip_indices] = 1 - y_train[flip_indices]
    
    # Augment features with bias term
    X_train_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
    X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])

    # -------------------------------------------------------------------------
    # 2. Initialize Logistic Regression Models
    # -------------------------------------------------------------------------
    w_no_ls = np.random.randn(X_train_aug.shape[1]) * 0.01
    w_ls    = np.random.randn(X_train_aug.shape[1]) * 0.01
    lr = 0.2  # Small learning rate to prevent saturation

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(X_aug, w):
        return sigmoid(X_aug @ w)

    def gradient_no_ls(X_aug, y, w):
        p = predict_proba(X_aug, w)
        return X_aug.T @ (p - y) / len(y)

    def gradient_ls(X_aug, y, w):
        p = predict_proba(X_aug, w)
        # Apply label smoothing to the labels
        label_smoothing_epsilon = 0.3
        y_smooth = y * (1 - label_smoothing_epsilon) + (1 - y) * label_smoothing_epsilon
        return X_aug.T @ (p - y_smooth) / len(y)

    # -------------------------------------------------------------------------
    # 3. Setup Plots: (a) Decision Boundaries, (b) Test Accuracy vs. Epoch
    # -------------------------------------------------------------------------
    fig, (ax_bound, ax_acc) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Test Accuracy: Label Smoothing vs. No Smoothing", fontsize=14)

    # (a) Decision Boundary Plot (using training data)
    scatter = ax_bound.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
    ax_bound.set_xlabel("x1")
    ax_bound.set_ylabel("x2")
    boundary_no_ls, = ax_bound.plot([], [], 'b-', label="No Smoothing")
    boundary_ls,    = ax_bound.plot([], [], 'r--', label="Label Smoothing")
    ax_bound.legend()

    # (b) Test Accuracy Plot
    ax_acc.set_xlim(0, n_frames)
    ax_acc.set_ylim(0, 1)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Test Accuracy")
    line_no_ls, = ax_acc.plot([], [], 'o-', color='blue', label="No Smoothing")
    line_ls,    = ax_acc.plot([], [], 'o-', color='orange', label="Label Smoothing")
    ax_acc.legend()

    # For plotting decision boundaries, create a mesh grid based on training data limits
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_aug = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])

    acc_no_ls_list = []
    acc_ls_list = []

    # -------------------------------------------------------------------------
    # 4. Animation Update Function
    # -------------------------------------------------------------------------
    def update(frame):
        nonlocal w_no_ls, w_ls

        # Update parameters using training data
        grad_nls = gradient_no_ls(X_train_aug, y_train, w_no_ls)
        grad_ls_ = gradient_ls(X_train_aug, y_train, w_ls)
        w_no_ls -= lr * grad_nls
        w_ls    -= lr * grad_ls_

        # --- Decision Boundary Updates (on training data) ---
        p_nls = predict_proba(grid_aug, w_no_ls).reshape(xx.shape)
        p_ls_ = predict_proba(grid_aug, w_ls).reshape(xx.shape)
        ax_bound.set_title(f"Decision Boundaries (Epoch {frame})")
        boundary_no_ls.set_data([], [])
        boundary_ls.set_data([], [])

        cs_nls = ax_bound.contour(xx, yy, p_nls, levels=[0.5], colors='blue')
        if cs_nls.collections:
            paths_nls = cs_nls.collections[0].get_paths()
            if paths_nls:
                max_path_nls = max(paths_nls, key=lambda p: p.vertices.shape[0])
                boundary_no_ls.set_data(max_path_nls.vertices[:, 0],
                                        max_path_nls.vertices[:, 1])
        for coll in cs_nls.collections:
            coll.remove()

        cs_ls = ax_bound.contour(xx, yy, p_ls_, levels=[0.5], colors='red', linestyles='--')
        if cs_ls.collections:
            paths_ls = cs_ls.collections[0].get_paths()
            if paths_ls:
                max_path_ls = max(paths_ls, key=lambda p: p.vertices.shape[0])
                boundary_ls.set_data(max_path_ls.vertices[:, 0],
                                     max_path_ls.vertices[:, 1])
        for coll in cs_ls.collections:
            coll.remove()

        # --- Test Accuracy Calculation ---
        p_nls_test = predict_proba(X_test_aug, w_no_ls)
        p_ls_test  = predict_proba(X_test_aug, w_ls)
        pred_nls = (p_nls_test >= 0.5).astype(int)
        pred_ls_ = (p_ls_test  >= 0.5).astype(int)
        acc_no_ls = (pred_nls == y_test).mean()
        acc_ls    = (pred_ls_ == y_test).mean()

        acc_no_ls_list.append(acc_no_ls)
        acc_ls_list.append(acc_ls)
        line_no_ls.set_data(range(len(acc_no_ls_list)), acc_no_ls_list)
        line_ls.set_data(range(len(acc_ls_list)), acc_ls_list)
        ax_acc.set_title(
            f"Test Accuracy (Epoch {frame})\n"
            f"No Smoothing: {acc_no_ls:.2f}, Label Smoothing: {acc_ls:.2f}"
        )

        return (boundary_no_ls, boundary_ls, line_no_ls, line_ls)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=10, blit=False)
    plt.tight_layout()
    return finalize_animation(fig, ani, video_path)