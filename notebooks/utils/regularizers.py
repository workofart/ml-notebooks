import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.datasets import make_regression
from IPython.display import HTML

def plot_l1(filename):
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    if os.path.exists(f"artifacts/{filename}.mp4"):
        return Video(f"artifacts/{filename}.mp4")
    
    # 1. CREATE SYNTHETIC DATA
    # ------------------------
    X, y, coef_true = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=3,
        noise=10,
        coef=True,
        random_state=42
    )

    # Fit OLS once (no regularization) for reference
    lr = LinearRegression()
    lr.fit(X, y)
    coef_ols = lr.coef_
    y_pred_ols = lr.predict(X)

    # 2. SETUP THE FIGURE AND SUBPLOTS
    # --------------------------------
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5), tight_layout=True)

    # Left Subplot: bar chart for OLS vs. Lasso coefficients
    feature_indices = np.arange(len(coef_ols))
    bar_width = 0.4

    # OLS bars (stay fixed)
    bars_ols = ax1.bar(
        feature_indices - bar_width/2, 
        coef_ols, 
        width=bar_width, 
        label="OLS (alpha=0)", 
        color="C2", 
        alpha=0.7
    )

    # Lasso bars (will be updated in the animation)
    bars_lasso = ax1.bar(
        feature_indices + bar_width/2, 
        np.zeros_like(coef_ols),  # initially all zero
        width=bar_width, 
        label="Lasso (alpha varies)", 
        color="C3", 
        alpha=0.7
    )

    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xticks(feature_indices)
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Coefficient Value")
    ax1.set_title("Coefficients: OLS vs. Lasso (alpha = 0.00)")
    ax1.legend()

    # Right Subplot: predicted vs. actual for OLS and Lasso
    # OLS scatter (fixed)
    ols_scatter = ax2.scatter(y, y_pred_ols, alpha=0.6, label="OLS", color="C2")

    # Lasso scatter (will be updated)
    lasso_scatter = ax2.scatter([], [], alpha=0.6, label="Lasso", color="C3")

    # Identity line for visual reference
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=1)
    ax2.set_xlabel("True y")
    ax2.set_ylabel("Predicted y")
    ax2.set_title("Predicted vs. Actual")
    ax2.legend()

    # 3. DEFINE THE ANIMATION LOGIC
    # -----------------------------
    alphas = np.linspace(0.01, 10, 100)  # You can adjust the range and number of steps

    def update(frame):
        alpha_val = alphas[frame]
        
        # Fit a new Lasso model at this alpha
        lasso = Lasso(alpha=alpha_val, max_iter=10000, random_state=42)
        lasso.fit(X, y)
        coef_lasso = lasso.coef_
        y_pred_lasso = lasso.predict(X)
        
        # Update bar heights for Lasso
        for bar, new_height in zip(bars_lasso, coef_lasso):
            bar.set_height(new_height)
        
        # Update the title
        ax1.set_title(f"Coefficients: OLS vs. Lasso (alpha = {alpha_val:.2f})")
        
        # Update the Lasso scatter (predicted vs. actual)
        # set_offsets() expects an Nx2 array of [x, y] pairs
        xy = np.column_stack((y, y_pred_lasso))
        lasso_scatter.set_offsets(xy)
        
        return bars_lasso, lasso_scatter

    # 4. CREATE THE ANIMATION
    # ------------------------
    ani = FuncAnimation(
        fig, update, frames=len(alphas), 
        interval=20,  # ms per frame
        blit=False
    )

    plt.close(fig)
    ani.save(f"artifacts/{filename}.mp4", writer="ffmpeg")
    return HTML(ani.to_jshtml())