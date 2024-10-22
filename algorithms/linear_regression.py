import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

"""
Ten baseline variables, age, sex, body mass index,
average blood pressure, and six blood serum measurements
were obtained for each of n = 442 diabetes patients,
as well as the response of interest, a quantitative
measure of disease progression one year after baseline.
"""
ds = load_diabetes()
print(ds.data.shape)
print(ds.feature_names)

# Linear regression formulation
# y = mx + b


def compute_linear_reg_line(feature_idx):
    X = ds.data[:, feature_idx]  # taking the first feature as example
    Y = ds.target

    # The objective is sum(y_i - y-hat_i) ^ 2
    # sum(y_i - mx_i - b) ^ 2
    # take partial derivative w.r.t m
    # sum(x_i * y_i) - m * sum(x_i^2) - b * sum(x_i) = 0
    # take partial derivative w.r.t b
    # sum(y_i) - m * sum(x_i) - n * b = 0
    n = len(X)
    sum_xy = sum(X * Y)
    sum_x = sum(X)
    sum_y = sum(Y)
    sum_x_squared = sum(X**2)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    b = (sum_y - m * sum_x) / n

    y_hat = m * X + b

    sns.scatterplot(x=X, y=y_hat, color="blue")
    sns.scatterplot(x=X, y=Y, color="red")
    plt.xlabel("X")
    plt.ylabel("Y or Y-hat")
    plt.title(
        f"Scatter Plot of Predicted vs Actual w.r.t. Feature[{ds.feature_names[i]}]"
    )
    plt.legend()

    # Show the plot
    plt.show()


for i in range(ds.data.shape[1]):
    compute_linear_reg_line(i)
