import matplotlib.pyplot as plt

def plot_y_pred(y, y_pred):
    mid_point = (min(y) + max(y)) // 2
    # Plotting the true labels and predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y)), y, color='red', marker='o', label='True Labels (y)')
    plt.scatter(range(len(y_pred)), y_pred, color='blue', marker='x', label='Predicted Values (y_pred)')
    plt.axhline(mid_point, color='gray', linestyle='--', label=f'Decision Boundary ({mid_point})')
    plt.title('True Labels vs Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

