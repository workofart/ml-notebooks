import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from probability import compute_bayes_rule


def simulate_diagnosis(p_cancer, accuracy_range, false_positive_range):
    """
    Function to simulate diagnosis and update p_cancer
    """

    # Each diagnosis has a different accuracy rate p(test positive | cancer)
    accuracies = [round(np.random.uniform(*accuracy_range), 2) for _ in range(N)]

    # We will define false positive rate p(test positive | no cancer), because defining the p(test positive) will inherently need to ensure it's consistent with the accuracy above
    false_positive_rates = [round(np.random.uniform(*false_positive_range), 2) for _ in range(N)]

    p_cancer_values = [p_cancer]  # Track p_cancer over diagnoses
    parameters = []  # Track parameters for each diagnosis

    for i in range(N):
        probs = compute_bayes_rule(
            p_cancer=p_cancer,
            # accuracy = p(test positive | cancer)
            p_test_positive_given_cancer=accuracies[i],
            # p(test positive | no cancer)
            p_test_positive_given_no_cancer=false_positive_rates[i],
        )

        # Now we randomize the probability that we receive a positive or negative diagnosis
        # Then we update our belief about getting cancer condtional on the new diagnosis information
        if np.random.random() < 0.2:
            p_cancer = probs["p_cancer_given_test_positive"]
            diagnosis = "Positive"
        else:
            p_cancer = probs["p_cancer_given_test_negative"]
            diagnosis = "Negative"

        p_cancer_values.append(p_cancer)
        parameters.append((accuracies[i], false_positive_rates[i], p_cancer, diagnosis))

    return p_cancer_values, parameters


def run_simulation_on_parameters(parameter_sets):
    """Simulate diagnosis for each parameter set and collect results."""
    all_p_cancer_values = []
    all_parameters = []
    for p_cancer_initial, accuracy_range, false_positive_range, _ in parameter_sets:
        p_cancer_values, parameters = simulate_diagnosis(p_cancer_initial, accuracy_range, false_positive_range)
        all_p_cancer_values.append(p_cancer_values)
        all_parameters.append(parameters)
    return all_p_cancer_values, all_parameters

# Initialize plot
def initialize_plot():
    """Set up the figure and axis for the animation."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1)
    line, = ax.plot([], [], lw=2)
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    return fig, ax, line, text

# Initialization function
def init():
    """Initialize the animation with empty data."""
    line.set_data([], [])
    text.set_text('')
    return line, text

# Animation function
def animate(frame):
    # Determine which parameter set and diagnosis we are on
    param_set_index, diagnosis_index = divmod(frame, N)
    p_cancer_values = all_p_cancer_values[param_set_index]
    parameters = all_parameters[param_set_index]

    # Get the current parameter set
    p_cancer_initial, accuracy_range, false_positive_range, title = parameter_sets[param_set_index]

    # Update the plot with the current diagnosis
    x = np.arange(diagnosis_index + 1)
    y = p_cancer_values[:diagnosis_index + 1]
    line.set_data(x, y)

    # Display the parameters for the current diagnosis
    p_test_positive_given_cancer, p_test_positive_given_no_cancer, p_cancer, diagnosis = parameters[diagnosis_index]
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


if __name__ == "__main__":
    N = 25
    # Define different sets of parameters for each iteration
    parameter_sets = [
        # Normal Case
        # low initial cancer belief, high accuracy, low false-positive rate
        (0.005, (0.8, 0.9), (0.01, 0.03), "normal case"),
        # Highly self-doubt + High Variance Diagnoses
        # high initial cancer belief, low accuracy, varying false-positive rate
        (0.5, (0.5, 0.5), (0.01, 0.5), "high initial self-doubt + high accuracy/false-positive rate variance"),
        # High false-positive and low accuracy Diagnoses
        # low initial cancer belief, high accuracy, high false-positive rate
        (0.005, (0.3, 0.5), (0.3, 0.5), "high false-positive and low accuracy rate"),
    ]

    # Prepare data
    all_p_cancer_values, all_parameters = run_simulation_on_parameters(parameter_sets)

    # Initialize plot
    fig, ax, line, text = initialize_plot()

    # Create the animation
    ani = FuncAnimation(fig, animate, init_func=init, frames=N * len(parameter_sets), interval=150, blit=True)

    # Display the animation
    plt.xlabel('Diagnosis Number')
    plt.ylabel('Probability of Cancer')
    plt.title('Evolution of p(cancer) Over Diagnoses with Varying Parameters')
    plt.show()
