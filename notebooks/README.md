## Notebooks

This folder contains some notebooks that implement and visualizes the algorithms mentioned below:

### 1. Bayesian Methods ([bayes.ipynb](https://github.com/workofart/ml-algorithms/blob/main/notebooks/bayes.ipynb))
- Implementation of Bayes' Theorem that demonstrates probability updates with new evidence
- Visualize the final posterior probability through medical diagnosis simulation with varying test accuracies and false positive rates setup

### 2. Estimation Methods ([estimation_methods.ipynb](https://github.com/workofart/ml-algorithms/blob/main/notebooks/estimation_methods.ipynb))
- Maximum Likelihood Estimation (MLE) implementation for Gaussian distributions
- Finding optimal parameters by minimizing negative log-likelihood

### 3. K-Nearest Neighbors ([k_nearest_neighbors.ipynb](https://github.com/workofart/ml-algorithms/blob/main/notebooks/k_nearest_neighbors.ipynb))
- Implementation of neighbor selection logic and majority voting mechanism using euclidean distance calculations
- Visualize the how each point is classified to its k-nearest neighbors

### 4. K-Means Clustering ([kmeans.ipynb](https://github.com/workofart/ml-algorithms/blob/main/notebooks/kmeans.ipynb))
- Implementation using Expectation-Maximization (EM) steps. K-mean is a specific case of the EM algorithm.
- Features cluster center initialization and convergence criteria
- Visualize loss and cluster assignment

### 5. Linear Regression ([linear_regression.ipynb](https://github.com/workofart/ml-algorithms/blob/main/notebooks/linear_regression.ipynb))
- Complete derivation of analytical solution for minimizing squared error
- Derivation of slope and intercept formula
- Visualize using diabetes dataset by performing feature-by-feature linear regression

### 6. Logistic Regression ([logistic_regression.ipynb](https://github.com/workofart/ml-algorithms/blob/main/notebooks/logistic_regression.ipynb))
- Complete mathematical derivation and implementation including:
  - Sigmoid function
  - Likelihood function
  - Loss function
  - Gradient calculations
- Visualize binary classification with simple gradient descent optimization using the derived gradients from above

### 7. Multi-layer Perceptron (MLP) ([multi_layer_perceptron.ipynb](https://github.com/workofart/ml-algorithms/blob/main/notebooks/multi_layer_perceptron.ipynb))
- Complete mathematical derivation and implementation including:
  - Forward pass computations
  - Loss functions (MSE and Binary Cross-Entropy)
  - Gradient calculations using chain rule
- Using the hand-built MLP to perform regression and classification tasks

### 8. Support Vector Machine (SVM) ([support_vector_machine.ipynb](https://github.com/workofart/ml-algorithms/blob/main/notebooks/support_vector_machine.ipynb))
- Complete mathematical derivation of:
  - Maximum margin classifier
  - Geometric margin
  - Hinge loss
  - Slack variables
- Examples with both hard and soft margin SVM
- Visualize binary classification using the SVM model constructed above