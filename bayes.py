"""
We compute the empirical risk for the following estimators:
f1(x) = 2 if x = 0, 0 otherwise
f*(x) = 0
"""

import numpy as np


def main() -> None:

    n_samples = int(1e6)

    X = np.random.multinomial(1, [0.1, 0.2, 0.3, 0.4], size=n_samples)

    # Copy X for parameters
    multinomial_parameters = np.zeros((n_samples, 3))  # Assuming there are 3 categories
    # Set parameters for each category using boolean indexing
    multinomial_parameters[X[:, 0] == 1] = [0.7, 0.2, 0.1]
    multinomial_parameters[X[:, 1] == 1] = [0.8, 0.15, 0.05]
    multinomial_parameters[(X[:, 2] == 1) | (X[:, 3] == 1)] = [0.9, 0.08, 0.02]

    # Generate new multinomial distribution based on updated parameters
    y = np.array([np.random.multinomial(1, p) for p in multinomial_parameters])

    y_pred_bayes = np.zeros((n_samples, 3))
    y_pred_bayes[:, 0] = 1

    # print(f"Bayes estimator: {y_pred_bayes}")
    y_pred_f1 = np.zeros((n_samples, 3))
    y_pred_f1[X[:, 0] == 1] = [0, 0, 1]
    y_pred_f1[(X[:, 1] == 1) | (X[:, 2] == 1) | (X[:, 3] == 1)] = [1, 0, 0]

    emperical_risk_bayes = np.mean(np.any(y != y_pred_bayes, axis=1))
    emperical_risk_f1 = np.mean(np.any(y != y_pred_f1, axis=1))

    print(f"Empirical risk for Bayes estimator: {emperical_risk_bayes}")
    print(f"Empirical risk for f1 estimator: {emperical_risk_f1}")


if __name__ == "__main__":
    main()
