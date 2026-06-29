"""
In this exercise, we are learning how to find the best order for polynomial features using Bayes Optimization
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# --- 1. Create Synthetic Data ---
# Let's create data where the true relationship is a 4th-degree polynomial.
# Our goal is to see if the Bayesian Optimization can discover that '4' is the best degree.

# Generate 100 data points
n_samples = 100
np.random.seed(42) # for reproducibility

# Feature X
X = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)

# Target y = 0.1*x^4 - 0.5*x^3 - 2*x^2 + x + 5 + noise
y_true = 0.1*X**4 - 0.5*X**3 - 2*X**2 + X + 5
# Add some random noise to make it a realistic problem
y = y_true.ravel() + np.random.normal(0, 15, n_samples)


# --- 2. Define the Search Space ---
# We tell the optimizer to search for an integer for the polynomial degree,
# for example, between 1 (linear) and 8 (highly complex).
search_space = [
    Integer(1, 8, name='degree')
]


# --- 3. Define the Objective Function ---
# This is the function the optimizer will try to MINIMIZE.
# It takes a set of parameters (in our case, just 'degree') and returns a score.
# A lower score should mean a better model. We will use Mean Squared Error (MSE).

# The '@use_named_args' decorator allows us to use parameter names directly
@use_named_args(search_space)
def objective(degree):
    """
    Trains and evaluates a polynomial regression model for a given degree.

    Args:
        degree (int): The polynomial degree to test.

    Returns:
        float: The average negative mean squared error from cross-validation.
               (We use negative MSE because scikit-optimize minimizes, and higher score is better).
    """
    # Create a pipeline that first creates polynomial features, then runs linear regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regr', LinearRegression())
    ])

    # Use 5-fold cross-validation to get a robust error estimate
    # 'neg_mean_squared_error' is used because skopt minimizes the function.
    # A smaller MSE is better, so a less negative score is better.
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

    # The optimizer will minimize this value. We return the mean of the scores.
    # A model that fits well will have an MSE close to 0.
    avg_score = -np.mean(scores)

    print(f"Testing degree={degree}... Average MSE: {avg_score:.2f}")

    return avg_score


# --- 4. Run Bayesian Optimization ---
# We use gp_minimize (Gaussian Process minimization) to run the search.
# n_calls: The total number of times we will call the objective function.
# random_state: For reproducibility of the optimization process.

print("Starting Bayesian Optimization...")
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=15, # Includes initial random points + exploration/exploitation steps
    n_initial_points=5, # How many random degrees to try before starting smart search
    random_state=42
)

# --- 5. Display the Results ---
print("\n--- Optimization Finished ---")
print(f"Optimal Polynomial Degree: {result.x[0]}")
print(f"Best Average MSE: {result.fun:.4f}")

