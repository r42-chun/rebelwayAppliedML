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

# ===================================
# --- 1. Generate starting data ---
# ===================================
# Generate 100 data points
n_samples = 100
np.random.seed(42)

X = np.random.uniform(low=-5, high=5, size=n_samples).reshape(-1, 1)

# Write an arbitrary polynomial function of order 4
y_true = 0.1+X**4 - 0.5*X**3 - 2*X**2 + X + 5
# Add some noise to the data
y = y_true.ravel() + np.random.normal(loc=0, scale=15, size=n_samples)

# ===================================
# --- 2. Define the search space ---
# ===================================
# We tell what the optimizer to find between 1 (linear) and 8 (high order)
search_space = [Integer(low=1, high=8, name='degree')]
print(f"search_space is {search_space} and type {type(search_space[0])}")

# ===================================
# --- 2. Define the objective function ---
# ===================================
# This is the function the optimizer will try to minimize.
# It takes a set of parameters (degree) and returns a score
# A lower score should mean a better model. We will use mean square error (MSE)

@use_named_args(search_space)
def objective(degree):
    """
    Trains and evaluates a polynomial regression model for a given degree

    :param
        degree (int): The polynomial degree to test

    :return:
        float: The average negative mean squared error from cross-validation
    """
    # Create the pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regr', LinearRegression())
    ])

    # Use 5-fold cross-validation to get a robust error estimate
    # 'neg_mean_squared_error' is used because skopt minimizes the function.
    # A smaller MSE is better, so a less negative score is better.
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

    # The optimizer will minimize this value. We return the mean of the scores
    avg_score = -np.mean(scores)

    print(f"Testing degree = {degree}... Average MSE: {avg_score:.2f}")

    return avg_score

# ===================================
# --- 4. Run Bayesian Optimization ---
# ===================================
# We use gp_minimize (Gaussian Process minimization) to run the search.
# n_calls: The total number of times we will call the objective function.
# random_state: For reproducibility of the optimization process.
print("Starting Bayesian Optimization...")
results = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=15, # Includes initial random points + exploration/exploitation steps
    n_initial_points=5, # How many random degrees to try before starting smart search
    random_state=42
)

# ===================================
# --- 5. Display the Results ---
# ===================================
print("\n--- Optimization Finished ---")
print(f"Optimal Polynomial Degree: {results.x[0]}")
print(f"Best Average MSE: {results.fun:.4f}")