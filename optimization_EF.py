import numpy as np
import pandas as pd
from scipy.optimize import minimize

## Data Visualization
import matplotlib.pyplot as plt
from matplotlib import cm


# Define the response surface model
def response_surface(factors):
    coefficients = [-29.507520, 8.026565, 7.828037, -0.404694]
    # model for enrichment factor:
    #   y = b1*x1 + b2*x2 + b11*x1^2 + b22*x2^2
    # factor = [X1,X2]
    X1, X2, X3 = factors
    Y = (
        +coefficients[0] * X1
        + coefficients[1] * X2
        + coefficients[2] * X1**2
        + coefficients[3] * X2**2
    )
    return Y


# Define the objective function to be maximized
def objective_function(factors):
    # Ellipsoidal region constraint (adjust as needed)
    a, b = 0.89, 4.54  # Semi-axes lengths of the ellipsoid
    constraint = ((factors[0] - 1.25) / a) ** 2 + ((factors[1] - 9) / b) ** 2 - 1

    if constraint <= 0:
        return -response_surface(factors)  # We negate to maximize
    else:
        return 1e6  # A large penalty if the constraint is violated


def optimization(factor_bounds, initial_factors_guess):
    # Bounds for factors (adjust as needed) non codified
    factor_bounds = [
        (0.36, 2.14),
        (4.46, 13.54),
    ]

    # Initial guess for factors (adjust as needed)
    initial_factors_guess = [1.25, 9]

    # Perform the optimization
    result = minimize(
        objective_function,
        initial_factors_guess,
        bounds=factor_bounds,
        method="L-BFGS-B",  # You can choose another optimization method if preferred
    )

    # Extract the optimized factors
    optimal_factors = result.x

    # Evaluate the optimized response variable
    optimal_response = response_surface(optimal_factors)
    # Print the results
    print("Optimal Factors (X1, X2):", optimal_factors)
    print("Optimal Response:", optimal_response)

    return optimal_factors, optimal_response


def model_plot(
    exp_df,
    response,
    var_effects,
    fixed_effect,
    fixed_val=390,
):
    """This function plot a response surface model with 3 factor and one reponse.
    Owing to the nature of the design, the response surface is 3-Dimensions. In order
    to plot the surface, the factor x3 has a fixed value.

    Input:
        results_df: dataframe that contains the columns factor and response
        X: dataframe with the three effects
        y: dataframe with the response
        fixed_val: fixed point for visualization of surface"""

    print("This function create the surface plot".center(80, "="))

    grid_space = 10  # fixed

    # Filtering experimental point for fixed effect
    exp_data = exp_df[exp_df[fixed_effect] == fixed_val]
    print(exp_data)

    # Definign experimetnal points
    x1_exp = exp_data[var_effects[0]]  # x axis
    x2_exp = exp_data[var_effects[1]]  # y axis
    y_exp = exp_data[response]  # z axis

    x1_pred = np.linspace(min(x1_exp), max(x1_exp), grid_space)
    x2_pred = np.linspace(min(x2_exp), max(x2_exp), grid_space)

    # Create grid to predict response
    x1_pred, x2_pred = np.meshgrid(x1_pred, x2_pred)

    # Predict using response_surfce function
    y_pred = response_surface([x1_pred, x2_pred])  # predict the response

    # Plot model visualization
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot experimental points
    ax.plot(
        x1_exp,
        x2_exp,
        y_exp,
        color="k",
        zorder=15,
        linestyle="none",
        marker="o",
        alpha=0.5,
    )
    # Plot surface
    ax.plot_surface(
        x1_pred, x2_pred, y_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # Plot maximal point
    ax.plot(
        optimal_factors[0],
        optimal_factors[1],
        optimal_response,
        color="g",
        zorder=15,
        linestyle="none",
        marker="o",
        alpha=1,
    )

    # Set axes configuration
    ax.set_xlabel(x1_exp.name, fontsize=12)
    ax.set_ylabel(x2_exp.name, fontsize=12)
    ax.set_zlabel(y_exp.name, fontsize=12)
    ax.locator_params(nbins=4, axis="x")
    ax.locator_params(nbins=5, axis="x")

    plt.show()


def main():
    path = r"C:\Users\marco\python-projects\rsm_design\rsm_results.csv"

    effects = [
        "Extractant-dispersant ratio",
        "Sample volume",
        "Extractant mixture volume",
    ]
    response = ["Enrichment factor"]

    print("Importing data".center(80, "="))

    data = pd.read_csv(path, sep=";", decimal=",")

    # Data frame with columns = [effects,response]
    exp_df = data[effects + response]
    print("\n", exp_df)

    ## Maximize the surface
    optimal_factors, optimal_response = optimization()

    ############### CREATE A FUNCTION WITH THIS ################################

    fixed_effect = "Extractant mixture volume"
    var_effects = ["Extractant-dispersant ratio", "Sample volume"]
    response = "Enrichment factor"


main()
