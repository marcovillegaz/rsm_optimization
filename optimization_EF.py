import numpy as np
import pandas as pd
from scipy.optimize import minimize

## Data Visualization
import matplotlib.pyplot as plt
from matplotlib import cm


# Define the response surface model
def response_surface(factors, coefficients):
    # model for enrichment factor:
    #   y = b1*x1 + b2*x2 + b11*x1^2 + b22*x2^2
    # factor = [X1,X2]
    X1, X2, X3 = factors
    Y = (
        coefficients[0]
        + coefficients[1] * X1
        + coefficients[2] * X2
        + coefficients[3] * X3
        + coefficients[4] * X1**2
        + coefficients[5] * X1 * X2
        + coefficients[6] * X1 * X3
        + coefficients[7] * X2**2
        + coefficients[8] * X2 * X3
        + coefficients[9] * X3**2
    )
    return Y


# Define the objective function to be maximized
def objective_function(factors, coefficients, length, center):
    # Ellipsoidal constraint parameters
    a, b, c = length  # Semi-axes lengths of the ellipsoid
    ellipsoid_constraint = (
        ((factors[0] - center[0]) / a) ** 2
        + ((factors[1] - center[1]) / b) ** 2
        + ((factors[2] - center[2]) / c) ** 2
        - 1
    )

    if ellipsoid_constraint <= 0:
        return -response_surface(factors, coefficients)  # We negate to maximize
    else:
        return 1e6  # A large penalty if the constraint is violated


def surface_plot(
    exp_df,
    coefficients,
    response,
    var_effects,
    fixed_effect,
    optimal_factors=[],
    optimal_response=[],
    fixed_val=390,
    path=None,
):
    """This function plot a response surface model with 3 factor and one reponse.
    Owing to the nature of the design, the response surface is 3-Dimensions. In order
    to plot the surface, the factor x3 has a fixed value.

    Input:
        results_df: dataframe that contains the columns factor and response
        X: dataframe with the three effects
        y: dataframe with the response
        fixed_val: fixed point for visualization of surface"""

    # print("This function create the surface plot".center(80, "="))

    grid_space = 500  # fixed

    # Filtering experimental point for fixed effect
    exp_data = exp_df[exp_df[fixed_effect] == fixed_val]

    # Definign experimetnal points
    x1_exp = exp_data[var_effects[0]]  # x axis
    x2_exp = exp_data[var_effects[1]]  # y axis
    x3_exp = exp_data[fixed_effect]  # Fixed effect
    y_exp = exp_data[response]  # z axis

    x1_pred = np.linspace(min(x1_exp), max(x1_exp), grid_space)
    x2_pred = np.linspace(min(x2_exp), max(x2_exp), grid_space)
    x3_pred = np.ones(grid_space) * fixed_val

    # Create grid to predict response
    x1_pred, x2_pred, x3_pred = np.meshgrid(x1_pred, x2_pred, x3_pred)

    # Predict using response_surface function
    y_pred = response_surface(
        [x1_pred, x2_pred, x3_pred],
        coefficients,
    )

    # Plot model visualization
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(elev=30, azim=45, roll=0)
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
        x1_pred[:, :, 0],
        x2_pred[:, :, 0],
        y_pred[:, :, 0],
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
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
    )

    # Set axes configuration
    ax.set_xlabel(x1_exp.name, fontsize=12)
    ax.set_ylabel(x2_exp.name, fontsize=12)
    ax.set_zlabel(y_exp.name, fontsize=12)
    ax.locator_params(nbins=4, axis="x")
    ax.locator_params(nbins=5, axis="x")

    if path != "None":
        path = path + r"\surface_plot.png"
        print("Saving figure in:", path)
        plt.savefig(path)
    else:
        plt.show()


def main(data_path, coeff_file, effects, response, save_path):
    ## READING DATA
    print("\nImporting experimental points")
    data = pd.read_csv(data_path, sep=";", decimal=",")
    exp_df = data[effects + response]  # Data frame with columns = [effects,response]
    print(exp_df)

    print("\nImporting model coefficientes as list.")
    coeff_df = pd.read_csv(coeff_file, index_col=None, header=None)
    print(coeff_df)
    coefficients = coeff_df.iloc[:, 1].to_list()

    ## OPTIMIZATION OF THE SURFACE
    # Semi-axes lengths of the ellipsoid (adjust as needed)
    # This is the diference beetween the center point and the axial point (alpha)
    # of the central composite design circumbcribed (CCC)

    length = [0.89, 4.54, 16.82]
    initial_factors_guess = [1.25, 9, 390]
    centre_point = [1.25, 9, 390]
    factor_bounds = [
        (0.36, 2.14),
        (4.46, 13.54),
        (373.18, 406.82),
    ]

    # Perform the optimization
    result = minimize(
        objective_function,
        initial_factors_guess,
        # Pass coefficients as an additional argument
        args=(coefficients, length, centre_point),
        bounds=factor_bounds,
        method="L-BFGS-B",  # You can choose another optimization method if preferred
    )

    # Extract the optimized factors
    optimal_factors = result.x
    # Evaluate the optimized response variable
    optimal_response = response_surface(optimal_factors, coefficients)

    # Print the results
    print("RESULTS".center(80, "="))
    print("Optimal factors")
    for i, opt_fact in enumerate(optimal_factors):
        print("\tX{}:{}".format(i + 1, opt_fact))
    print("Optimal Response")
    print("\ty:{}".format(optimal_response))

    ############### CREATE A FUNCTION WITH THIS ################################
    fixed_effect = "Extractant mixture volume"
    var_effects = ["Extractant-dispersant ratio", "Sample volume"]
    response = "Enrichment factor"

    surface_plot(
        exp_df,
        coefficients,
        response,
        var_effects,
        fixed_effect,
        optimal_factors,
        optimal_response,
        fixed_val=390,
        path=save_path,
    )


###############################################################################
data_path = "rsm_results.csv"
coeff_file = "model_coeff_EF.csv"
save_path = "EF"

effects = [
    "Extractant-dispersant ratio",
    "Sample volume",
    "Extractant mixture volume",
]
response = ["Enrichment factor"]

main(data_path, coeff_file, effects, response, save_path)
