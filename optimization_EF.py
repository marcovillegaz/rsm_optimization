import time
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
    exp_data,
    coefficients,
    response,
    effects,
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

    print("This function create the surface plot".center(80, "="))
    grid_space = 30  # fixed

    # Accessing values without extracting
    # fixed_effect = next(iter(fixed_effect_dic))
    # fixed_val = float(fixed_effect_dic[fixed_effect])
    # print("Fixed effect: ", fixed_effect, " at ", fixed_val)

    # Filtering experimental point for fixed effect
    # exp_data = exp_data[exp_data[fixed_effect] == fixed_val]
    # print("This are the experimetnal points to plot\n", exp_data)

    # Definign experimetnal points
    print("effects: ", effects)
    print("response: ", response)
    print(exp_data[effects[2]])
    x1_exp = exp_data[effects[0]]  # x1
    x2_exp = exp_data[effects[1]]  # x2
    x3_exp = exp_data[effects[2]]  # x3
    y_exp = exp_data[response]  # y response

    # Define surface boundary (change linspace to step array)
    x1_pred = np.linspace(min(x1_exp), max(x1_exp), grid_space)
    x2_pred = np.linspace(min(x2_exp), max(x2_exp), grid_space)
    x3_pred = np.linspace(min(x3_exp), max(x3_exp), grid_space)
    print(x1_exp)

    # Create grid to predict response
    x1_pred, x2_pred, x3_pred = np.meshgrid(x1_pred, x2_pred, x3_pred)

    # Predict using response_surface function
    y_pred = response_surface(
        [x1_pred, x2_pred, x3_pred],
        coefficients,
    )
    print(y_pred)

    # FIXED VAR AND VALUE
    # Supossing fixed var is x1 at the i element.
    fixed_index = 15

    # Plot model visualization
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(elev=30, azim=45, roll=0)

    # Plot surface
    surface_plot = ax.plot_surface(
        x1_pred[:, :, fixed_index],
        x2_pred[:, :, fixed_index],
        y_pred[:, :, fixed_index],
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )

    # Loop
    for i in range(grid_space):
        # creating new Y values
        new_x = x1_pred[:, :, i]
        new_y = x2_pred[:, :, i]
        new_z = y_pred[:, :, i]

        # updating data values
        surface_plot.set_xdata(new_x)
        surface_plot.set_ydata(new_y)
        surface_plot.set_zdata(new_z)

        # drawing updated values
        fig.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        fig.canvas.flush_events()

        time.sleep(0.1)

    # Plot maximal point
    # ax.plot(
    #     optimal_factors[0],
    #     optimal_factors[1],
    #     optimal_response,
    #     color="g",
    #     zorder=15,
    #    linestyle="none",
    #    marker="o",
    # )

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


def main(data_path, coeff_file, response):
    ## READING DATA
    # print("\nImporting experimental points")
    # data = pd.read_csv(data_path, sep=";", decimal=",")
    # exp_df = data[effects + response]  # Data frame with columns = [effects,response]
    # print(exp_df)

    print("\nImporting model coefficientes as list.")
    coeff_df = pd.read_csv(coeff_file, index_col=None, header=None)
    print(coeff_df)
    coefficients = coeff_df.iloc[:, 1].to_list()

    ## OPTIMIZATION OF THE SURFACE
    # Semi-axes lengths of the ellipsoid (adjust as needed)
    # This is the diference beetween the center point and the axial point (alpha)
    # of the central composite design circumbcribed (CCC)

    length = [0.89, 4.54, 185]
    initial_factors_guess = [1.25, 9, 300]
    centre_point = [1.25, 9, 300]
    factor_bounds = [
        (0.36, 2.14),
        (4.46, 13.54),
        (115, 485),
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
    # Import experimental data
    exp_data = pd.read_csv(data_path, sep=";", decimal=",")  # Import csv file
    exp_data = exp_data.dropna()  # Drop NaN values
    exp_data["Recovery"] = exp_data["Recovery"] * 100
    print("\n This is the raw_data:\n", exp_data)
    print(exp_data.columns)


###############################################################################
data_path = r"C:\Users\marco\python-projects\rsm_design\RSM_3.csv"
coeff_file_REC = (
    r"C:\Users\marco\python-projects\rsm_design\Result_REC - run3\model_params.csv"
)

coeff_file_EF = (
    r"C:\Users\marco\python-projects\rsm_design\Result_EF - run3\model_params.csv"
)


save_path = "REC optimized"

effects = [
    "Extractant-Dispersant Ratio",
    "Sample volume",
    "Extraction mixture volume",
]

print("\n" + "RECOVERY OPTIMIZATION".center(80, "="))
main(
    data_path,
    coeff_file_REC,
    response="Recovery",
)

""" print("\n" + "EF OPTIMIZATION".center(80, "="))
main(
    data_path,
    coeff_file_EF,
    response="Enrichment factor",
) """
