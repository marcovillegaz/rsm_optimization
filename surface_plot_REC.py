import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# from mpl_toolkits.mplot3d import Axes3D


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


def import_coeff(coeff_path):
    """This function import the coefficient of the model, saved as csv file, and
    return a string for further processing"""

    print("\nImporting model coefficientes as list.")
    coeff_df = pd.read_csv(coeff_path, index_col=None, header=None)
    coefficients = coeff_df.iloc[:, 1].to_list()
    print(coeff_df)

    return coefficients


def import_experiments(experimental_path, effects, response):
    print("\nImporting experimental results for: ", response)
    data_df = pd.read_excel(experimental_path, sheet_name="RSM_3", skiprows=1)
    data_df = data_df.drop([0, 1])  # drop first two rows (useless)
    data_df = data_df[effects + response]  # Rearrangement of data
    data_df = data_df.reset_index(drop=True)  # Reset indexes
    data_df = data_df.apply(pd.to_numeric, errors="coerce")  # Convert to numeric type
    data_df["Recovery"] = data_df["Recovery"] * 100
    print(data_df)

    return data_df


def rsm_optimization(coefficients, exp_data):
    """this function optimize the response surface and return the optimal factors
    and the corresponding response"""

    centre_point = exp_data.iloc[:, :3].mode().values.tolist()[0]
    max_values = exp_data.iloc[:, :3].max().values.tolist()
    length = (np.array(max_values) - np.array(centre_point)).tolist()
    initial_factors_guess = centre_point
    factor_bounds = [
        (min(exp_data.iloc[:, 0]), max(exp_data.iloc[:, 0])),
        (min(exp_data.iloc[:, 1]), max(exp_data.iloc[:, 1])),
        (min(exp_data.iloc[:, 2]), max(exp_data.iloc[:, 2])),
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

    return optimal_factors, optimal_response


def rsm_plot(
    coefficients,
    exp_data,
    response,
    effects,
    fixed_effect,
    plot_type,
    scatter=None,
    optimum=None,
    path=None,
):
    grid_space = 100  # CONSTANT

    # Definign experimetnal points
    print("effects: ", effects)
    print("fixed effect: ", fixed_effect)
    print("response: ", response)

    # Find indexes of variable effects
    idx_of_var = [index for index, value in enumerate(effects) if value != fixed_effect]
    print(idx_of_var)

    # PREDICT SURFACE RESPONSE SURFACE FROM MODEL
    # Define surface boundary
    X_pred = np.ones((3, grid_space))  # Eeach row correspond to a effect
    for i, effect in enumerate(effects):
        # The fixed effect is taken in the central value
        if effect == fixed_effect:
            centre_point = exp_data[effect].mode().values[0]  # central value
            X_pred[i, :] = X_pred[i, :] * centre_point

        else:
            lb = min(exp_data[effect])  # lower bound
            ub = max(exp_data[effect])  # Upper bound
            dif = max(exp_data[effect]) - min(exp_data[effect])

            X_pred[i, :] = np.linspace(lb - 0.25 * dif, ub + 0.25 * dif, grid_space)

    print(X_pred.transpose(), "\n")

    # Create grid to predict response
    x1_pred, x2_pred, x3_pred = np.meshgrid(X_pred[0, :], X_pred[1, :], X_pred[2, :])
    mesh = [x1_pred, x2_pred, x3_pred]
    # print(mesh)

    # Predict using response_surface function
    y_pred = response_surface(mesh, coefficients)
    # print(y_pred)

    # PLOT SURFACE OR CONTOUR
    if plot_type == "contour":
        # Create a figure and axes object
        fig, ax = plt.subplots(figsize=(6, 5))
        # Plot the contour plot using the contour function
        contour = ax.contourf(
            mesh[idx_of_var[0]][0, :, :],
            mesh[idx_of_var[1]][0, :, :],
            y_pred[0, :, :],
            levels=20,
            cmap="hot",
            alpha=0.9,
            antialiased=False,
        )

        # Plot experimental points
        if scatter == "True":
            ax.scatter(
                x=exp_data.iloc[:, idx_of_var[0]],
                y=exp_data.iloc[:, idx_of_var[1]],
                marker=".",
                c="blue",
            )

        if optimum != "None":
            ax.scatter(
                x=optimum[0][idx_of_var[0]],
                y=optimum[0][idx_of_var[1]],
                marker="*",
                c="green",
            )

        # Set labels and title
        fig.colorbar(contour, shrink=1, aspect=10)  # Add a color bar
        ax.set_xlabel(effects[idx_of_var[0]] + " (mL)")
        ax.set_ylabel(effects[idx_of_var[1]] + " ($\mu$L)")

    elif plot_type == "surface":
        # Create an empty figure and 3D axis
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax = plt.axes(projection="3d", computed_zorder=False)
        plt.subplots_adjust(left=0.0, right=1, bottom=0.0, top=1.1)

        surf = ax.plot_surface(
            mesh[idx_of_var[0]][:, :, 0],
            mesh[idx_of_var[1]][:, 0, :],
            y_pred[:, 0, :],
            cmap="coolwarm",
            alpha=0.6,
            antialiased=True,
            # rcount=100,
            # ccount=100,
            zorder=5,
        )

        # Plot experimental points
        if scatter == "True":
            x = exp_data.iloc[:, 0:3].to_numpy()
            y_pred = response_surface([x[:, 0], x[:, 1], x[:, 2]], coefficients)
            resd = exp_data[response].to_numpy() - y_pred
            print("REsiduals:\n", resd)

            up_idx = np.where(resd < 0)[0]
            low_idx = np.where(resd > 0)[0]

            ax.scatter(
                xs=exp_data.iloc[up_idx, idx_of_var[0]],
                ys=exp_data.iloc[up_idx, idx_of_var[1]],
                zs=exp_data[response].to_numpy()[up_idx],
                marker=".",
                c="red",
                zorder=2,
                depthshade=False,
            )

            ax.scatter(
                xs=exp_data.iloc[low_idx, idx_of_var[0]],
                ys=exp_data.iloc[low_idx, idx_of_var[1]],
                zs=exp_data[response].to_numpy()[low_idx],
                marker=".",
                c="red",
                zorder=10,
                depthshade=False,
            )

        if optimum != "None":
            ax.scatter(
                xs=optimum[0][idx_of_var[0]],
                ys=optimum[0][idx_of_var[1]],
                zs=optimum[1] * 1.05,
                marker="*",
                c="green",
                zorder=1,
            )

        ax.view_init(azim=-45, elev=20)  # Set view
        # fig.colorbar(surf, shrink=0.5, aspect=10, location="left")  # Add a color bar

        # Set labels and title
        ax.set_xlabel(effects[idx_of_var[0]] + " (mL)")
        ax.set_ylabel(effects[idx_of_var[1]] + " ($\mu$L)")
        ax.set_zlabel(response + " (%)")

    if path != None:
        path = path + r"\RSM_REC.png"
        print("Saving figure in:", path)
        plt.savefig(path, dpi=800)
    else:
        plt.show()


################################################################################
ef_coeff_path = r"Final results - REC\model_params.csv"
experimental_path = r"C:\Users\marco\OneDrive - usach.cl\DLLME of PCB77 employing designed DES\Hojas de calculo\Experimental.xlsx"

exp_data = import_experiments(
    experimental_path,
    effects=[
        "Extractant-Dispersant Ratio",
        "Sample volume",
        "Extraction mixture volume",
    ],
    response=["Enrichment factor", "Recovery"],
)

coefficients = import_coeff(ef_coeff_path)
optimal_factors, optimal_response = rsm_optimization(coefficients, exp_data)

rsm_plot(
    coefficients=coefficients,
    exp_data=exp_data,
    response="Recovery",
    effects=[
        "Extractant-Dispersant Ratio",
        "Sample volume",
        "Extraction mixture volume",
    ],
    fixed_effect="Extractant-Dispersant Ratio",
    plot_type="surface",
    scatter="True",
    optimum=[optimal_factors, optimal_response],
    path=r"C:\Users\marco\python-projects\rsm_design\surface_plots",
)
