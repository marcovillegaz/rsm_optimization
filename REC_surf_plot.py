"""This scripts compute the optimum response for REC_model_simplified
and then create a surface plot with the experimental points"""

from utils.rsm import *


def rsm_plot(
    coefficients,
    experimental_df,
    response,
    factors,
    fixed_factor,
    scatter=None,
    optimum=None,
    save_path=None,
):
    """Create a surface plot of the simplified model. This function is only for
    the results obtained in this studie. When working with multivariable functions,
    to create a surface plot, two independent variable are varied while the
    others remains constant. This makes difficult to create a unified fucntion
    that generate the surface plots when are more than 2 independent variables.

    params:
    coefficientes (list): model coefficients
    experimental_df (pd.DataFrame): experimental data (factors + response)
    response (string): response to be plotted
    factors (list): list of factors
    fixed_factor (string): factor that is mantain constant
    scatter (bool): plot the experimental points
    """
    grid_space = 100  # CONSTANT

    # Find indexes of variable factors in experimetnal dataFrame
    idx_of_var = [index for index, value in enumerate(factors) if value != fixed_factor]
    print("Variable factors:", experimental_df.columns[idx_of_var])

    # PREDICT SURFACE RESPONSE SURFACE FROM MODEL
    # Define surface boundary
    X_pred = np.ones((3, grid_space))  # Each row correspond to a effect
    for i, factor in enumerate(factors):
        # The fixed effect is taken in the central value
        if factor == fixed_factor:
            # The central value is repited n = 5 times
            centre_point = experimental_df[factor].mode().values[0]
            X_pred[i, :] = X_pred[i, :] * centre_point

        else:
            lb = min(experimental_df[factor])  # lower bound
            ub = max(experimental_df[factor])  # Upper bound
            dif = max(experimental_df[factor]) - min(experimental_df[factor])

            # The surface is extended 25% of the experimental domain
            percent = 0.25
            X_pred[i, :] = np.linspace(
                lb - percent * dif,
                ub + percent * dif,
                grid_space,
            )

    # print(X_pred.transpose(), "\n")

    # Create grid to predict response
    x1_pred, x2_pred, x3_pred = np.meshgrid(X_pred[0, :], X_pred[1, :], X_pred[2, :])
    mesh = [x1_pred, x2_pred, x3_pred]
    # print(mesh)

    # Predict using response_surface function
    y_pred = response_surface(mesh, coefficients)
    # print(y_pred)

    # Create an empty figure and 3D axis
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax = plt.axes(projection="3d", computed_zorder=False)
    plt.subplots_adjust(left=0.0, right=1, bottom=0.0, top=1.1)

    surf = ax.plot_surface(
        mesh[idx_of_var[0]][:, :, 0],
        mesh[idx_of_var[1]][:, 0, :],
        y_pred[:, 0, :],
        cmap="jet",
        alpha=0.6,
        antialiased=True,
        # rcount=100,
        # ccount=100,
        zorder=5,
    )

    # Plot experimental points
    if scatter == True:
        x = experimental_df.iloc[:, 0:3].to_numpy()
        y_pred = response_surface([x[:, 0], x[:, 1], x[:, 2]], coefficients)
        resd = experimental_df[response].to_numpy() - y_pred
        print("Residuals:\n", resd)

        up_idx = np.where(resd < 0)[0]
        low_idx = np.where(resd > 0)[0]

        # Experimental points above the surface
        ax.scatter(
            xs=experimental_df.iloc[up_idx, idx_of_var[0]],
            ys=experimental_df.iloc[up_idx, idx_of_var[1]],
            zs=experimental_df[response].to_numpy()[up_idx],
            marker=".",
            c="red",
            zorder=2,
            depthshade=False,
        )

        # Experimental points below the surface
        ax.scatter(
            xs=experimental_df.iloc[low_idx, idx_of_var[0]],
            ys=experimental_df.iloc[low_idx, idx_of_var[1]],
            zs=experimental_df[response].to_numpy()[low_idx],
            marker=".",
            c="red",
            zorder=10,
            depthshade=False,
        )

    if optimum != None:
        ax.scatter(
            xs=optimum[0][idx_of_var[0]],
            ys=optimum[0][idx_of_var[1]],
            zs=optimum[1] * 1.05,
            marker="*",
            c="green",
            zorder=1,
        )

    # Set view and label
    ax.view_init(azim=-45, elev=20)  # Set view
    ax.set_xlabel(factors[idx_of_var[0]] + " (mL)")
    ax.set_ylabel(factors[idx_of_var[1]] + " ($\mu$L)")
    ax.set_zlabel(response)

    if save_path != None:
        file_name = f'{response.replace(" ", "")}_surface.png'
        print(f"Saving {file_name}")
        plt.savefig(os.path.join(save_path, file_name), dpi=800)
    else:
        plt.show()


########################################################################

# Extract experimental data for Enrichment FActor
experimental_data = extract_data(
    file_path=r"experimental_results.xlsx",
    sheet_name="RSM_3",
    factors=[
        "Extractant-Dispersant Ratio",
        "Sample volume",
        "Extraction mixture volume",
    ],
    response="Recovery",
)
# print(experimental_data)

# Import model coefficients
model_coefficients = import_coeff(r"rsm_results_simplified\Recovery_model_params.csv")
# print(model_coefficients)

# Response surface optimization
optimal_factors, optimal_response = rsm_optimization(
    model_coefficients,
    experimental_data,
)

# Create surface plot
rsm_plot(
    coefficients=model_coefficients,
    experimental_df=experimental_data,
    response="Recovery",
    factors=[
        "Extractant-Dispersant Ratio",
        "Sample volume",
        "Extraction mixture volume",
    ],
    fixed_factor="Extractant-Dispersant Ratio",
    scatter=True,
    optimum=[optimal_factors, optimal_response],
    save_path="rsm_results_simplified",
)
