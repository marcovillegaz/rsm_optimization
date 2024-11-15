"""This scripts contains function to to be utilized in the
Response Surface Methodology"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from utils.plots import residual_plot, coeff_plot, prediction_plot


# Change to import_experiment
def extract_data(file_path, sheet_name, factors, response):
    """This function extract the factors and the response in the experimental
    results matrix allocated ina spreadsheet.
    input:
        file_path:
        sheet_name:
        factors:
        response:
    return:
        df:"""

    data_df = pd.read_excel(file_path, sheet_name, skiprows=1)
    data_df = data_df.drop([0, 1])  # drop first two rows (useless)
    data_df = data_df[factors + [response]]  # Rearrangement of data
    data_df = data_df.reset_index(drop=True)  # Reset indexes
    data_df = data_df.apply(pd.to_numeric, errors="coerce")  # Convert to numeric type

    if response == "Recovery":  # Recovery in % if it needed
        data_df["Recovery"] = data_df["Recovery"] * 100

    return data_df


def import_coeff(coefficients_path):
    """Import model coefficient from .csv file

    params:
        coefficients_path (string): path where the coefficient are allocated

    return:
        model_coeff (list): List with model coefficients
    """

    coeff_df = pd.read_csv(coefficients_path, index_col=None, header=None)
    coefficients = coeff_df.iloc[:, 1].to_list()

    return coefficients


def model_fitting(data_df, factors, response, del_coeff, results_folder):
    """
    Fit the experimental results of the central composite design to a linear
    quadratic model

    params:

    data_df (pd.DataFrame): experimental matrix that contains the results of
    the CCD ordered by columns,
    factor (list): List the factor studied in the CCD (X1,x2,x3,),
    response (string): Response studied in the CCD (y),
    del_coeff (list): list with the coefficient you want to delet in the model
    results_folder (string): Folder where the result will be saved.
    """
    print(f"Fitting model to {response} response.")

    # List with coefficents name
    coeff = ["b0", "b1", "b2", "b3", "b11", "b12", "b13", "b22", "b23", "b33"]

    # Redefine data
    X = data_df[factors]
    y = data_df[response]

    # Define the model as quadratic
    polynomial_features = PolynomialFeatures(degree=2)
    # Apply cuadratic model to X
    X = polynomial_features.fit_transform(X)
    # Define X as dataframe
    X = pd.DataFrame(data=X, columns=coeff)
    # Deleting terms of the model (for best fit)
    X = X.drop(del_coeff, axis=1)

    # Fitting the model and show summary
    model = sm.OLS(y, X).fit()

    # MODEL PARAMETERS
    coeff_series = pd.Series(0.0, index=coeff)  # Series with zeros
    params = model.params  # Extract model coefficients
    coeff_series[params.index] = params  # Update the coefficients Series

    # Saving coefficient in csv file
    print("\tSaving model coefficient.")
    coeff_series.to_csv(
        os.path.join(results_folder, f"{response}_model_params.csv"), header=False
    )

    # Save model stats in results folder
    print(f"\tSaving model statistics.")
    with open(os.path.join(results_folder, f"{response}_model_stats.txt"), "w") as fh:
        fh.write(model.summary().as_text())

    ## PLOT REGRESSION ANAYLISIS
    print("CREATING PLOT TO ANALYSE COEFFICIENTS:")
    # Coefficient plot
    coeff_plot(model, response, normalize=None, whiskers="ci", folder=results_folder)
    # Residual plot
    residual_plot(model, X, y, response, normalize=True, folder=results_folder)
    # Predicted vs experimental plot
    prediction_plot(model, X, y, response, folder=results_folder)


def response_surface(X_mesh, coefficients):
    """Apply response surface equation

    params:
    X_mesh (np.array): mesh obtained from
    Coefficients (np.array)"""
    # model for enrichment factor:
    #   y = b1*x1 + b2*x2 + b11*x1^2 + b22*x2^2
    # factor = [X1,X2]
    X1, X2, X3 = X_mesh
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


# Define the o
def objective_function(factors, coefficients, length, center):
    """Objective function to be maximized. In Response Surface Methodology
    with central composite design (CCD), the model is valid only inside
    the area circumscribed by the experimental points, so the optimization
    must be applied inside this region.

    params:
        factors: List of factors (variables) being optimized.
        coefficients (list): Coefficients for the response surface equation.
        length (list): Semi-axes lengths of the ellipsoid.
        center (list): Center coordinates of the ellipsoid.
    return:
        value: Objective function value.
    """

    # Ellipsoidal constraint parameters
    a, b, c = length  # Semi-axes lengths of the ellipsoid
    ellipsoid_constraint = (
        ((factors[0] - center[0]) / a) ** 2
        + ((factors[1] - center[1]) / b) ** 2
        + ((factors[2] - center[2]) / c) ** 2
        - 1
    )

    if ellipsoid_constraint <= 0:
        value = -response_surface(factors, coefficients)  # We negate to maximize
    else:
        value = 1e6  # A large penalty if the constraint is violated
    return value


def rsm_optimization(coefficients, exp_data):
    """this function optimize the response surface and return the optimal factors
    and the corresponding response"""

    print("rsm_optimization()")
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
    for i, opt_fact in enumerate(optimal_factors):
        print("\tX{}:{}".format(i + 1, opt_fact))
    print("\ty:{}".format(optimal_response))

    return optimal_factors, optimal_response
