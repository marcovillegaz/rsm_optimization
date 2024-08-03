"""This scripts contains function to to be utilized in the
Response Surface Methodology"""

import os
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures
from utils.plots import residual_plot, coeff_plot, prediction_plot


def extract_data(file_path, sheet_name, factors, response):
    """This function extract the factors and the response in the experimental
    results matrix
    input:
        file_path:
        sheet_name:
        factors:
        response:
    return:
        df:"""

    data_df = pd.read_excel(file_path, sheet_name, skiprows=1)
    data_df = data_df.drop([0, 1])  # drop first two rows (useless)
    print(data_df.columns)
    data_df = data_df[factors + [response]]  # Rearrangement of data
    data_df = data_df.reset_index(drop=True)  # Reset indexes
    data_df = data_df.apply(pd.to_numeric, errors="coerce")  # Convert to numeric type

    if response == "Recovery":  # Recovery in % if it needed
        data_df["Recovery"] = data_df["Recovery"] * 100

    return data_df


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
