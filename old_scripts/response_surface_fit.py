"""
Title: "Response surface methodology for three factors and one response"
Author: Marco Villegas
Date: 10/10/2013
Github: https://github.com/Marcodelflow

Description:
This scrips fit a cuadratic multilinear model in order to generate a repsonse 
surface for further optimization. The code used the experimetnal results of a 
central composite design (CCD) which data is allocated ina csv file (see
rsm_results.csv).

The code is only valid for three effects and one response. You can adapt to code 
modifiying the coeff list in main() and adding more effects """

# lIBRARIES
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from matplotlib import pyplot as plt


def residual_plot(model, X, y_exp, normalize="None", path="None"):
    """This function create a residual plot to analyses their distribution

    Input:
        model: object that contains all the information of the fitted model
            *see statsmodel OLS.
        X: dataframe with the polinomial features of skilearn
        y_exp: experimental response.
        normalize: True if yoU want a normalized plot
        path: folder where the plot is going to be stored
    """
    # PROCESSING DATA
    y_exp = np.reshape(y_exp, (len(y_exp), 1))  # Experimental response
    y_pred = np.reshape(model.predict(X), (len(y_exp), 1))  # Predicted response
    resd = y_exp - y_pred  # Residual

    # Calculate MSE
    mse = mean_squared_error(y_exp, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")

    # Calculate RMSE
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    ## Normalized residual statement
    if normalize == "True":
        resd = resd / np.std(resd)
        ylabel = "Normalized residuals"
    elif normalize == "None":
        ylabel = "Residuals"

    # MAKING PLOT
    fig = plt.figure(figsize=(10, 6))  # Create figure
    fig.suptitle("Residual plot", fontsize=16)
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 1))  # Grid spec

    ## Residual plot
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(y_pred, resd)
    ax.axhline(y=0)
    ax.set_xlabel("y predicted")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    ## Normal distribution with histogram
    N = len(y_exp)  # number measurement
    num_bins = 10  # numebr of bars in histograms
    y = np.linspace(resd.min(), resd.max(), 1000)
    bin_width = (resd.max() - resd.min()) / num_bins

    ax_marg_y = fig.add_subplot(gs[0, 1])
    ax_marg_y.hist(x=resd, bins=num_bins, orientation="horizontal")
    ax_marg_y.plot(norm.pdf(y) * N * bin_width, y)

    # SAVE OR SHOW
    if path != "None":
        path = path + r"\residual_plot.png"
        print("Saving figure in:", path)
        plt.savefig(path)
    else:
        plt.show()


def coef_plot(model, normalize="None", whiskers="ci", path="None"):
    """This function create a coefficient plot to show the confidence interval
     of the model cofficients. If the whiskers cut the horizontal line,
     the coefficient is inappropriate.

    Input:
        model: object that contains all the information of the fitted model
            *see statsmodel OLS.
        normalize: True if yoU want a normalized plot
        whiskers:
            "ci" for cofidence intervals
            "stderr" for standar error
        path: folder where the plot is going to be stored
    """
    # Preprocessing
    # Get the coefficients and their confidence intervals
    coefficients = model.params
    coeff_names = model.params.index

    if normalize == "True":
        error = model.bse
        coefficients = coefficients / error
        title = "Normalized coefficient Plot with error bars"

    elif normalize == "None":
        # Bar plot with coefficient intervals
        if whiskers == "ci" and normalize == "None":
            ci = model.conf_int().T
            ci = ci.to_numpy()
            ci_low = ci[0, :]
            ci_high = ci[1, :]
            error = [coefficients - ci_low, ci_high - coefficients]
            title = "Coefficient Plot with confidence intervals"

        # Bar plot with coefficient intervals
        elif whiskers == "stderr" and normalize == "None":
            error = model.bse
            title = "Coefficient Plot with error bars"

    # Create a coefficient plot with error bars using Axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Error bars with confidence intervals
    ax.errorbar(
        range(len(coefficients)),
        coefficients,
        yerr=error,
        fmt="s",
        capsize=5,
    )
    # Horizontal line at zero
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # Customize x-axis ticks and labels (replace with your variable names)
    ax.set_xticks(range(len(coefficients)))
    ax.set_xticklabels(coeff_names)  # Replace with your variable names

    # Labels and title
    ax.set_xlabel("Variables")
    ax.set_ylabel("Coefficients")
    ax.set_title(title)

    # Gridlines
    ax.grid(True)

    if path != "None":
        path = path + r"\coefficient_plot.png"
        print("Saving figure in:", path)
        plt.savefig(path)
    else:
        plt.show()


def pred_exp_plot(model, X, y_exp, path="None"):
    """This function create experimental vs predicted plot to show the
        linearity of the fitted model.

    Input:
        model: object that contains all the information of the fitted model
            *see statsmodel OLS.
        X: dataframe with the polinomial features of skilearn
        y: experimetnal response.
        path: folder where the plot is going to be stored
    """
    ## PREPROCESSING DATA
    # Experimental response
    y_exp = y_exp.to_numpy()
    # Predicted response
    y_pred = model.predict(X)
    y_pred = np.reshape(y_pred, (len(y_pred), 1))

    # Create Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Experimental vs Predicted plot", fontsize=16)

    # Predecited vs experimental plot
    ax.scatter(x=y_pred, y=y_exp)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)

    # ax.set_xlim([20, 50])
    # ax.set_ylim([20, 50])
    ax.set_ylabel("Experimental")
    ax.set_xlabel("Predicted")
    ax.grid(True)

    if path != "None":
        path = path + r"\prediction_plot.png"
        print("Saving figure in:", path)
        plt.savefig(path)
    else:
        plt.show()


def main(data_path, effects, response, result_path, del_coeff=[]):
    """main() function.

    Inputs:
        data_path: path/filename where the experimental data is
        effects: independent variables
        response: dependent variables (just one)
        save_path: path/folder name to store plots
        coeff_file: filename to allocate the coefficient of the fitted model
        del_coeff: list of the ommited coefficient of the model. Used to
            get better fit discarting terms with high p-values.
    """
    # CONSTANTS
    ## Create dataframe with coefficents name
    coeff = ["b0", "b1", "b2", "b3", "b11", "b12", "b13", "b22", "b23", "b33"]

    # PREPROCESSING
    ## Create result folder
    if not os.path.exists(result_path):
        # If it doesn't exist, create the folder
        os.mkdir(result_path)
        print(f"Folder '{result_path}' created.")
    else:
        print(f"Folder '{result_path}' already exists.")

    ## IMPORTING AND PREPROCESSING DATAFRAME
    print("Importing data".center(80, "="))

    data_df = pd.read_excel(data_path, sheet_name="RSM_3", skiprows=1)
    data_df = data_df.drop([0, 1])  # drop first two rows (useless)
    data_df = data_df[effects + [response]]  # Rearrangement of data
    data_df = data_df.reset_index(drop=True)  # Reset indexes
    data_df = data_df.apply(pd.to_numeric, errors="coerce")  # Convert to numeric type

    if response == "Recovery":  # Recovery in % if it needed
        data_df["Recovery"] = data_df["Recovery"] * 100

    print("\n", data_df, "\n")

    ## MODEL FITTING PREPROCESSING
    X = data_df[effects]
    y = data_df[response]

    polynomial_features = PolynomialFeatures(degree=2)  # Define the model as quadratic
    X = polynomial_features.fit_transform(X)  # Apply cuadratic model to X
    X = pd.DataFrame(data=X, columns=coeff)  # Define X as dataframe
    X = X.drop(del_coeff, axis=1)  # Deleting terms of the model (for best fit)

    # Fitting the model and show summary
    model = sm.OLS(y, X).fit()
    print("\n", model.summary(), "\n")

    # Save model stats in results folder
    print("Saving model statistics...\n")
    with open(os.path.join(result_path, "model_stats.txt"), "w") as fh:
        fh.write(model.summary().as_text())

    # MODEL PARAMETERS
    coeff_series = pd.Series(0.0, index=coeff)  # Series with zeros
    params = model.params  # Extract model coefficients
    coeff_series[params.index] = params  # Update the coefficients Series

    ## Saving coefficient in csv file
    print("\nSaving model coefficient...\n", coeff_series)
    coeff_series.to_csv(os.path.join(result_path, "model_params.csv"), header=False)

    ## PLOT REGRESSION ANAYLISIS
    # Coefficient plot
    coef_plot(model, normalize="None", whiskers="ci", path=result_path)
    # Residual plot
    residual_plot(model, X, y, normalize="True", path=result_path)
    # Predicted vs experimental plot
    pred_exp_plot(model, X, y, path=result_path)


########################### MAIN CODE ####################################
path = r"C:\Users\marco\OneDrive - usach.cl\DLLME of PCB77 employing designed DES\Hojas de calculo\Experimental.xlsx"

## SECOND RUN WITH EXPERIMENTAL SAMPLE CONCENTRATION
main(
    data_path=path,
    effects=[
        "Extractant-Dispersant Ratio",
        "Sample volume",
        "Extraction mixture volume",
    ],
    response="Recovery",  # ONLY ONE RESPONSE
    result_path=r"Final results - REC",
    del_coeff=[
        # "b0",
        "b1",  #
        # "b2",
        # "b3",
        "b12",  #
        "b13",  #
        "b23",  #
        "b11",  #
        "b22",  #
        # "b33",
    ],  # Ommited terms of the model
)

main(
    data_path=path,
    effects=[
        "Extractant-Dispersant Ratio",
        "Sample volume",
        "Extraction mixture volume",
    ],
    response="Enrichment factor",
    result_path=r"Final results - EF",
    del_coeff=[
        # "b0",
        # "b1",
        "b2",  #
        # "b3",
        "b12",  #
        "b13",  #
        "b23",  #
        # "b11",
        "b22",  #
        "b33",  #
    ],  # Ommited terms of the model
)
