"""This script present multiple function to plot the result obtain in this study."""

import os
import numpy as np

from scipy.stats import norm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def residual_plot(model, X, y_exp, response, normalize=None, folder=None):
    """This function create a residual plot to analyses their distribution

    Input:
        model: object that contains all the information of the fitted model
            *see statsmodel OLS.
        X: dataframe with the polinomial features of skilearn
        y_exp: experimental response.
        normalize: True if yoU want a normalized plot
        path: folder where the plot is going to be stored
    """
    print("residual_plot()")
    # PROCESSING DATA
    y_exp = np.reshape(y_exp, (len(y_exp), 1))  # Experimental response
    y_pred = np.reshape(model.predict(X), (len(y_exp), 1))  # Predicted response
    resd = y_exp - y_pred  # Residual

    # Calculate MSE
    mse = mean_squared_error(y_exp, y_pred)
    print(f"\tMean Squared Error (MSE): {mse}")

    # Calculate RMSE
    rmse = np.sqrt(mse)
    print(f"\tRoot Mean Squared Error (RMSE): {rmse}")

    ## Normalized residual statement
    if normalize == True:
        resd = resd / np.std(resd)
        ylabel = "Normalized residuals"
    elif normalize == None:
        ylabel = "Residuals"

    # MAKING PLOT
    fig = plt.figure(figsize=(10, 6))  # Create figure
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
    if folder != None:
        file_name = f'{response.replace(" ", "")}_rsd_plot.png'
        print(f"\tSaving {file_name}")
        plt.savefig(os.path.join(folder, file_name))
    else:
        plt.show()


def coeff_plot(model, response, normalize=None, whiskers="ci", folder=None):
    """
    Create a coefficient plot to show the confidence interval of the model
    cofficients. If the whiskers cut the horizontal line, the coefficient is
    inappropriate.

    params:

    model: object that contains all the information of the fitted model
    normalize: True if you want a normalized plot
    whiskers: "ci" for cofidence intervals, "stderr" for standard error
    path: folder where the plot is going to be stored
    """
    print("coeff_plot()")
    # Preprocessing
    # Get the coefficients and their confidence intervals
    coefficients = model.params
    coeff_names = model.params.index

    if normalize == True:
        error = model.bse
        coefficients = coefficients / error

    elif normalize == None:
        # Bar plot with coefficient intervals
        if whiskers == "ci":
            ci = model.conf_int().T
            ci = ci.to_numpy()
            ci_low = ci[0, :]
            ci_high = ci[1, :]
            error = [coefficients - ci_low, ci_high - coefficients]

        # Bar plot with coefficient intervals
        elif whiskers == "stderr":
            error = model.bse

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
    # Gridlines
    ax.grid(True)

    if folder != None:
        file_name = f'{response.replace(" ", "")}_coeff_plot.png'
        print(f"\tSaving {file_name}")
        plt.savefig(os.path.join(folder, file_name))
    else:
        plt.show()


def prediction_plot(model, X, y_exp, response, folder=None):
    """Create experimental vs predicted plot to show the linearity of the
    fitted model.

    params:

    model: object that contains all the information of the fitted model
        *see statsmodel OLS.
    X: dataframe with the polinomial features of skilearn
    y: experimetnal response.
    path: folder where the plot is going to be stored
    """
    print("prediction_plot()")
    ## PREPROCESSING DATA
    # Experimental response
    y_exp = y_exp.to_numpy()
    # Predicted response
    y_pred = model.predict(X)
    y_pred = np.reshape(y_pred, (len(y_pred), 1))

    # Create Figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Predecited vs experimental plot
    ax.scatter(x=y_pred, y=y_exp)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)

    # ax.set_xlim([20, 50])
    # ax.set_ylim([20, 50])
    ax.set_ylabel("Experimental")
    ax.set_xlabel("Predicted")
    ax.grid(True)

    if folder != None:
        file_name = f'{response.replace(" ", "")}_prediction_plot.png'
        print(f"\tSaving {file_name}")
        plt.savefig(os.path.join(folder, file_name))
    else:
        plt.show()
