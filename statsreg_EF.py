import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from scipy.stats import norm

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.gridspec import GridSpec


def data_arrange(effects, response, path):
    """This function import the experimental results (effects) and the response
    (experimental points) of multiple responses. THe function reorder the data
    in a signle dataframe wich is used in further analysis.

    Input:
        effects: list with the effects names
        response: list with the only response name
        path: path of the csv fiole that contains the data

        *effects and response must be equal to the columns name in csv file.

    Output:
        data_df: dataframe that the first columns are the effects and the last
        columns is the repsonse."""

    print("Importing data".center(80, "="))

    data = pd.read_csv(path, sep=";", decimal=",")

    # Data frame with columns = [effects,response]
    data_df = data[effects + response]
    print("\n", data_df.head())

    return data_df


def residual_plot(model, X, y_exp, normalize="None", path="None"):
    """This function create a residual plot to analyses their distribution

    Input:
        model: object that contains all the information of the fitted model
            *see statsmodel OLS.
        X: dataframe with the polinomial features of skilearn
        y_exp: experimental response.
        normalize: True if yo want a normalized plot
    """
    ## PREPROCESSING DATA
    # Experimental response
    y_exp = y_exp.to_numpy()
    # Predicted response
    y_pred = model.predict(X)
    y_pred = np.reshape(y_pred, (len(y_pred), 1))
    # Residual
    resd = y_exp - y_pred

    # Normalized residual statement
    if normalize == "True":
        resd = resd / np.std(resd)
        ylabel = "Normalized residuals"
    elif normalize == "None":
        ylabel = "Residuals"

    ## PLOT
    # Create figure
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(6)
    fig.suptitle("Residual plot", fontsize=16)
    # Grid spec
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 1))

    # 1. Residual plot
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(y_pred, resd)
    ax.axhline(y=0)
    ax.set_xlabel("y predicted")
    ax.set_ylabel(ylabel)

    # 2. Normal distribution with histogram
    N = 17  # number measurement
    num_bins = 9  # numebr of bars in histograms

    y = np.linspace(resd.min(), resd.max(), 1000)
    bin_width = (resd.max() - resd.min()) / num_bins

    ax_marg_y = fig.add_subplot(gs[0, 1])
    ax_marg_y.hist(x=resd, bins=num_bins, orientation="horizontal")
    ax_marg_y.plot(norm.pdf(y) * N * bin_width, y)

    if path != "None":
        path = path + r"\residual_plot.png"
        print("Saving figure in:", path)
        plt.savefig(path)
    else:
        plt.show()


def coef_plot(model, normalize="None", whiskers="ci", path="None"):
    """This function create two plots to analyse the model:
    Experimental vs predicted: to show the linearity of the model
    Coefficient plot: to show the confidence interval of the model cofficients

    Input:
        model: object that contains all the information of the fitted model
            *see statsmodel OLS.
        X: dataframe with the polinomial features of skilearn
        y: experimetnal response.
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
    """This function create two plots to analyse the model:
    Experimental vs predicted: to show the linearity of the model
    Coefficient plot: to show the confidence interval of the model cofficients

    Input:
        model: object that contains all the information of the fitted model
            *see statsmodel OLS.
        X: dataframe with the polinomial features of skilearn
        y: experimetnal response.
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

    ax.set_xlim([0, 40])
    ax.set_ylim([0, 40])
    ax.set_ylabel("Experimental")
    ax.set_xlabel("Predicted")

    if path != "None":
        path = path + r"\prediction_plot.png"
        print("Saving figure in:", path)
        plt.savefig(path)
    else:
        plt.show()


def response_surface(factors):
    coefficients = [-29.507520, 8.026565, 7.828037, -0.404694]
    # model for enrichment factor:
    #   y = b1*x1 + b2*x2 + b11*x1^2 + b22*x2^2
    X1 = factors[:, 0]
    X2 = factors[:, 1]
    Y = (
        +coefficients[0] * X1
        + coefficients[1] * X2
        + coefficients[2] * X1**2
        + coefficients[3] * X2**2
    )
    return Y


def rsm_plot_2var(data_df):
    """This function plot the response surface of the model

    Input:
        model: object that contains all the information of the fitted model
            *see statsmodel OLS.
        X: dataframe with the polinomial features of skilearn
        y: experimetnal response.

    """

    print("This function create the surface plot".center(80, "="))
    grid_space = 100  # fixed

    # fILTER DATAFRAME
    data_df = data_df[data_df["Extractant mixture volume"] == 390]

    x1_exp = data_df.iloc[:, 0]
    x2_exp = data_df.iloc[:, 1]
    y_exp = data_df.iloc[:, -1]

    # model notation: y = model(x1,x2)

    # range fo variable effects
    x1_pred = np.linspace(min(x1_exp), max(x1_exp), grid_space)
    x2_pred = np.linspace(min(x2_exp), max(x2_exp), grid_space)

    x1_pred = np.linspace(0, max(x1_exp), grid_space)
    x2_pred = np.linspace(0, max(x2_exp), grid_space)

    # Create grid to predict response
    x1_pred, x2_pred = np.meshgrid(x1_pred, x2_pred)

    X_pred = np.array([x1_pred.flatten(), x2_pred.flatten()]).T

    y_pred = response_surface(X_pred)
    y_pred = y_pred.reshape(x1_pred.shape)

    # Plot model visualization
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

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

    ax.plot_surface(
        x1_pred, x2_pred, y_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax.set_xlabel("Extractant-dispersant ratio", fontsize=12)
    ax.set_ylabel("Sample volume", fontsize=12)
    ax.set_zlabel("Enrichment factor", fontsize=12)
    ax.set_zlim([0, 50])
    ax.locator_params(nbins=4, axis="x")
    ax.locator_params(nbins=5, axis="x")
    ax.set_title(
        "model: $y(x_{1},x_{2}) = b_{1}x_{1} + b_{2}x_{2} + b_{11}x_{1}^2 + b_{22}x_{2}^2$",
        fontsize=12,
    )

    plt.show()


def main(path, effects, response, del_coeff=[]):
    # Polinomial feature: define the model as quadratic

    # Create dataframe with coefficents name
    coeff = ["b0", "b1", "b2", "b3", "b11", "b12", "b13", "b22", "b23", "b33"]

    # Polinomial feature: define the model as quadrati
    polynomial_features = PolynomialFeatures(degree=2)

    # Arrange experimental data into dataframe
    data_df = data_arrange(effects, response, path)

    X = data_df[effects]
    X = polynomial_features.fit_transform(X)
    X = pd.DataFrame(data=X, columns=coeff)

    y = data_df[response]

    # Deleting terms of the model
    X = X.drop(del_coeff, axis=1)

    # Fitting the model
    model = sm.OLS(y, X).fit()
    print("\n", model.summary(), "\n")

    # Coefficient plot
    coef_plot(model, normalize="None", whiskers="ci", path=path_save)
    # Residual plot
    residual_plot(model, X, y, normalize="True", path=path_save)
    # Predicted vs experimental plot
    pred_exp_plot(model, X, y, path=path_save)


path = r"C:\Users\marco\python-projects\rsm_design\rsm_results.csv"
path_save = r"C:\Users\marco\python-projects\rsm_design\EF"

effects = ["Extractant-dispersant ratio", "Sample volume", "Extractant mixture volume"]
response = ["Enrichment factor"]

# This terms of the model will be deleted
del_coeff = ["b0", "b3", "b12", "b13", "b23", "b33"]

main(path, effects, response, del_coeff)
