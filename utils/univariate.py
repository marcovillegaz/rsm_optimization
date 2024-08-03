"""This script contains function to work with the univariate analysis"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def rearrenge_data(file_path, response, sheet_name):
    # Import data from excel
    data_df = pd.read_excel(file_path, sheet_name, skiprows=1)

    # Rearrangement of data
    new_data = pd.DataFrame(
        {
            "Group": data_df["Sample name"],
            "Response": pd.to_numeric(data_df[response], errors="coerce"),
        }
    )

    new_data = new_data.drop([0, 1])  # delete unuseful rows
    new_data = new_data.reset_index(drop=True)  # reset indexes

    # Separate individual screenings
    DES_df = new_data.iloc[0:12]  # Screening of extractant solvente
    centrifuge_df = new_data.iloc[12:]  # Screening of centrifugue time

    return DES_df, centrifuge_df


def stats_analysis(df):
    # Perform one-way ANOVA using statsmodels
    model = ols(formula="Response ~ C(Group)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print ANOVA table
    print("\nANOVA Table:")
    print(anova_table)

    # Perform Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(df["Response"], df["Group"])

    # Print the Tukey HSD results
    print("\nTukey HSD Results:")
    print(tukey_result.summary())


def final_boxplot(df, color, xlabel, text, file_name):
    # Create a single subplot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Box plot for the dataframe
    df.boxplot(
        column="Response",
        by="Group",
        ax=ax,
        patch_artist=True,
        boxprops=dict(facecolor=color, color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="white"),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("EF")
    ax.set_title(" ")

    # Add label (a) below the subplot
    ax.text(
        0.475, -0.15, text, transform=ax.transAxes, fontsize=14, verticalalignment="top"
    )

    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(file_name, dpi=800)
    plt.show()
