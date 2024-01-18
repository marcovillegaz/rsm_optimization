import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def stats_analysis(df):
    print("The data to analyse is:\n", df)
    # Perform one-way ANOVA using statsmodels
    model = ols(formula="Response ~ C(Group)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print ANOVA table
    print("ANOVA Table:")
    print(anova_table)

    # Perform Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(df["Response"], df["Group"])

    # Print the Tukey HSD results
    print("\nTukey HSD Results:")
    print(tukey_result.summary())


def screening_plot(df, response, color, title=""):
    # Create a boxplot using seaborn for both responses
    plt.figure(figsize=(5, 5))
    sns.boxplot(
        x="Group",
        y="Response",
        data=df,
        color=color,
    )
    # Hide the x-axis label
    plt.xlabel(None)
    plt.ylabel("Enrichment factor")
    # plt.xticks(rotation=45)  # Rotate x-axis labels vertically
    plt.title(title)
    plt.show()


# ================================================================================
path = r"C:\Users\marco\OneDrive - usach.cl\DLLME of PCB77 employing designed DES\Hojas de calculo\Experimental.xlsx"
response = "Enrichment factor"

# import data from excel
data_df = pd.read_excel(path, sheet_name="Screening_3", skiprows=1)

print("Columns in sheet are: ")
[print("\t" + col) for col in data_df.columns]

# Rearrangement of data
new_data = pd.DataFrame(
    {
        "Group": data_df["Sample name"],
        "Response": pd.to_numeric(data_df[response], errors="coerce"),
    }
)
new_data = new_data.drop([0, 1])  # delete unuseful rows
new_data = new_data.reset_index(drop=True)  # reset indexes
print("\nThe useful data are:\n", new_data)

# Separate individual screenings
DES_screening_df = new_data.iloc[0:12]  # Screening of extractant solvente
centrifugue_screening_df = new_data.iloc[12:]  # Screening of centrifugue time
print(DES_screening_df)
print(centrifugue_screening_df)


print("\n", "SCREENING OF EXTRACTANT SOLVENT".center(80, "-"))
stats_analysis(DES_screening_df)
print("\n", "SCREENING OF CENTRIFUGUE TIME".center(80, "-"))
stats_analysis(centrifugue_screening_df)

""" screening_plot(
    DES_screening_df,
    response,
    color="royalblue",
    title="DES screening",
)

screening_plot(
    centrifugue_screening_df,
    response,
    color="orange",
    title="Centrifugue time screening",
)
 """

## PLOT FOR THE PAPER
# Create subplots with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot for the first category
sns.boxplot(ax=axes[0], x="Group", y="Response", data=DES_screening_df, color="skyblue")
axes[0].set_title("Extractant solvent")
axes[0].set_xlabel("(a)")
axes[0].set_ylabel("Enrichment factor")

# Boxplot for the second category
sns.boxplot(
    ax=axes[1], x="Group", y="Response", data=centrifugue_screening_df, color="salmon"
)
axes[1].set_title("Centrifugue time")
axes[1].set_xlabel("(b)")
axes[1].set_ylabel("Enrichment factor")

# Show the plot
plt.savefig("screening.png", dpi=500)
