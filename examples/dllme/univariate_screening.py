"""This script performs the statistical analysis (ANOVA and Tukey HSD) of the
univariate screenings of the DLLME: extractant solvent and centrifuge time."""

from pathlib import Path

from rsm_design._logging import enable_console_logging
from rsm_design.univariate import group_boxplot, load_groups, one_way_anova

here = Path(__file__).resolve().parent
experiments_file = here / ".." / ".." / "data" / "dllme" / "experimental_results.xlsx"
results_folder = here / "results" / "univariate"
sheet_name = "Screening_3"
group_column = "Sample name"
response_column = "Enrichment factor 260"
n_des_rows = 12  # CONSTANT: first rows are the screening of extractant solvent

####################################################################################

enable_console_logging()

# EXTRACT EXPERIMENTAL RESULTS
data_df = load_groups(
    path=experiments_file,
    sheet=sheet_name,
    group_column=group_column,
    response_column=response_column,
)
DES_df = data_df.iloc[0:n_des_rows]  # Screening of extractant solvent
centrifuge_df = data_df.iloc[n_des_rows:]  # Screening of centrifuge time

# STATISTICAL ANALYSIS
print("extractant solvent".center(80, "-"))
DES_result = one_way_anova(df=DES_df)

print("centrifuge time".center(80, "-"))
centrifuge_result = one_way_anova(df=centrifuge_df)

# BOX PLOTS
group_boxplot(
    df=DES_df,
    color="tab:blue",
    xlabel="DES",
    ylabel="EF",
    panel_label="(a)",
    folder=results_folder,
    file_name="DES_uni.jpg",
)

group_boxplot(
    df=centrifuge_df,
    color="tab:orange",
    xlabel="Centrifuge time",
    ylabel="EF",
    panel_label="(b)",
    folder=results_folder,
    file_name="centrifuge_uni.jpg",
)
