"""This script perform the statistic analysis for the univariate anaylisis"""

from utils.univariate import *


file_path = r"C:\Users\marco\OneDrive - usach.cl\DLLME of PCB77 employing designed DES\Hojas de calculo\Experimental.xlsx"
response = "Enrichment factor"
sheet_name = "Screening_3"

print("UNIVARIATE ANALYSIS".center(80, "="))
# Extract experimental results
DES_df, centrifuge_df = rearrenge_data(file_path, response, sheet_name)

# Perform statistical analysis
print("extractant solvent".center(80, "-"))
print("experimental data:\n", DES_df)
stats_analysis(DES_df)

print("centrifuge time".center(80, "-"))
print("experimental data:\n", centrifuge_df)
stats_analysis(centrifuge_df)

# Create box plot for DES
final_boxplot(
    DES_df,
    color="tab:blue",
    xlabel="DES",
    text="(a)",
    file_name="univariate_results\DES_uni.jpg",
)

# Create box plot for centrifuge time
final_boxplot(
    centrifuge_df,
    color="tab:orange",
    xlabel="Centriuge time",
    text="(b)",
    file_name="univariate_results\centrifuge_uni.jpg",
)
