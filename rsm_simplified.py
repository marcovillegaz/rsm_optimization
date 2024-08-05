"""In this script"""

from utils.rsm import *


experiments_file = r"experimental_results.xlsx"
results_folder = r"rsm_results_simplified"
sheet_name = "RSM_3"
factors = ["Extractant-Dispersant Ratio", "Sample volume", "Extraction mixture volume"]

# Extracta data for REC response
exp_rec = extract_data(
    file_path=experiments_file,
    sheet_name=sheet_name,
    factors=factors,
    response="Recovery",
)

# Extract data for EF response
exp_ef = extract_data(
    file_path=experiments_file,
    sheet_name=sheet_name,
    factors=factors,
    response="Enrichment Factor",
)

# Fit model for REC response
model_fitting(
    exp_rec,
    factors,
    response="Recovery",
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
    ],  # Ommited terms of the model, results_folder)
    results_folder=results_folder,
)

# FIt model for Ef response
model_fitting(
    exp_ef,
    factors,
    response="Enrichment Factor",
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
    ],  # Ommited terms of the model, results_folder)
    results_folder=results_folder,
)
