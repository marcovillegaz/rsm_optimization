"""In this script"""

from utils.rsm import *


file_path = r"experimental_results.xlsx"
sheet_name = "RSM_3"
factors = ["Extractant-Dispersant Ratio", "Sample volume", "Extraction mixture volume"]

exp_rec = extract_data(
    file_path=file_path, sheet_name=sheet_name, factors=factors, response="Recovery"
)
exp_ef = extract_data(
    file_path=file_path,
    sheet_name=sheet_name,
    factors=factors,
    response="Enrichment Factor",
)

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
    results_folder="rsm_results",
)

# Change the delted coefficients
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
    results_folder="rsm_results",
)
