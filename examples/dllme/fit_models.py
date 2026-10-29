"""This script fits the quadratic response surface models of the DLLME (CCD of
three factors) to the Recovery and the Enrichment Factor. First the full model
(all the terms) is fitted, then the simplified model without the terms that are
not significant. The coefficients, the statistical summary and the diagnostic
plots of each model are saved in its results folder."""

from pathlib import Path

from rsm_design._logging import enable_console_logging
from rsm_design.data import load_experiments
from rsm_design.model import fit_model, save_results

here = Path(__file__).resolve().parent
experiments_file = here / ".." / ".." / "data" / "dllme" / "experimental_results.xlsx"
full_folder = here / "results" / "rsm_full"
simplified_folder = here / "results" / "rsm_simplified"
sheet_name = "RSM_3"
factors = ["Extractant-Dispersant Ratio", "Sample volume", "Extraction mixture volume"]

####################################################################################

enable_console_logging()

# EXTRACT EXPERIMENTAL RESULTS
rec_df = load_experiments(
    path=experiments_file,
    factors=factors,
    response="Recovery",
    sheet=sheet_name,
    header_row=1,
    skip_rows=2,
    scale=100,  # Recovery from fraction to percent
)

ef_df = load_experiments(
    path=experiments_file,
    factors=factors,
    response="Enrichment Factor",
    sheet=sheet_name,
    header_row=1,
    skip_rows=2,
)

# FULL MODEL
print("full model".center(80, "-"))
rec_full = fit_model(
    df=rec_df,
    factors=factors,
    response="Recovery",
    drop_terms=[
        # "b0",
        # "b1",
        # "b2",
        # "b3",
        # "b12",
        # "b13",
        # "b23",
        # "b11",
        # "b22",
        # "b33",
    ],  # Omitted terms of the model
)
save_results(result=rec_full, folder=full_folder)

ef_full = fit_model(
    df=ef_df,
    factors=factors,
    response="Enrichment Factor",
    drop_terms=[
        # "b0",
        # "b1",
        # "b2",
        # "b3",
        # "b12",
        # "b13",
        # "b23",
        # "b11",
        # "b22",
        # "b33",
    ],  # Omitted terms of the model
)
save_results(result=ef_full, folder=full_folder)

# SIMPLIFIED MODEL
print("simplified model".center(80, "-"))
rec_simplified = fit_model(
    df=rec_df,
    factors=factors,
    response="Recovery",
    drop_terms=[
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
    ],  # Omitted terms of the model
)
save_results(result=rec_simplified, folder=simplified_folder)

ef_simplified = fit_model(
    df=ef_df,
    factors=factors,
    response="Enrichment Factor",
    drop_terms=[
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
    ],  # Omitted terms of the model
)
save_results(result=ef_simplified, folder=simplified_folder)
