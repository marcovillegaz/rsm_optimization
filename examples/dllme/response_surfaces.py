"""This script computes the optimum of the simplified models of the Enrichment
Factor and the Recovery inside the region of the experimental design, and then
creates the surface plot and the contour plot of each response with the
experimental points. The simplified coefficients are read from the results of
fit_models.py.

The optimum can differ from the one reported in 2024 (L-BFGS-B with a penalty
that returned 1e6 outside the ellipsoid): the optimum is now found with SLSQP
and the ellipsoid as a real constraint, and it predicts a higher response."""

from pathlib import Path

from rsm_design._logging import enable_console_logging
from rsm_design.data import load_coefficients, load_experiments
from rsm_design.optimize import DesignRegion, optimize
from rsm_design.plots import contour_plot, surface_plot

here = Path(__file__).resolve().parent
experiments_file = here / ".." / ".." / "data" / "dllme" / "experimental_results.xlsx"
coefficients_folder = here / "results" / "rsm_simplified"
surfaces_folder = here / "results" / "surfaces"
contours_folder = here / "results" / "contours"
sheet_name = "RSM_3"
factors = ["Extractant-Dispersant Ratio", "Sample volume", "Extraction mixture volume"]
labels = {
    "Extractant-Dispersant Ratio": "Extractant-Dispersant Ratio (-)",
    "Sample volume": "Sample volume (mL)",
    "Extraction mixture volume": "Extraction mixture volume (µL)",
}

####################################################################################

enable_console_logging()

# EXTRACT EXPERIMENTAL RESULTS
ef_df = load_experiments(
    path=experiments_file,
    factors=factors,
    response="Enrichment Factor",
    sheet=sheet_name,
    header_row=1,
    skip_rows=2,
)

rec_df = load_experiments(
    path=experiments_file,
    factors=factors,
    response="Recovery",
    sheet=sheet_name,
    header_row=1,
    skip_rows=2,
    scale=100,  # Recovery from fraction to percent
)
rec_df = rec_df.rename(columns={"Recovery": "Recovery (%)"})  # Label of the z axis

# MODEL COEFFICIENTS
ef_coefficients = load_coefficients(
    path=coefficients_folder / "EnrichmentFactor_model_params.csv"
)
rec_coefficients = load_coefficients(
    path=coefficients_folder / "Recovery_model_params.csv"
)

# RESPONSE SURFACE OPTIMIZATION
region = DesignRegion.from_data(df=ef_df, factors=factors)  # Same design for both

print("enrichment factor".center(80, "-"))
ef_optimum = optimize(coefficients=ef_coefficients, region=region)
print(ef_optimum.factors, ef_optimum.response)
print(
    "2024 L-BFGS-B optimum (can differ): ratio 0.35865, volume 9.0, "
    "mixture 299.77, EF 47.047"
)

print("recovery".center(80, "-"))
rec_optimum = optimize(coefficients=rec_coefficients, region=region)
print(rec_optimum.factors, rec_optimum.response)

# ENRICHMENT FACTOR PLOTS
ef_vary = ("Extractant-Dispersant Ratio", "Extraction mixture volume")
surface_plot(
    coefficients=ef_coefficients,
    factors=factors,
    vary=ef_vary,
    data=ef_df,
    response="Enrichment Factor",
    fixed={"Sample volume": region.center[1]},
    optimum=ef_optimum,
    labels=labels,
    view=(20, 45),
    folder=surfaces_folder,
    file_name="EnrichmentFactor_surface.png",
)
contour_plot(
    coefficients=ef_coefficients,
    factors=factors,
    vary=ef_vary,
    data=ef_df,
    response="Enrichment Factor",
    fixed={"Sample volume": region.center[1]},
    optimum=ef_optimum,
    labels=labels,
    folder=contours_folder,
    file_name="EnrichmentFactor_contour.png",
)

# RECOVERY PLOTS
rec_vary = ("Sample volume", "Extraction mixture volume")
surface_plot(
    coefficients=rec_coefficients,
    factors=factors,
    vary=rec_vary,
    data=rec_df,
    response="Recovery (%)",
    fixed={"Extractant-Dispersant Ratio": region.center[0]},
    optimum=rec_optimum,
    labels=labels,
    view=(20, -45),
    folder=surfaces_folder,
    file_name="Recovery_surface.png",
)
contour_plot(
    coefficients=rec_coefficients,
    factors=factors,
    vary=rec_vary,
    data=rec_df,
    response="Recovery (%)",
    fixed={"Extractant-Dispersant Ratio": region.center[0]},
    optimum=rec_optimum,
    labels=labels,
    folder=contours_folder,
    file_name="Recovery_contour.png",
)
