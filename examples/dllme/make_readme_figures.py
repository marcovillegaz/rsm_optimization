"""This script creates the figures of the README in the assets folder: the hero
image (response surface and contour map of the Enrichment Factor) and a grid
with the diagnostic plots (coefficients, residuals and prediction) of the same
model. The simplified coefficients are read from the results of fit_models.py."""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rsm_design._logging import enable_console_logging
from rsm_design.data import load_coefficients, load_experiments
from rsm_design.model import fit_model
from rsm_design.optimize import DesignRegion, optimize
from rsm_design.plots import plot_diagnostics, response_grid

here = Path(__file__).resolve().parent
experiments_file = here / ".." / ".." / "data" / "dllme" / "experimental_results.xlsx"
coefficients_file = (
    here / "results" / "rsm_simplified" / "EnrichmentFactor_model_params.csv"
)
assets_folder = here / ".." / ".." / "assets"
sheet_name = "RSM_3"
factors = ["Extractant-Dispersant Ratio", "Sample volume", "Extraction mixture volume"]
response = "Enrichment Factor"
vary = ("Extractant-Dispersant Ratio", "Extraction mixture volume")
fixed_factor = "Sample volume"
ef_drop_terms = ["b2", "b12", "b13", "b23", "b22", "b33"]
DPI = 300  # CONSTANT
extend = 0.25  # Fraction of the data range the surface goes beyond it

####################################################################################

enable_console_logging()
assets_folder.mkdir(parents=True, exist_ok=True)

# PROCESSING DATA
data_df = load_experiments(
    path=experiments_file,
    factors=factors,
    response=response,
    sheet=sheet_name,
    header_row=1,
    skip_rows=2,
)
coefficients = load_coefficients(path=coefficients_file)
region = DesignRegion.from_data(df=data_df, factors=factors)
optimum = optimize(coefficients=coefficients, region=region)
fixed = {fixed_factor: region.center[factors.index(fixed_factor)]}

# Same limits of the surface plot (data range extended) and the contour (data range)
limits = {}
for factor in vary:
    low, high = data_df[factor].min(), data_df[factor].max()
    limits[factor] = (low - extend * (high - low), high + extend * (high - low))
X1, X2, Y = response_grid(
    coefficients=coefficients,
    factors=factors,
    vary=vary,
    fixed=fixed,
    limits=limits,
    grid=100,
)
opt_x = [optimum.factors[factor] for factor in vary]

# HERO IMAGE
print("hero_surface.png".center(80, "-"))
fig = plt.figure(figsize=(13, 5.2))
ax3d = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
ax3d.plot_surface(X1, X2, Y, cmap="viridis", alpha=0.85, antialiased=True, zorder=5)
ax3d.scatter(
    xs=data_df[vary[0]],
    ys=data_df[vary[1]],
    zs=data_df[response],
    c="red",
    s=14,
    depthshade=False,
    zorder=10,
)
ax3d.scatter(
    xs=opt_x[0],
    ys=opt_x[1],
    zs=optimum.response * 1.05,
    marker="*",
    c="gold",
    edgecolors="k",
    s=180,
    zorder=11,
)
ax3d.view_init(elev=22, azim=45)
ax3d.set_xlabel("Extractant-Dispersant Ratio (-)", labelpad=8)
ax3d.set_ylabel("Extraction mixture volume (µL)", labelpad=8)
ax3d.set_zlabel(response)
ax3d.set_title("Response surface")

ax = fig.add_subplot(1, 2, 2)
contour = ax.contourf(X1, X2, Y, levels=20, cmap="viridis")
ax.contour(X1, X2, Y, levels=10, colors="white", linewidths=0.4, alpha=0.6)
ax.scatter(data_df[vary[0]], data_df[vary[1]], c="red", s=22, label="Experiments")
ax.scatter(
    opt_x[0],
    opt_x[1],
    marker="*",
    c="gold",
    edgecolors="k",
    s=260,
    label=f"Optimum (EF = {optimum.response:.1f})",
)
for set_lim, factor in ((ax.set_xlim, vary[0]), (ax.set_ylim, vary[1])):
    low, high = data_df[factor].min(), data_df[factor].max()
    set_lim(low - 0.06 * (high - low), high + 0.06 * (high - low))
ax.set_xlabel("Extractant-Dispersant Ratio (-)")
ax.set_ylabel("Extraction mixture volume (µL)")
ax.set_title(f"Contour map ({fixed_factor.lower()} = {fixed[fixed_factor]:g} mL)")
ax.legend(loc="upper right", framealpha=0.9)
fig.colorbar(contour, ax=ax, label=response)
fig.tight_layout()
fig.savefig(assets_folder / "hero_surface.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# DIAGNOSTICS GRID
print("diagnostics_grid.png".center(80, "-"))
result = fit_model(
    df=data_df, factors=factors, response=response, drop_terms=ef_drop_terms
)
with tempfile.TemporaryDirectory() as temp_folder:
    paths = plot_diagnostics(result=result, folder=temp_folder, dpi=200)
    images = [plt.imread(path) for path in paths]


def trim(image, margin=10):
    """Crop the white border of an image."""
    content = np.where(image[:, :, :3].min(axis=2) < 0.98)
    rows, columns = content
    return image[
        max(rows.min() - margin, 0) : rows.max() + margin,
        max(columns.min() - margin, 0) : columns.max() + margin,
    ]


images = [trim(image) for image in images]
widths = [image.shape[1] / image.shape[0] for image in images]  # Aspect ratios
fig, axes = plt.subplots(
    1, 3, figsize=(5 * sum(widths), 5), gridspec_kw={"width_ratios": widths}
)
for ax, image, label in zip(axes, images, ["(a)", "(b)", "(c)"], strict=True):
    ax.imshow(image)
    ax.set_title(label, loc="left", fontsize=14)
    ax.axis("off")
fig.tight_layout()
fig.savefig(assets_folder / "diagnostics_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
