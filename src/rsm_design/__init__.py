"""Response Surface Methodology: fit, optimize and plot quadratic models."""

__version__ = "0.1.0"

from rsm_design.data import (
    load_coefficients,
    load_experiments,
    read_table,
    read_units,
    save_coefficients,
)
from rsm_design.model import (
    FitResult,
    RankDeficientDesignError,
    check_estimability,
    design_matrix,
    file_stem,
    fit_model,
    n_factors_from_terms,
    n_terms,
    predict,
    quadratic_form,
    save_results,
    term_names,
)
from rsm_design.optimize import (
    DesignRegion,
    OptimizationResult,
    optimize,
    stationary_point,
)
from rsm_design.plots import (
    coeff_plot,
    contour_plot,
    plot_diagnostics,
    prediction_plot,
    residual_plot,
    response_grid,
    surface_plot,
)
from rsm_design.univariate import (
    UnivariateResult,
    group_boxplot,
    load_groups,
    one_way_anova,
)

__all__ = [
    "DesignRegion",
    "FitResult",
    "OptimizationResult",
    "RankDeficientDesignError",
    "UnivariateResult",
    "check_estimability",
    "coeff_plot",
    "contour_plot",
    "design_matrix",
    "file_stem",
    "fit_model",
    "group_boxplot",
    "load_coefficients",
    "load_experiments",
    "load_groups",
    "n_factors_from_terms",
    "n_terms",
    "one_way_anova",
    "optimize",
    "plot_diagnostics",
    "predict",
    "prediction_plot",
    "quadratic_form",
    "read_table",
    "read_units",
    "residual_plot",
    "response_grid",
    "save_coefficients",
    "save_results",
    "stationary_point",
    "surface_plot",
    "term_names",
]
