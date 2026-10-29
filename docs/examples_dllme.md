# DLLME examples

Scripts in `examples/dllme/` reproduce the analyses of the published study.
They follow the thin-script style: parameters at the top, keyword-argument
calls below. Inputs come from `data/dllme/`; outputs go to
`examples/dllme/results/`.

| Script | What it does | Writes |
|---|---|---|
| `fit_models.py` | Fits the full and simplified quadratic models for Enrichment Factor and Recovery (sheet `RSM_3`) | `results/rsm_full/`, `results/rsm_simplified/` (params CSV, stats TXT, coefficient / residual / prediction plots) |
| `response_surfaces.py` | Optimises both responses and draws surfaces and contours | `results/surfaces/`, `results/contours/` |
| `univariate_screening.py` | One-way ANOVA + Tukey screening and box plots (sheet `Screening_3`) | `results/univariate/` |
| `cosmo_screening.py` | COSMO-RS solvent screening plot | `results/cosmo/COSMO_RS_screening.jpeg` |
| `sigma_profile.py` | Sigma-profile plots from `data/dllme/cosmo/` | `results/cosmo/sigma_profile.png` |
| `make_readme_figures.py` | Builds the README images | `assets/hero_surface.png`, `assets/diagnostics_grid.png` |

Run from the repository root:

```powershell
python examples/dllme/fit_models.py
```

Back to [index](index.md).
