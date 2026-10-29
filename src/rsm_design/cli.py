"""Command line interface of rsm_design: ``rsm-design fit|optimize|plot``.

fit       fit the quadratic model; write coefficients, statistics and plots
optimize  optimum of a fitted model inside the validity region of the design
plot      response surface and/or contour map of a fitted model
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from rsm_design._logging import enable_console_logging
from rsm_design.model import RankDeficientDesignError, file_stem

DEFAULT_OUT = "rsm_results"  # CONSTANT
DEFAULT_DPI = 600  # CONSTANT
DEFAULT_EXTEND = {"surface": 0.25, "contour": 0.0}  # CONSTANT, as in plots.py

########################################################################
# ARGUMENTS
########################################################################


def _common_parser() -> argparse.ArgumentParser:
    """Parent parser with the options that define how the data are read.

    return:
        parent (argparse.ArgumentParser): shared options of all subcommands
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("data", metavar="DATA", help="Excel or CSV file")
    parent.add_argument("--sheet", default=0, help="Excel sheet name (default: first)")
    parent.add_argument(
        "--factors", nargs="+", required=True, metavar="F", help="factor columns"
    )
    parent.add_argument(
        "--response", required=True, metavar="R", help="response column"
    )
    parent.add_argument("--header-row", type=int, default=0, help="header row index")
    parent.add_argument(
        "--skip-rows", type=int, default=0, help="rows below the header to drop"
    )
    parent.add_argument(
        "--scale", type=float, default=1.0, help="multiply the response by this"
    )
    parent.add_argument("--sep", default=",", help="CSV separator")
    parent.add_argument("--decimal", default=".", help="CSV decimal mark")
    group = parent.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="debug messages")
    group.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parent


def _add_region_options(parser: argparse.ArgumentParser) -> None:
    """Add --region and --center (validity region of the optimum)."""
    parser.add_argument(
        "--region", choices=["ellipsoid", "box"], default="ellipsoid", help="region"
    )
    parser.add_argument(
        "--center", choices=["mode", "midrange"], default="mode", help="region centre"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with the three subcommands.

    return:
        parser (argparse.ArgumentParser): parser of ``rsm-design``
    """
    parser = argparse.ArgumentParser(
        prog="rsm-design",
        description="Response surface methodology for designed experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()

    # FIT
    fit = sub.add_parser("fit", parents=[common], help="fit the quadratic model")
    fit.add_argument("--drop", nargs="+", default=[], metavar="TERM", help="terms")
    fit.add_argument("--out", default=DEFAULT_OUT, help="results folder")
    fit.add_argument("--no-plots", action="store_true", help="skip diagnostic plots")
    fit.add_argument(
        "--whiskers",
        choices=["ci", "stderr"],
        default="ci",
        help="error bars of the coefficient plot",
    )
    fit.add_argument(
        "--allow-rank-deficient",
        action="store_true",
        help="warn instead of failing when a term cannot be estimated",
    )
    fit.add_argument("--dpi", type=int, default=DEFAULT_DPI)

    # OPTIMIZE
    opt = sub.add_parser("optimize", parents=[common], help="optimum of the model")
    opt.add_argument("--coefficients", required=True, metavar="CSV")
    opt.add_argument("--minimize", action="store_true", help="minimize, not maximize")
    _add_region_options(opt)
    opt.add_argument("--out", default=None, help="folder for {stem}_optimum.csv")

    # PLOT
    plot = sub.add_parser("plot", parents=[common], help="surface / contour plots")
    plot.add_argument("--coefficients", required=True, metavar="CSV")
    plot.add_argument("--vary", nargs=2, required=True, metavar=("F1", "F2"))
    plot.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="constant factor in real units (repeatable); default: centre point",
    )
    plot.add_argument("--kind", choices=["surface", "contour", "both"], default="both")
    plot.add_argument("--extend", type=float, default=None, help="range extension")
    plot.add_argument("--no-points", action="store_true", help="hide experiments")
    plot.add_argument("--optimum", action="store_true", help="mark the optimum")
    _add_region_options(plot)
    plot.add_argument("--azim", type=float, default=45)
    plot.add_argument("--elev", type=float, default=20)
    target = plot.add_mutually_exclusive_group()
    target.add_argument("--out", default=None, help="folder for the images")
    target.add_argument("--show", action="store_true", help="show, do not save")
    plot.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser


########################################################################
# HELPERS
########################################################################


def _parse_fixed(items: Sequence[str]) -> dict[str, float]:
    """Parse repeated NAME=value options into a dict (real units).

    params:
        items (list): strings such as "Sample volume=9"
    return:
        fixed (dict): factor -> value
    """
    fixed: dict[str, float] = {}
    for item in items:
        name, sep, value = item.rpartition("=")
        if not sep or not name:
            raise ValueError(f"--fixed expects NAME=value, got {item!r}")
        try:
            fixed[name.strip()] = float(value)
        except ValueError:
            raise ValueError(f"--fixed {name!r}: {value!r} is not a number") from None
    return fixed


def _load(args: argparse.Namespace) -> pd.DataFrame:
    """Read the experiments as defined by the shared options.

    params:
        args (Namespace): parsed arguments
    return:
        data_df (pd.DataFrame): factors and response
    """
    from rsm_design.data import load_experiments

    sheet: Any = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)  # --sheet 1 means the second sheet
    return load_experiments(
        path=args.data,
        factors=args.factors,
        response=args.response,
        sheet=sheet,
        header_row=args.header_row,
        skip_rows=args.skip_rows,
        scale=args.scale,
        sep=args.sep,
        decimal=args.decimal,
    )


def _optimum(args: argparse.Namespace, data_df: pd.DataFrame, coefficients: Any):
    """Optimum of the model inside the region defined by the data.

    params:
        args (Namespace): parsed arguments (region, center, minimize)
        data_df (pd.DataFrame): experiments
        coefficients (pd.Series): fitted coefficients
    return:
        result (OptimizationResult): optimal factors and response
    """
    from rsm_design.optimize import DesignRegion, optimize

    region = DesignRegion.from_data(
        df=data_df, factors=args.factors, center=args.center, shape=args.region
    )
    return optimize(
        coefficients=coefficients,
        region=region,
        maximize=not getattr(args, "minimize", False),
    )


def _data_defaults(
    data_df: pd.DataFrame, args: argparse.Namespace, kind: str, fixed: dict
) -> tuple[dict, dict]:
    """Fixed values and limits from the data, for plots that hide the points.

    params:
        data_df (pd.DataFrame): experiments
        args (Namespace): parsed arguments
        kind (str): "surface" or "contour" (sets the default extension)
        fixed (dict): values given by the user, kept as they are
    return:
        fixed, limits (dict, dict): constants of the other factors, axis limits
    """
    extend = DEFAULT_EXTEND[kind] if args.extend is None else args.extend
    limits = {}
    for factor in args.vary:
        lo, hi = float(data_df[factor].min()), float(data_df[factor].max())
        limits[factor] = (lo - extend * (hi - lo), hi + extend * (hi - lo))

    fixed = dict(fixed)
    for factor in args.factors:
        if factor not in args.vary and factor not in fixed:
            values = data_df[factor]
            mode = values.mode().iloc[0]
            repeated = (values == mode).sum() > 1
            fixed[factor] = float(
                mode if repeated else (values.min() + values.max()) / 2
            )
    return fixed, limits


########################################################################
# SUBCOMMANDS
########################################################################


def _run_fit(args: argparse.Namespace) -> None:
    """Fit the model and save the results."""
    from rsm_design.model import fit_model, save_results

    logger = logging.getLogger("rsm_design")
    data_df = _load(args)
    result = fit_model(
        df=data_df,
        factors=args.factors,
        response=args.response,
        drop_terms=args.drop,
        on_rank_deficient="warn" if args.allow_rank_deficient else "raise",
    )
    logger.info(
        f"R2 = {result.r_squared:.4f}, adjusted R2 = {result.adj_r_squared:.4f}, "
        f"RMSE = {result.rmse:.4g}"
    )
    logger.info("Coefficients:\n%s", result.coefficients.to_string())
    save_results(result=result, folder=args.out, plots=False, dpi=args.dpi)
    if not args.no_plots:  # Plotted here so --whiskers can be chosen
        from rsm_design.plots import coeff_plot, prediction_plot, residual_plot

        coeff_plot(result=result, whiskers=args.whiskers, folder=args.out, dpi=args.dpi)
        residual_plot(result=result, folder=args.out, dpi=args.dpi)
        prediction_plot(result=result, folder=args.out, dpi=args.dpi)


def _run_optimize(args: argparse.Namespace) -> None:
    """Print the optimum and, with --out, write it to a CSV."""
    from rsm_design.data import load_coefficients

    data_df = _load(args)
    coefficients = load_coefficients(args.coefficients, n_factors=len(args.factors))
    result = _optimum(args, data_df, coefficients)

    goal = "Minimum" if args.minimize else "Maximum"
    lines = [f"{goal} of {args.response} ({args.region} region):"]
    lines += [f"    {name} = {value:.6g}" for name, value in result.factors.items()]
    lines.append(f"    predicted {args.response} = {result.response:.6g}")
    if not result.inside_region:
        lines.append("    warning: the optimum is outside the region")
    print("\n".join(lines))

    if args.out is not None:  # Only when asked explicitly
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        rows = list(result.factors.items()) + [(args.response, result.response)]
        path = out / f"{file_stem(args.response)}_optimum.csv"
        pd.DataFrame(rows).to_csv(path, header=False, index=False)
        print(f"Saved {path}")


def _run_plot(args: argparse.Namespace) -> None:
    """Draw the surface and/or contour plots."""
    from rsm_design.data import load_coefficients
    from rsm_design.plots import contour_plot, surface_plot

    data_df = _load(args)
    coefficients = load_coefficients(args.coefficients, n_factors=len(args.factors))
    fixed = _parse_fixed(args.fixed)
    for name in list(args.vary) + list(fixed):
        if name not in args.factors:
            raise KeyError(f"{name!r} is not one of the factors {args.factors}")
    if args.vary[0] == args.vary[1]:
        raise ValueError("--vary needs two different factors")

    optimum = _optimum(args, data_df, coefficients) if args.optimum else None
    folder = None if args.show else Path(args.out or DEFAULT_OUT)

    paths = []
    for kind, function in (("surface", surface_plot), ("contour", contour_plot)):
        if args.kind not in (kind, "both"):
            continue
        options: dict[str, Any] = dict(
            coefficients=coefficients,
            factors=args.factors,
            vary=tuple(args.vary),
            data=None if args.no_points else data_df,
            response=args.response,
            fixed=fixed or None,
            optimum=optimum,
            folder=folder,
            dpi=args.dpi,
        )
        if args.extend is not None:
            options["extend"] = args.extend
        if kind == "surface":
            options["view"] = (args.elev, args.azim)
        if args.no_points:  # Limits and centre point still come from the data
            options["fixed"], options["limits"] = _data_defaults(
                data_df, args, kind, fixed
            )
        paths.append(function(**options))

    for path in paths:
        if path is not None:
            print(f"Saved {path}")


########################################################################
# MAIN
########################################################################


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``rsm-design``.

    params:
        argv (list): arguments without the program name; None = sys.argv
    return:
        code (int): 0 on success, 2 on a user error (message on stderr)
    """
    args = build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        level = logging.WARNING
    enable_console_logging(level)

    if not (args.command == "plot" and args.show):
        import matplotlib

        matplotlib.use("Agg")  # No windows unless --show

    runners = {"fit": _run_fit, "optimize": _run_optimize, "plot": _run_plot}
    try:
        runners[args.command](args)
    except (RankDeficientDesignError, KeyError, ValueError, FileNotFoundError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 2
    return 0
