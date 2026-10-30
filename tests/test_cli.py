"""Tests of the command line interface."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from rsm_design.cli import main

FACTORS = [
    "Extractant-Dispersant Ratio",
    "Sample volume",
    "Extraction mixture volume",
]
DROP = ["b2", "b12", "b13", "b23", "b22", "b33"]


def data_args(xlsx: Path) -> list[str]:
    """Arguments that read the DLLME workbook."""
    return [
        str(xlsx),
        "--sheet",
        "RSM_3",
        "--header-row",
        "1",
        "--skip-rows",
        "2",
        "--factors",
        *FACTORS,
        "--response",
        "Enrichment Factor",
    ]


@pytest.fixture
def fitted(dllme_xlsx, tmp_path):
    """Run `fit` once and return the results folder."""
    code = main(
        ["fit", *data_args(dllme_xlsx), "--drop", *DROP, "--out", str(tmp_path)]
    )
    assert code == 0
    return tmp_path


def test_fit_writes_files(fitted):
    for name in ("model_params.csv", "model_stats.txt", "coeff_plot.png"):
        assert (fitted / f"EnrichmentFactor_{name}").exists()
    coeff = pd.read_csv(fitted / "EnrichmentFactor_model_params.csv", header=None)
    values = dict(zip(coeff[0], coeff[1], strict=True))
    assert values["b0"] == pytest.approx(77.04742942964698, rel=1e-6)
    assert values["b1"] == pytest.approx(-47.02551840202079, rel=1e-6)
    assert values["b3"] == pytest.approx(-0.04967289354553598, rel=1e-6)
    assert values["b11"] == pytest.approx(13.649456917604395, rel=1e-6)


def test_optimize_reads_coefficients(fitted, dllme_xlsx, tmp_path, capsys):
    params = fitted / "EnrichmentFactor_model_params.csv"
    out = tmp_path / "opt"
    code = main(
        ["optimize", *data_args(dllme_xlsx), "--coefficients", str(params)]
        + ["--out", str(out)]
    )
    assert code == 0
    assert "predicted Enrichment Factor" in capsys.readouterr().out
    rows = pd.read_csv(out / "EnrichmentFactor_optimum.csv", header=None)
    assert list(rows[0]) == FACTORS + ["Enrichment Factor"]


def test_optimize_without_out_writes_nothing(fitted, dllme_xlsx, tmp_path, monkeypatch):
    params = fitted / "EnrichmentFactor_model_params.csv"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    args = ["optimize", *data_args(dllme_xlsx), "--coefficients", str(params)]
    assert main(args + ["--minimize", "--region", "box", "--center", "midrange"]) == 0
    assert list(work.iterdir()) == []


def test_plot_surface_and_contour(fitted, dllme_xlsx, tmp_path):
    params = fitted / "EnrichmentFactor_model_params.csv"
    out = tmp_path / "plots"
    code = main(
        ["plot", *data_args(dllme_xlsx), "--coefficients", str(params)]
        + ["--vary", FACTORS[0], FACTORS[2], "--fixed", "Sample volume=9"]
        + ["--optimum", "--out", str(out), "--dpi", "50"]
    )
    assert code == 0
    assert (out / "EnrichmentFactor_surface.png").exists()
    assert (out / "EnrichmentFactor_contour.png").exists()


def test_plot_no_points_single_kind(fitted, dllme_xlsx, tmp_path):
    params = fitted / "EnrichmentFactor_model_params.csv"
    out = tmp_path / "plots"
    code = main(
        ["plot", *data_args(dllme_xlsx), "--coefficients", str(params)]
        + ["--vary", FACTORS[0], FACTORS[1], "--kind", "contour", "--no-points"]
        + ["--out", str(out), "--dpi", "50"]
    )
    assert code == 0
    assert (out / "EnrichmentFactor_contour.png").exists()
    assert not (out / "EnrichmentFactor_surface.png").exists()


def test_bad_factor_name(dllme_xlsx, tmp_path, capsys):
    args = data_args(dllme_xlsx)
    args[args.index("--factors") + 1] = "No such factor"
    assert main(["fit", *args, "--out", str(tmp_path)]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_missing_file(tmp_path, capsys):
    code = main(["fit", str(tmp_path / "no.xlsx"), "--factors", "a", "--response", "y"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_bad_fixed_option(fitted, dllme_xlsx):
    params = fitted / "EnrichmentFactor_model_params.csv"
    args = ["plot", *data_args(dllme_xlsx), "--coefficients", str(params)]
    assert main(args + ["--vary", FACTORS[0], FACTORS[1], "--fixed", "oops"]) == 2


@pytest.fixture
def duplicated_factorial(tmp_path) -> Path:
    """A 2^2 factorial run twice: it cannot estimate b11 and b22."""
    rows = [(-1, -1), (1, -1), (-1, 1), (1, 1)] * 2
    data_df = pd.DataFrame(rows, columns=["A", "B"])
    data_df["y"] = [1, 2, 3, 5, 1.1, 2.1, 2.9, 5.2]
    path = tmp_path / "factorial.csv"
    data_df.to_csv(path, index=False)
    return path


def test_rank_deficient_design(duplicated_factorial, tmp_path, capsys):
    base = ["fit", str(duplicated_factorial), "--factors", "A", "B", "--response", "y"]
    base += ["--out", str(tmp_path / "out"), "--no-plots"]
    assert main(base) == 2
    err = capsys.readouterr().err
    assert "b11" in err and "b22" in err

    with pytest.warns(UserWarning):
        assert main(base + ["--allow-rank-deficient"]) == 0
    assert main(base + ["--drop", "b11", "b22"]) == 0


def test_module_help():
    run = subprocess.run(
        [sys.executable, "-m", "rsm_design", "--help"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0
    assert "fit" in run.stdout and "optimize" in run.stdout and "plot" in run.stdout
