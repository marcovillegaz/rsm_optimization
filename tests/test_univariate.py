"""Tests of the univariate ANOVA, Tukey HSD and box plot."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rsm_design.univariate import (
    UnivariateResult,
    group_boxplot,
    load_groups,
    one_way_anova,
)


@pytest.fixture
def groups_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for label, mean in (("A", 10.0), ("B", 20.0), ("C", 30.0)):
        for value in rng.normal(mean, 1.0, 4):
            rows.append((label, value))
    return pd.DataFrame(rows, columns=["Group", "Response"])


def write_sheet(path, extra_row=None):
    rows = [
        ["Screening", None, None],
        ["Group name", "Other", "Resp "],
        ["Notation", "-", "-"],
        ["Units", "-", "ppm"],
        ["x", "g1", 1.0],
        [None, "g1", 2.0],
        [None, "g2", 5.0],
        [None, "g2", 6.0],
    ]
    if extra_row is not None:
        rows.append(extra_row)
    pd.DataFrame(rows).to_excel(path, sheet_name="S", header=False, index=False)


def test_anova_significant_and_tukey_pairs(groups_df):
    result = one_way_anova(df=groups_df)
    assert isinstance(result, UnivariateResult)
    assert result.anova_table.loc["C(Group)", "PR(>F)"] < 0.05
    tukey_df = result.tukey_frame()
    assert len(tukey_df) == 3  # 3 groups -> 3 pairs
    assert bool(tukey_df["reject"].all())


def test_load_groups_synthetic(tmp_path):
    path = tmp_path / "book.xlsx"
    write_sheet(path)
    data_df = load_groups(
        path=path,
        sheet="S",
        group_column="Other",
        response_column="Resp ",
        header_row=1,
        skip_rows=2,
    )
    assert list(data_df.columns) == ["Group", "Response"]
    assert data_df["Group"].tolist() == ["g1", "g1", "g2", "g2"]
    assert data_df["Response"].tolist() == [1.0, 2.0, 5.0, 6.0]


def test_load_groups_drops_non_numeric_with_warning(tmp_path, caplog):
    path = tmp_path / "book.xlsx"
    write_sheet(path, extra_row=[None, "g2", "n/a"])
    with caplog.at_level(logging.WARNING, logger="rsm_design"):
        data_df = load_groups(
            path=path, sheet="S", group_column="Other", response_column="Resp "
        )
    assert len(data_df) == 4
    assert "dropped" in caplog.text


def test_load_groups_missing_column_lists_available(tmp_path):
    path = tmp_path / "book.xlsx"
    write_sheet(path)
    with pytest.raises(KeyError, match="Available columns"):
        load_groups(path=path, sheet="S", group_column="Nope", response_column="Resp ")


def test_load_groups_real_file(dllme_xlsx):
    data_df = load_groups(
        path=dllme_xlsx,
        sheet="Screening_3",
        group_column="Sample name",
        response_column="Enrichment factor 260",
    )
    assert len(data_df) > 12
    assert data_df["Response"].dtype.kind == "f"
    result = one_way_anova(df=data_df.iloc[0:12])
    assert result.anova_table.loc["C(Group)", "df"] == 3


def test_group_boxplot_saves_and_closes(tmp_path, groups_df):
    path = group_boxplot(
        df=groups_df,
        color="tab:blue",
        xlabel="Group",
        ylabel="EF",
        panel_label="(a)",
        folder=tmp_path / "out",
        file_name="box.png",
        dpi=50,
    )
    assert path is not None and path.exists() and path.stat().st_size > 0
    assert plt.get_fignums() == []
