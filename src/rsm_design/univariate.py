"""One-way ANOVA with Tukey HSD and box plots for univariate screenings."""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import TukeyHSDResults
from statsmodels.stats.multicomp import pairwise_tukeyhsd

logger = logging.getLogger(__name__)

DPI = 800


@dataclass
class UnivariateResult:
    """ANOVA table and Tukey HSD comparison of one screening.

    params:
        anova_table (pd.DataFrame): type II ANOVA table of "Response ~ C(Group)"
        tukey (TukeyHSDResults): pairwise Tukey HSD comparison of the groups
    """

    anova_table: pd.DataFrame
    tukey: TukeyHSDResults

    def tukey_frame(self) -> pd.DataFrame:
        """Tukey HSD summary as a DataFrame, one row per pair of groups.

        params:
            None
        return:
            tukey_df (pd.DataFrame): group1, group2, meandiff, p-adj, lower, upper,
                reject
        """
        table = self.tukey.summary().data  # First row holds the column names
        tukey_df = pd.DataFrame(table[1:], columns=table[0])
        return tukey_df


def load_groups(
    path: str | Path,
    sheet: str,
    group_column: str,
    response_column: str,
    header_row: int = 1,
    skip_rows: int = 2,
) -> pd.DataFrame:
    """Read a screening sheet into a tidy table with the columns Group and Response.

    The sheet has a header row followed by rows with notation and units, which are
    not data and are dropped with skip_rows. Non numeric responses are discarded.

    params:
        path (str | Path): Excel workbook
        sheet (str): sheet name
        group_column (str): column with the group label of each experiment
        response_column (str): column with the measured response
        header_row (int): zero based row of the column names
        skip_rows (int): rows below the header that are not data (notation, units)
    return:
        data_df (pd.DataFrame): columns Group and Response, one row per experiment
    """
    raw_df = pd.read_excel(path, sheet_name=sheet, header=header_row)

    for name in (group_column, response_column):
        if name not in raw_df.columns:
            available = [str(col) for col in raw_df.columns]
            raise KeyError(
                f"Column {name!r} not found in sheet {sheet!r}. "
                f"Available columns: {available}"
            )

    raw_df = raw_df.iloc[skip_rows:]  # drop notation and units rows
    data_df = pd.DataFrame(
        {
            "Group": raw_df[group_column],
            "Response": pd.to_numeric(raw_df[response_column], errors="coerce"),
        }
    )

    n_total = len(data_df)
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)
    if len(data_df) < n_total:
        logger.warning(
            "%d of %d rows without a valid group or numeric response were dropped",
            n_total - len(data_df),
            n_total,
        )
    logger.info(
        "Loaded %d experiments in %d groups from sheet %s",
        len(data_df),
        data_df["Group"].nunique(),
        sheet,
    )
    return data_df


def one_way_anova(
    df: pd.DataFrame,
    group: str = "Group",
    response: str = "Response",
    alpha: float = 0.05,
) -> UnivariateResult:
    """One-way ANOVA followed by Tukey HSD to find which groups differ.

    params:
        df (pd.DataFrame): table with a group column and a response column
        group (str): name of the group column
        response (str): name of the response column
        alpha (float): significance level of the Tukey test
    return:
        result (UnivariateResult): ANOVA table and Tukey HSD result
    """
    data_df = pd.DataFrame(
        {"Group": df[group].to_numpy(), "Response": df[response].to_numpy()}
    )

    model = ols(formula="Response ~ C(Group)", data=data_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    logger.info("ANOVA table:\n%s", anova_table)

    tukey = pairwise_tukeyhsd(data_df["Response"], data_df["Group"], alpha=alpha)
    logger.info("Tukey HSD results:\n%s", tukey.summary())

    return UnivariateResult(anova_table=anova_table, tukey=tukey)


def group_boxplot(
    df: pd.DataFrame,
    color: str,
    xlabel: str,
    ylabel: str,
    panel_label: str | None = None,
    folder: str | Path | None = None,
    file_name: str = "boxplot.png",
    dpi: int = DPI,
) -> Path | None:
    """Box plot of the response of each group; saved if folder is given, else shown.

    params:
        df (pd.DataFrame): table with the columns Group and Response
        color (str): face colour of the boxes
        xlabel (str): label of the x axis
        ylabel (str): label of the y axis
        panel_label (str | None): panel tag such as "(a)" written below the axes
        folder (str | Path | None): output folder, created if missing; None shows
            the plot
        file_name (str): name of the image file inside folder
        dpi (int): resolution of the saved image
    return:
        path (Path | None): saved file, None when the plot was only shown
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # MAKING PLOT
    df.boxplot(
        column="Response",
        by="Group",
        ax=ax,
        patch_artist=True,
        boxprops=dict(facecolor=color, color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="white"),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(" ")
    if panel_label is not None:
        ax.text(
            0.475,
            -0.15,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )
    fig.suptitle("")
    fig.tight_layout()

    # SAVE OR SHOW
    path = None
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / file_name
        fig.savefig(path, dpi=dpi)
        logger.info("Box plot saved in %s", path)
    else:
        plt.show()
    plt.close(fig)
    return path
