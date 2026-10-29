"""Shared fixtures for the rsm_design tests."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No windows while testing

import pytest

ROOT = Path(__file__).resolve().parents[1]
DLLME_XLSX = ROOT / "data" / "dllme" / "experimental_results.xlsx"


@pytest.fixture(scope="session")
def dllme_xlsx() -> Path:
    """Path of the public DLLME workbook; skip the test if it is missing."""
    if not DLLME_XLSX.exists():
        pytest.skip(f"{DLLME_XLSX} not found")
    return DLLME_XLSX
