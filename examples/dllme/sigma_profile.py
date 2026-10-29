"""This script plots the sigma profiles of water, PCB 77 and selected HDES
obtained from the COSMO-RS calculations"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sigma_file = repo_root / "data" / "dllme" / "cosmo" / "sigma_profiles.xlsx"
results_folder = Path(__file__).parent / "results" / "cosmo"
sheet_name = "Hoja2"  # Replace with your actual sheet name if different
molecules_list = ["Water", "PCB 77", "Lid:Meth(b)", "LacA:ChCl"]
DPI = 800

results_folder.mkdir(parents=True, exist_ok=True)

# Read the relevant part of the Excel file
sigma_profile_df = pd.read_excel(sigma_file, sheet_name=sheet_name)
print(sigma_profile_df)

# Create Figure
fig = plt.figure(figsize=(7, 6))

# Adding vertical lines at -0.01 and +0.01
plt.axvline(x=-0.01, color="k", linestyle="--", linewidth=1)
plt.axvline(x=0.01, color="k", linestyle="--", linewidth=1)

# Plotting Sigma vs one of the notations (e.g., p(1))
for molecule in molecules_list:
    plt.plot(
        sigma_profile_df["sigma"] * 0.01,
        sigma_profile_df[molecule],
        linestyle="-",
        label=molecule,
    )

# Set limits
plt.xlim((-0.03, 0.03))
plt.ylim((0, 50))

# Adding labels and title
plt.xlabel(r"$\sigma (e/A^{2})$")
plt.ylabel(r"$p(\sigma)$")
plt.legend()
plt.tight_layout()

# Save the plot as a PNG image
fig.savefig(results_folder / "sigma_profile.png", dpi=DPI)
plt.close(fig)
