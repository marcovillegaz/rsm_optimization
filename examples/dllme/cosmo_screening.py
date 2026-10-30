"""This script create a bar plot with the result obtaining from the COSMO-RS
screening between PCB77 and different HDES"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

repo_root = Path(__file__).resolve().parents[2]
screening_file = repo_root / "data" / "dllme" / "cosmo" / "COSMO_screening.csv"
results_folder = Path(__file__).parent / "results" / "cosmo"
DPI = 800

results_folder.mkdir(parents=True, exist_ok=True)

# Read CSV file into a DataFrame
df = pd.read_csv(screening_file, sep=";", decimal=",", encoding="utf-8-sig")
# Set the "indexes" column as the index
df = df.set_index("DES")
# Display the DataFrame
print(df)

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 6))

# Plot the bar plot
color = "tab:blue"
ax1.set_xlabel("DES")
ax1.set_ylabel("PCB77", color=color)
ax1.bar(df.index, df["PCB77"], color=color, alpha=0.6, label="PCB77")
ax1.set_ylabel(r"$Ln(\gamma_{PCB77})$")
ax1.tick_params(axis="y", labelcolor=color)

# Create another y-axis for the scatter plot
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Water", color=color)
ax2.scatter(df.index, df["Water"], color=color, marker="s", label="Water")
ax2.set_ylabel(r"$Ln(\gamma_{water})$")
ax2.tick_params(axis="y", labelcolor=color)

# Add horizontal line to the primary y-axis
ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

# Align the y-axes
ax1.set_ylim([-2, 8])
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax2.set_ylim([-2, 8])
ax2.yaxis.set_major_locator(MultipleLocator(1))

# Combine legends from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Create a combined legend
fig.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    bbox_to_anchor=(0.11, 0.97),
    frameon=False,
)

ax1.set_xticks(range(len(df.index)))
ax1.set_xticklabels(df.index, rotation=45, ha="right")
fig.tight_layout()

# Save the plot as a JPEG image
fig.savefig(results_folder / "COSMO_RS_screening.jpeg", dpi=DPI)
plt.close(fig)
