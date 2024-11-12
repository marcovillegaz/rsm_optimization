import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "cosmo_screening_result\sigma_profiles.xlsx"
sheet_name = "Hoja2"  # Replace with your actual sheet name if different
results_folder = "cosmo_screening_result"
molecules_list = ["Water", "PCB 77", "Lid:Meth(b)", "LacA:ChCl"]

# Read the relevant part of the Excel file
sigma_profile_df = pd.read_excel(file_path, sheet_name=sheet_name)
print(sigma_profile_df)

# Create Figure
plt.figure(figsize=(7, 6))

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
plt.xlabel("$\sigma (e/A^{2})$")
plt.ylabel("$p(\sigma)$")
plt.legend()
plt.tight_layout()
# Show the plot
plt.savefig(os.path.join(results_folder, "sigma_profile.png"), dpi=800)
