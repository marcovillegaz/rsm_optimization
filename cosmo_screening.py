import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def cosmos_screening(file_path, save_path="None"):
    """DESCRIPTION"""
    df = pd.read_csv(file_path, sep=";", decimal=",")  # Read CSV file into a DataFrame
    print(df)
    df.set_index("DES", inplace=True)  # Set the "indexes" column as the index
    print(df)  # Display the DataFrame

    # Plotting
    ax = df.plot(
        kind="bar",
        figsize=(5, 6),
        rot=0,
        # colormap="Dark2",
        width=0.8,
    )

    # Set axis labels and title
    ax.set_xlabel("Polymers")
    ax.set_ylabel(r"$Ln(\gamma)$")
    ax.grid(linewidth=0.5)

    # Annotate each bar with its numeric value
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            label_type="edge",
            fontsize=8,
            color="black",
            # weight="bold",
            rotation=0,
        )

    plt.xticks(rotation=45)  # Rotate x-axis labels vertically
    # Adjust layout
    plt.tight_layout()

    # Show or save image
    if save_path == "None":
        plt.show()
    else:
        save_path = os.path.join(save_path, "COSMOS_bar_plot.png")
        print("\nSaving in plot in: ", save_path)
        plt.savefig(save_path, dpi=800)

    # plt.close("all")


# Replace 'your_file.csv' with the actual path to your CSV file
file_path = r"C:\Users\marco\OneDrive - usach.cl\DLLME of PCB77 employing designed DES\Hojas de calculo\COSMO_screening.csv"

cosmos_screening(
    file_path,
    save_path=r"C:\Users\marco\OneDrive - usach.cl\DLLME of PCB77 employing designed DES\images",
)
