import csv

# Specify the path to your .txt file
txt_file_path = (
    r"C:\Users\marco\python-projects\rsm_design\Final results - REC\model_stats.txt"
)

# Read the content from the .txt file
with open(txt_file_path, "r") as txt_file:
    ols_results_text = txt_file.read()

# Create a CSV file and write the text content
csv_file_path = "ols_results.csv"
with open(csv_file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    for line in ols_results_text.split("\n"):
        csv_writer.writerow([line])

print(f"The results have been saved to {csv_file_path}")
