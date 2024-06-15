import os
import re
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Define the path to the root folder
root_folder = r"C:\Users\40gil\Desktop\final_project\tensor_training\PostModelAnalyze\loaded_models_outputs\letters"

# Dictionary to store the data
data = {}

# Iterate through the folders
for date_folder in os.listdir(root_folder):
    date_path = os.path.join(root_folder, date_folder)
    if os.path.isdir(date_path):
        try:
            # Parse the date
            date = datetime.strptime(date_folder, "%d-%m-%Y")
        except ValueError:
            continue

        # Iterate through the model folders
        for model_folder in os.listdir(date_path):
            model_path = os.path.join(date_path, model_folder)
            if os.path.isdir(model_path):
                # Extract the model name up to the size number
                match = re.match(r"^(.*?bs\d+ts\d+X\d+)", model_folder)
                if match:
                    model_name = match.group(1)
                else:
                    continue

                # Read the accuracy from acc.txt
                acc_file_path = os.path.join(model_path, "acc.txt")
                if os.path.isfile(acc_file_path):
                    with open(acc_file_path, "r") as acc_file:
                        try:
                            accuracy = float(acc_file.read().strip())
                        except ValueError:
                            continue

                # Store the data
                if model_name not in data:
                    data[model_name] = []
                data[model_name].append((date, accuracy))

# Plot the data
plt.figure(figsize=(12, 8))

# Initialize lists to store all dates and accuracies
all_dates = []
all_accuracies = []

for model_name, values in data.items():
    values.sort()
    dates = [v[0] for v in values]
    accuracies = [v[1] for v in values]
    all_dates.extend(dates)
    all_accuracies.extend(accuracies)
    m_size = accuracies[0] * 70
    if m_size > 40:
        m_size = 40
    elif m_size < 10:
        m_size = 10

    # Plot the points
    plt.plot(dates, accuracies, marker='o', markersize=m_size, label=model_name)

    # Add model name as annotation for points with accuracy > 0.55
    for date, acc in zip(dates, accuracies):
        if acc > 0.55:
            plt.text(date, acc, f'{acc:.2f}', fontsize=12, ha='center', va='bottom', color='white', weight='bold')
            plt.text(date, acc, model_name, fontsize=12, ha='center', va='top', color='black')
        # if acc > 0.55:
        #     plt.text(date, acc, model_name, fontsize=15, ha='right', va='bottom')
        #     plt.text(date, acc, f'{acc:.2f}', fontsize=12, ha='right', va='bottom')


# Determine plot range based on all dates
min_date = min(all_dates)
max_date = max(all_dates)

# Set x-axis ticks to be every 4 days
plt.yticks(fontsize=15)
plt.xticks(pd.date_range(start=min_date, end=max_date, freq='4D'), rotation=45, fontsize=15)

plt.xlabel('Date', fontsize=25)
plt.ylabel('Accuracy', fontsize=25)
plt.title('Model Accuracy Over Time', fontsize=25)
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.tight_layout()

# Add legend outside the plot

plt.show()
