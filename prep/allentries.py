import os
import pandas as pd

# Base directory to start the search
base_directory = "./predictions"

# all predictions independent of file
all_predictions = pd.DataFrame()

# Traverse the directory
for root, dirs, files in os.walk(base_directory):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        # Check if the file exists
        if os.path.isfile(file_path):
            try:
                # Read the CSV file
                data = pd.read_csv(file_path)
                print(file_name, data.shape[0])
                all_predictions = pd.concat([all_predictions, data], ignore_index=True)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

print('total rows', all_predictions.shape[0])
grouped = all_predictions.groupby('Predictions').size()
print(grouped)  