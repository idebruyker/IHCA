import os
import pandas as pd
import joblib  # For loading the saved model

# Load the saved model
model_filename = 'best_model.pkl'
loaded_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Load data from multiple files
base_directory = "./data"
for root, dirs, files in os.walk(base_directory):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        file_path = f"{dir_path}/{dir_name}.csv"
        if os.path.isfile(file_path):
            print(dir_name, dir_path, file_path)
            data = pd.read_csv(file_path,usecols=[7,8,21,27,33,39,51]) #X Y measurement columns and mean columns
            data.rename(columns={'Centroid X µm': 'Centroid.X.µm', 
                                 'Centroid Y µm': 'Centroid.Y.µm', 
                                 'Nucleus: Opal 480 mean': 'Nucleus..Opal.480.mean', 
                                 'Nucleus: Opal 520 mean': 'Nucleus..Opal.520.mean', 
                                 'Nucleus: Opal 570 mean': 'Nucleus..Opal.570.mean', 
                                 'Nucleus: Opal 620 mean': 'Nucleus..Opal.620.mean', 
                                 'Nucleus: Opal 690 mean': 'Nucleus..Opal.690.mean'}, 
                                 inplace=True)
            print(data.head())
            # Make predictions on the new dataset
            predictions = loaded_model.predict(data)
            # Add predictions to the new dataset 
            data['Predictions'] = predictions
            print(data.head())
            # Save the predictions to a new file 
            outputfile_name = f"./predictions/{dir_name}_predictions.csv"
            data.to_csv(outputfile_name, index=False)
            print(f"Predictions saved to {outputfile_name}")
        else:
            print(f"'{file_path}' does not exist.")



