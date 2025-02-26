import pandas as pd
import os

base_directory = "./data/30810_2"


file_path = os.path.join(base_directory, '30810_2a.csv')
file1 = pd.read_csv(file_path)
print(file1.shape[0])

file_path = os.path.join(base_directory, '30810_2b.csv')
file2 = pd.read_csv(file_path)
print(file2.shape[0])
new_file = pd.concat([file1, file2], axis=1)
new_file.to_csv('data/30810_2/30810_2.csv', index=False)
