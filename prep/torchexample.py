# Step 1: Install PyTorch (if not already installed)
# Run this command in your terminal or notebook:
# !pip install torch

# Step 2: Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Step 3: Load and Preprocess the Dataset

# Load the training datasets
cd8_53904_1 = pd.read_csv('data/53904_1/53904_1cd8cells.csv',usecols=[8,9,22,28,34,40,52])
cd4_53904_1 = pd.read_csv('data/53904_1/53904_1cd4cells.csv',usecols=[8,9,22,28,34,40,52])
mhcII_53904_1 = pd.read_csv('data/53904_1/53904_1mhccells.csv',usecols=[8,9,22,28,34,40,52])
pd1_53904_1 = pd.read_csv('data/53904_1/53904_1pd1cells.csv',usecols=[8,9,22,28,34,40,52])
pd1tcf_53904_1 = pd.read_csv('data/53904_1/53904_1pd1tcfcells.csv',usecols=[8,9,22,28,34,40,52])
tcf_53904_1 = pd.read_csv('data/53904_1/53904_1tcfcells.csv',usecols=[8,9,22,28,34,40,52])

cd8_33683_1 = pd.read_csv('data/33683_1/33683_1cd8cells.csv',usecols=[8,9,22,28,34,40,52])
cd4_33683_1 = pd.read_csv('data/33683_1/33683_1cd4cells.csv',usecols=[8,9,22,28,34,40,52])
mhcII_33683_1 = pd.read_csv('data/33683_1/33683_1mhccells.csv',usecols=[8,9,22,28,34,40,52])
pd1_33683_1 = pd.read_csv('data/33683_1/33683_1pd1cells.csv',usecols=[8,9,22,28,34,40,52])
pd1tcf_33683_1 = pd.read_csv('data/33683_1/33683_1pd1tcfcells.csv',usecols=[8,9,22,28,34,40,52])
tcf_33683_1 = pd.read_csv('data/33683_1/33683_1tcfcells.csv',usecols=[8,9,22,28,34,40,52])

cd8_33270_2 = pd.read_csv('data/33270_2/33270_2cd8cells.csv',usecols=[8,9,22,28,34,40,52])
cd4_33270_2 = pd.read_csv('data/33270_2/33270_2cd4cells.csv',usecols=[8,9,22,28,34,40,52])
mhcII_33270_2 = pd.read_csv('data/33270_2/33270_2mhccells.csv',usecols=[8,9,22,28,34,40,52])
pd1_33270_2 = pd.read_csv('data/33270_2/33270_2pd1cells.csv',usecols=[8,9,22,28,34,40,52])
pd1tcf_33270_2 = pd.read_csv('data/33270_2/33270_2pd1tcfcells.csv',usecols=[8,9,22,28,34,40,52])
tcf_33270_2 = pd.read_csv('data/33270_2/33270_2tcfcells.csv',usecols=[8,9,22,28,34,40,52])

print("Model data loaded")

# add cell class column
cd8_53904_1['cellclass'] = 10
cd4_53904_1['cellclass'] = 20
mhcII_53904_1['cellclass'] = 30
pd1_53904_1['cellclass'] = 40
pd1tcf_53904_1['cellclass'] = 50
tcf_53904_1['cellclass'] = 60

cd8_33683_1['cellclass'] = 10
cd4_33683_1['cellclass'] = 20
mhcII_33683_1['cellclass'] = 30
pd1_33683_1['cellclass'] = 40
pd1tcf_33683_1['cellclass'] = 50
tcf_33683_1['cellclass'] = 60

cd8_33270_2['cellclass'] = 10
cd4_33270_2['cellclass'] = 20
mhcII_33270_2['cellclass'] = 30
pd1_33270_2['cellclass'] = 40
pd1tcf_33270_2['cellclass'] = 50
tcf_33270_2['cellclass'] = 60

# concatenate all the dataframes
frames = [cd8_53904_1,cd4_53904_1,mhcII_53904_1,pd1_53904_1,pd1tcf_53904_1,tcf_53904_1, 
          cd8_33683_1,cd4_33683_1,mhcII_33683_1,pd1_33683_1,pd1tcf_33683_1,tcf_33683_1,
          cd8_33270_2,cd4_33270_2,mhcII_33270_2,pd1_33270_2,pd1tcf_33270_2,tcf_33270_2]
data = pd.concat(frames)
data.sample(frac=1,ignore_index=True) #shuffle the data
print(data.shape[0])

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last one are features
y = data.iloc[:, -1].values   # The last column is the label

# Encode labels (convert string labels to integers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 4: Define the Model
class MultiClassModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for the first hidden layer
        x = torch.relu(self.fc2(x))  # Activation function for the second hidden layer
        x = self.fc3(x)              # Output layer (no activation function, softmax will be applied in loss)
        return x

# Step 5: Initialize the Model
input_size = 7  # Number of features
hidden_size = 10  # Number of neurons in the hidden layer
num_classes = len(np.unique(y))  # Number of unique classes
model = MultiClassModel(input_size, hidden_size, num_classes)

# Step 6: Define the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 7: Train the Model
epochs = 200
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Step 8: Evaluate the Model
# Set the model to evaluation mode
model.eval()

# Disable gradient computation for evaluation
with torch.no_grad():
    # Forward pass on the test set
    y_test_pred = model(X_test)
    _, y_test_pred = torch.max(y_test_pred, 1)  # Convert logits to class labels

    # Calculate accuracy
    accuracy = (y_test_pred == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")