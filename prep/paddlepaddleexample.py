# Step 1: Install PaddlePaddle (if not already installed)
# Run this command in your terminal or notebook:
# !pip install paddlepaddle

# Step 2: Import Libraries
import paddle
import paddle.nn.functional as F
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

# Convert data to Paddle tensors
X_train = paddle.to_tensor(X_train, dtype='float32')
y_train = paddle.to_tensor(y_train, dtype='int64')
X_test = paddle.to_tensor(X_test, dtype='float32')
y_test = paddle.to_tensor(y_test, dtype='int64')

# Step 4: Define the Model
class MultiClassModel(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassModel, self).__init__()
        self.fc1 = paddle.nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = paddle.nn.Linear(hidden_size, hidden_size)  # Hidden layer to hidden layer
        self.fc3 = paddle.nn.Linear(hidden_size, num_classes)  # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for the first hidden layer
        x = F.relu(self.fc2(x))  # Activation function for the second hidden layer
        x = self.fc3(x)          # Output layer (no activation function, softmax will be applied in loss)
        return x

# Step 5: Initialize the Model
input_size = 7  # Number of features
hidden_size = 10  # Number of neurons in the hidden layer
num_classes = len(np.unique(y))  # Number of unique classes
model = MultiClassModel(input_size, hidden_size, num_classes)

# Step 6: Define the Loss Function and Optimizer
loss_fn = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.01)

# Step 7: Train the Model
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.numpy()[0]:.4f}")

# Step 8: Evaluate the Model
# Set the model to evaluation mode
model.eval()

# Disable gradient computation for evaluation
with paddle.no_grad():
    # Forward pass on the test set
    y_test_pred = model(X_test)
    y_test_pred = paddle.argmax(y_test_pred, axis=1)  # Convert logits to class labels

    # Calculate accuracy
    accuracy = paddle.metric.accuracy(y_test_pred, y_test)
    print(f"Test Accuracy: {accuracy.numpy()[0] * 100:.2f}%")