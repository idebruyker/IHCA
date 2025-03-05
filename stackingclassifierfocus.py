import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib  # For saving the model
from datetime import datetime

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

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of models to evaluate
print("Modelling")

models = [

    ('Stacking Classifier', StackingClassifier(estimators=[
        ('lr', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)), ('rf', GradientBoostingClassifier(random_state=42)), ('svc', SVC(probability=True))
    ], final_estimator=LogisticRegression()))
    ]

# Evaluate each model using cross-validation
best_model = None
best_score = -np.inf

for name, model in models:
    # Create a pipeline with StandardScaler for models that require scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    print(f'{name}: {datetime.now().strftime("%H:%M:%S")}: Evaluating...')
    try:
        # Perform cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        # Calculate the mean score
        mean_score = np.mean(scores)
        
        print(f"{name}: Mean Cross-Validation Accuracy = {mean_score:.4f}")
        
        # Check if this model is the best so far
        if mean_score > best_score:
            best_score = mean_score
            best_model = name
    except Exception as e:
        print(f"{name} failed due to: {e}")

print(f"\n{datetime.now().strftime("%H:%M:%S")}: Best Model: {best_model} with Accuracy: {best_score:.4f}")

# Train the best model on the full training set and evaluate on the test set
best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', dict(models)[best_model])
])

best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

# Evaluate the best model on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with {best_model}: {test_accuracy:.4f}")

# Save the best model to a file
model_filename = 'stackingclassifier_model.pkl'
joblib.dump(best_pipeline, model_filename)
print(f"Best model saved to {model_filename}")

# Test Accuracy with Stacking Classifier: 0.8749