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

# Load the training datasets
cd8_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivocd8cells.csv',usecols=[8,9,22,28,34,40,52])
cd4_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivocd4cells.csv',usecols=[8,9,22,28,34,40,52])
mhcII_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivomhccells.csv',usecols=[8,9,22,28,34,40,52])
pd1_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivopd1cells.csv',usecols=[8,9,22,28,34,40,52])
pd1tcf_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivopd1tcfcells.csv',usecols=[8,9,22,28,34,40,52])
tcf_CC01_PreNivo = pd.read_csv('data/CC01_PreNivo/CC01_PreNivotcfcells.csv',usecols=[8,9,22,28,34,40,52])

cd8_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivocd8cells.csv',usecols=[8,9,22,28,34,40,52])
cd4_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivocd4cells.csv',usecols=[8,9,22,28,34,40,52])
mhcII_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivomhccells.csv',usecols=[8,9,22,28,34,40,52])
pd1_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivopd1cells.csv',usecols=[8,9,22,28,34,40,52])
pd1tcf_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivopd1tcfcells.csv',usecols=[8,9,22,28,34,40,52])
tcf_CC09_PreNivo = pd.read_csv('data/CC09_PreNivo/CC09_PreNivotcfcells.csv',usecols=[8,9,22,28,34,40,52])

print("Model data loaded")

# add cell class column
cd8_CC01_PreNivo['cellclass'] = 10
cd4_CC01_PreNivo['cellclass'] = 20
mhcII_CC01_PreNivo['cellclass'] = 30
pd1_CC01_PreNivo['cellclass'] = 40
pd1tcf_CC01_PreNivo['cellclass'] = 50
tcf_CC01_PreNivo['cellclass'] = 60

cd8_CC09_PreNivo['cellclass'] = 10
cd4_CC09_PreNivo['cellclass'] = 20
mhcII_CC09_PreNivo['cellclass'] = 30
pd1_CC09_PreNivo['cellclass'] = 40
pd1tcf_CC09_PreNivo['cellclass'] = 50
tcf_CC09_PreNivo['cellclass'] = 60

# concatenate all the dataframes
frames = [cd8_CC01_PreNivo,cd4_CC01_PreNivo,mhcII_CC01_PreNivo,pd1_CC01_PreNivo,pd1tcf_CC01_PreNivo,tcf_CC01_PreNivo] 
        #   cd8_CC09_PreNivo,cd4_CC09_PreNivo,mhcII_CC09_PreNivo,pd1_CC09_PreNivo,pd1tcf_CC09_PreNivo,tcf_CC09_PreNivo]
data = pd.concat(frames)
print(data.head())
data.sample(frac=1,ignore_index=True) #shuffle the data

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of models to evaluate
print("Modelling")

models = [

    # Linear Models
    ('Logistic Regression', LogisticRegression(solver='lbfgs', max_iter=1000)),
    ('Linear Discriminant Analysis (LDA)', LinearDiscriminantAnalysis()),
    ('Ridge Classifier', RidgeClassifier()),
    ('Stochastic Gradient Descent (SGD)', SGDClassifier(loss='log_loss')),

    # Tree-Based Models
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    # ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),
    ('LightGBM', LGBMClassifier()),
    ('Extra Trees', ExtraTreesClassifier()),

    # Support Vector Machines
    ('SVC', SVC(probability=True)),
    ('Linear SVC', LinearSVC()),

    # Nearest Neighbors
    ('K-Nearest Neighbors (KNN)', KNeighborsClassifier()),

    # Neural Networks
    ('Multi-Layer Perceptron (MLP)', MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)),

    # Ensemble Methods
    ('AdaBoost', AdaBoostClassifier()),
    ('Bagging', BaggingClassifier()),
    ('Voting Classifier', VotingClassifier(estimators=[
        ('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svc', SVC(probability=True))
    ])),
    ('Stacking Classifier', StackingClassifier(estimators=[
        ('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svc', SVC(probability=True))
    ], final_estimator=LogisticRegression())),

    # Probabilistic Models
    ('Gaussian Naive Bayes', GaussianNB()),
    ('Multinomial Naive Bayes', MultinomialNB()),
    ('Bernoulli Naive Bayes', BernoulliNB()),

    # Clustering-Based Models
    ('K-Means', KMeans(n_clusters=3)),
    ('Gaussian Mixture Model (GMM)', GaussianMixture(n_components=3)),

    # Other Models
    ('Quadratic Discriminant Analysis (QDA)', QuadraticDiscriminantAnalysis()),
    ('Partial Least Squares (PLS)', PLSRegression(n_components=2))

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

print(f"\nBest Model: {best_model} with Accuracy: {best_score:.4f}")

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
model_filename = 'best_model_cc.pkl'
joblib.dump(best_pipeline, model_filename)
print(f"Best model saved to {model_filename}")

# Best model saved to best_model.pkl