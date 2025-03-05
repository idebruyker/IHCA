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
    # ('Random Forest', RandomForestClassifier(random_state=42)),
    # ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    # ('Logistic Regression', LogisticRegression(max_iter=1000)),
    # ('SVM', SVC(kernel='linear')),
    # ('KNN', KNeighborsClassifier()),
    # ('Decision Tree', DecisionTreeClassifier(random_state=42))

################
    # Linear Models
    ('Logistic Regression', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)),
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
    ################
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

#############################################################
# Results based upon CD8, CD4, MHCII data sets

# Random Forest: Mean Cross-Validation Accuracy = 0.9529
# Gradient Boosting: Mean Cross-Validation Accuracy = 0.9653
# Logistic Regression: Mean Cross-Validation Accuracy = 0.7983
# SVM: Mean Cross-Validation Accuracy = 0.8185
# KNN: Mean Cross-Validation Accuracy = 0.9521
# Decision Tree: Mean Cross-Validation Accuracy = 0.9499

# Best Model: Gradient Boosting with Accuracy: 0.9653

#############################################################
# Results based upon CD8, CD4, MHCII, PD1, PD1TCF, TCF data sets

# Random Forest: Mean Cross-Validation Accuracy = 0.6953
# Gradient Boosting: Mean Cross-Validation Accuracy = 0.7793
# Logistic Regression: Mean Cross-Validation Accuracy = 0.6341
# SVM: Mean Cross-Validation Accuracy = 0.6698
# KNN: Mean Cross-Validation Accuracy = 0.7276
# Decision Tree: Mean Cross-Validation Accuracy = 0.6924

# Best Model: Gradient Boosting with Accuracy: 0.7793
# Test Accuracy with Gradient Boosting: 0.7807

#############################################################
# Results based upon CD8, CD4, MHCII, PD1, PD1TCF, TCF data sets with enhanced data cleanup classification - TCF removed

# Random Forest: Mean Cross-Validation Accuracy = 0.9514
# Gradient Boosting: Mean Cross-Validation Accuracy = 0.9610
# Logistic Regression: Mean Cross-Validation Accuracy = 0.7424
# SVM: Mean Cross-Validation Accuracy = 0.7849
# KNN: Mean Cross-Validation Accuracy = 0.9224
# Decision Tree: Mean Cross-Validation Accuracy = 0.9460

# Best Model: Gradient Boosting with Accuracy: 0.9610
# Test Accuracy with Gradient Boosting: 0.9607

#############################################################
# Results based upon CD8, CD4, MHCII, PD1, PD1TCF, TCF data sets with enhanced data cleanup classification

# Random Forest: Mean Cross-Validation Accuracy = 0.8349
# Gradient Boosting: Mean Cross-Validation Accuracy = 0.8859
# Logistic Regression: Mean Cross-Validation Accuracy = 0.6854
# SVM: Mean Cross-Validation Accuracy = 0.7332
# KNN: Mean Cross-Validation Accuracy = 0.8374
# Decision Tree: Mean Cross-Validation Accuracy = 0.8306

# Best Model: Gradient Boosting with Accuracy: 0.8859
# Test Accuracy with Gradient Boosting: 0.8859

#############################################################

# Save the best model to a file
model_filename = 'best_model.pkl'
# joblib.dump(best_pipeline, model_filename)
# print(f"Best model saved to {model_filename}")

# Best model saved to best_model.pkl