# Research Project: Data Cleanup, Data Classification, Data Analytics, Machine Learning Modeling, Prediction, and Presentation for Biomedical Applications

**Author:** Isabelle De Bruyker, Biomedical Engineer  
**Graduation:** Cum Laude, 2024  
**Institution:** Georgia Institute of Technology  

## Abstract
This research project presents a comprehensive approach to data-driven biomedical research, focusing on data data cleanup, data classification, data analytics, machine learning modeling, model selection, prediction and presentation. The study aims to demonstrate the importance of robust data preprocessing and advanced analytical techniques in deriving meaningful insights and accurate predictions from biomedical datasets. The research was conducted by Isabelle De Bruyker, a Biomedical Engineer who graduated cum laude from Georgia Tech in 2024. The paper outlines the methodologies employed, the challenges encountered, and the results achieved, providing a framework for future research in the field.

## 1. Introduction
Biomedical research increasingly relies on large datasets to uncover patterns, predict outcomes, and inform decision-making. However, raw biomedical data is often noisy, incomplete, and inconsistent, necessitating rigorous data cleanup and preprocessing. This research explores the end-to-end process to classify and present cell components through statistics, more particularily machine learning process, to address these challenges and deliver actionable insights. The study is grounded in real-world biomedical applications, leveraging the expertise of Isabelle De Bruyker, a recent graduate of Georgia Tech's Biomedical Engineering program.

## 2. Methodology
The end-to-end process to classify and present cell components through statistics, more particularily machine learning process, includes:
* Data Collection: Gather raw data from various sources.
* Data Cleanup: Preprocess and clean the data.
* Data Classification: Understand and categorize the data.
* Data Analytics: Explore and analyze the data for insights.
* Machine Learning Modeling: Build and train models.
* Model Selection: Evaluate and select the best model.
* Prediction: Use the model to make predictions.
* Presentation: Communicate results effectively.

### 2.1 Data Collection

### 2.2 Data Cleanup

### 2.3 Data Classification

### 2.4 Data Analytics

### 2.5 Machine Learning Modeling
#### 2.5.1 Data Set Selection for Modeling
Three random cell data sets were selected for modeling:
* 53904_1
* 33683_1
* 33270_2
#### 2.5.2 Machine Learning Models
The following machine learning models were evaluated:
* Random Forest
* Gradient Boosting
* Logistic Regression
* SVM
* KNN
* Decision Tree
### 2.5.3 Features
The evaluation was done based upon all features (139), but in line with previous data analysis, limited incremental benefit was achieved by chosing all features versus a limited set. The retained features are:
* Centroid.X.µm
* Centroid.Y.µm
* Nucleus..Opal.570.mean
* Nucleus..Opal.690.mean
* Nucleus..Opal.480.mean
* Nucleus..Opal.620.mean
* Nucleus..Opal.520.mean
### 2.5.4 Training / Testing
Models were trained on 80% of the data and tested on 20% of the data.
### 2.5.5 Results
Based upon the statistical accuracy for each machine learning model with the underlying data sets, the --Gradient Boosting-- machine learning model presents the highest mean cross-validation accuracy (**88.59%**).
Results of all selected machine learning models based upon limited feature set:
* Random Forest: Mean Cross-Validation Accuracy = 0.8349
* Gradient Boosting: Mean Cross-Validation Accuracy = 0.8859
* Logistic Regression: Mean Cross-Validation Accuracy = 0.6854
* SVM: Mean Cross-Validation Accuracy = 0.7332
* KNN: Mean Cross-Validation Accuracy = 0.8374
* Decision Tree: Mean Cross-Validation Accuracy = 0.8306
### 2.5.6 High Level Model Description
**Random Forests**
__Description:__ An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting. Each tree is trained on a random subset of the data and features.
__Strengths:__ Robust, handles high-dimensional data, and reduces overfitting compared to single decision trees.
__Weaknesses:__ Less interpretable than individual decision trees, computationally expensive for large datasets.
__Applications:__ Predicting disease progression, identifying biomarkers.
**Gradient Boosting Machines (GBM)**
__Description:__ An ensemble technique that builds models sequentially, with each new model correcting errors made by the previous ones. Common implementations include XGBoost, LightGBM, and CatBoost.
__Strengths:__ High accuracy, handles heterogeneous data, and robust to outliers.
__Weaknesses:__ Computationally expensive, prone to overfitting if not tuned properly.
__Applications:__ Predicting patient readmission rates, risk stratification.
**Logistic Regression**
__Description:__ A supervised learning algorithm used for binary classification problems. It predicts the probability of an outcome using a logistic function.
__Strengths:__ Easy to implement, provides probabilistic interpretations.
__Weaknesses:__ Limited to linear decision boundaries and may underperform on non-linear data.
__Applications:__ Disease diagnosis (e.g., predicting presence or absence of a disease).
**Support Vector Machines (SVM)**
__Description:__ A supervised learning algorithm used for classification and regression. It finds the optimal hyperplane that separates data points of different classes with the maximum margin.
__Strengths:__ Effective in high-dimensional spaces, versatile with kernel functions for non-linear data.
__Weaknesses:__ Computationally intensive, requires careful tuning of hyperparameters.
__Applications:__ Classifying medical images, predicting patient outcomes.
**K-Nearest Neighbors (KNN)**
__Description:__ A non-parametric, instance-based learning algorithm that classifies data points based on the majority class of their k-nearest neighbors in the feature space.
__Strengths:__ Simple, no training phase, and adapts easily to new data.
__Weaknesses:__ Computationally expensive for large datasets, sensitive to irrelevant features.
__Applications:__ Patient clustering, disease classification.
**Decision Trees**
__Description:__ A tree-like model that splits data into subsets based on feature values. Each internal node represents a decision based on a feature, and each leaf node represents an outcome.
__Strengths:__ Easy to interpret, handles non-linear relationships, and requires minimal data preprocessing.
__Weaknesses:__ Prone to overfitting, especially with deep trees.
__Applications:__ Classifying patient risk levels, predicting treatment outcomes.
### 2.6 Prediction

Objective: Use the selected model to make predictions on new or unseen data.
Steps:
Prepare New Data: Ensure the new data is preprocessed in the same way as the training data.
Generate Predictions: Use the trained model to predict outcomes.
Post-Processing: Convert predictions into a usable format (e.g., class labels, probabilities).
Outcome: Predictions for the given input data.

### 2.7 Presentation

Objective: Communicate the results and insights effectively to stakeholders.
Steps:
Visualize Results: Create charts, graphs, and dashboards to present predictions and insights.
Generate Reports: Summarize the process, findings, and recommendations in a report or presentation.
Explain Model Decisions: Use techniques like SHAP or LIME to explain model predictions (if applicable).
Interactive Demos: Build interactive tools or dashboards (e.g., using Streamlit, Dash, or Power BI).
Outcome: Clear, actionable insights and predictions presented to stakeholders.

## Contributors
Isabelle De Bruyker, BS Biomedical Engineering, Georgia Institute of Technology

## Notes
* Original datasets have not been made available in github
* 'prep' folder contains intermediatory trial code only. Code is not used for final outcomes.
