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
Based upon the statistical accuracy for each machine learning model with the underlying data sets, the _Stacking Classifier_ machine learning model presents the highest mean cross-validation accuracy (**88.43%**).
Results of all selected machine learning models based upon limited feature set:
* **Linear Models**
  - Logistic Regression: Mean Cross-Validation Accuracy = 0.6447
  - Linear Discriminant Analysis (LDA): Mean Cross-Validation Accuracy = 0.6075
  - Ridge Classifier: Mean Cross-Validation Accuracy = 0.6134
  - Stochastic Gradient Descent (SGD): Mean Cross-Validation Accuracy = 0.6546
* **Tree-Based Models**
  - Decision Tree: Mean Cross-Validation Accuracy = 0.8034
  - Random Forest: Mean Cross-Validation Accuracy = 0.8084
  - Gradient Boosting: Mean Cross-Validation Accuracy = 0.8647
  - LightGBM: Mean Cross-Validation Accuracy = 0.8347
  - Extra Trees: Mean Cross-Validation Accuracy = 0.8035
* **Support Vector Machines**
  - SVC: Mean Cross-Validation Accuracy = 0.8507
  - Linear SVC: Mean Cross-Validation Accuracy = 0.6549
* **Nearest Neighbors**
  - K-Nearest Neighbors (KNN): Mean Cross-Validation Accuracy = 0.8117
* **Neural Networks**
  - Multi-Layer Perceptron (MLP): Mean Cross-Validation Accuracy = 0.8588
* **Ensemble Methods**
  - AdaBoost: Mean Cross-Validation Accuracy = 0.5366
  - Bagging: Mean Cross-Validation Accuracy = 0.8083
  - Voting Classifier: Mean Cross-Validation Accuracy = 0.8212
  - Stacking Classifier: Mean Cross-Validation Accuracy = 0.8843
* **Probabilistic Models**
  - Gaussian Naive Bayes: Mean Cross-Validation Accuracy = 0.5245
  - Bernoulli Naive Bayes: Mean Cross-Validation Accuracy = 0.6390
* **Clustering-Based Models**
  - K-Means: Mean Cross-Validation Accuracy = 0.0000
  - Gaussian Mixture Model (GMM): Mean Cross-Validation Accuracy = 0.0000
* **Other Models**
  - Quadratic Discriminant Analysis (QDA): Mean Cross-Validation Accuracy = 0.6737
  - Partial Least Squares (PLS): Mean Cross-Validation Accuracy = nan

Best Model: Stacking Classifier with Accuracy: 0.8843

The Stacking Classifier was modeled with LogisticRegression, RandomForestClassifier, SVC. Other combinations with higher individual outcomes were tested, but did not provide a better outcome.

### 2.5.6 High Level Model Description
**Logistic Regression**
_Description:_ Models the probability of a binary or multi-class outcome using a logistic function.  
_Weaknesses:_ Assumes linear decision boundaries; struggles with non-linear data.  
_Applications:_ Spam detection, disease diagnosis, credit scoring.  
**Linear Discriminant Analysis (LDA)**
_Description:_ Finds a linear combination of features to separate classes, assuming Gaussian distributions.  
_Weaknesses:_ Sensitive to outliers; assumes equal class covariances.  
_Applications:_ Face recognition, medical diagnosis, marketing.  
**Ridge Classifier**
_Description:_ Classification with L2 regularization to prevent overfitting.  
_Weaknesses:_ Requires tuning of the regularization parameter; not ideal for non-linear data.  
_Applications:_ Text classification, small datasets with many features.  
**Stochastic Gradient Descent (SGD)**
_Description:_ Optimizes models incrementally using small data batches.  
_Weaknesses:_ Sensitive to learning rate; may converge to local minima.  
_Applications:_ Large-scale learning, online learning, deep learning.  
**Decision Tree**  
_Description:_ Splits data into branches based on feature values to make predictions.  
_Weaknesses:_ Prone to overfitting; unstable with small data changes.  
_Applications:_ Customer segmentation, fraud detection, medical diagnosis.  
**Random Forest**  
_Description:_ Ensemble of decision trees to reduce overfitting and improve accuracy.  
_Weaknesses:_ Computationally expensive; less interpretable than single trees.  
_Applications:_ Predictive modeling, feature importance analysis, anomaly detection.  
**Gradient Boosting**  
_Description:_ Sequentially builds trees to correct errors from previous trees.  
_Weaknesses:_ Computationally intensive; sensitive to noisy data.  
_Applications:_ Kaggle competitions, financial forecasting, ranking.  
**LightGBM**  
_Description:_ Gradient boosting framework optimized for speed and efficiency.  
_Weaknesses:_ May overfit on small datasets; requires careful hyperparameter tuning.  
_Applications:_ Click-through rate prediction, high-performance tasks.  
**Extra Trees (Extremely Randomized Trees)**  
_Description:_ Randomizes feature selection and split points to reduce variance.  
_Weaknesses:_ Less interpretable; may underfit with too much randomization.  
_Applications:_ Similar to Random Forest but faster and less prone to overfitting.  
**SVC (Support Vector Classification)**  
_Description:_ Finds the optimal hyperplane to separate classes, with kernel support.  
_Weaknesses:_ Computationally expensive for large datasets; requires tuning.  
_Applications:_ Image classification, bioinformatics, text classification.  
**Linear SVC**  
_Description:_ Optimized SVM for linear decision boundaries.  
_Weaknesses:_ Limited to linear relationships; sensitive to scaling.  
_Applications:_ Text classification, large datasets.  
**K-Nearest Neighbors (KNN)**  
_Description:_ Predicts based on the majority class or average of the nearest neighbors.  
_Weaknesses:_ Computationally expensive for large datasets; sensitive to irrelevant features.  
_Applications:_ Recommendation systems, anomaly detection, image recognition.  
**Multi-Layer Perceptron (MLP)**  
_Description:_ Neural network with one or more hidden layers for non-linear modeling.  
_Weaknesses:_ Requires large data; prone to overfitting; hard to interpret.  
_Applications:_ Image recognition, natural language processing, time-series forecasting.  
**AdaBoost**  
_Description:_ Combines weak classifiers by focusing on misclassified samples.  
_Weaknesses:_ Sensitive to noisy data; may overfit.  
_Applications:_ Face detection, customer churn prediction, fraud detection.  
**Bagging**  
_Description:_ Trains multiple models on bootstrapped samples and averages predictions.  
_Weaknesses:_ Computationally expensive; less effective for simple models.  
_Applications:_ Improving stability of decision trees, regression models.  
**Voting Classifier**  
_Description:_ Combines predictions from multiple models using hard or soft voting.  
_Weaknesses:_ Requires diverse models; computationally expensive.  
_Applications:_ Improving accuracy by leveraging diverse models.  
**Stacking Classifier**  
_Description:_ Uses a meta-model to combine predictions from base models.  
_Weaknesses:_ Complex to implement; requires careful tuning.  
_Applications:_ Competitions, complex datasets.  
**Gaussian Naive Bayes**  
_Description:_ Assumes features are conditionally independent and follow a Gaussian distribution.  
_Weaknesses:_ Struggles with correlated features; oversimplifies data.  
_Applications:_ Text classification, spam filtering, sentiment analysis.  
**Bernoulli Naive Bayes**  
_Description:_ Assumes binary features (e.g., presence/absence of words).  
_Weaknesses:_ Limited to binary data; oversimplifies relationships.  
_Applications:_ Document classification, sentiment analysis.  
**K-Means**  
_Description:_ Partitions data into K clusters by minimizing variance.  
_Weaknesses:_ Sensitive to initialization; assumes spherical clusters.  
_Applications:_ Customer segmentation, image compression, anomaly detection.  
**Gaussian Mixture Model (GMM)**  
_Description:_ Models data as a mixture of Gaussian distributions for soft clustering.  
_Weaknesses:_ Computationally expensive; sensitive to initialization.  
_Applications:_ Anomaly detection, speech recognition, image segmentation.  
**Quadratic Discriminant Analysis (QDA)**  
_Description:_ Similar to LDA but allows for non-linear decision boundaries.  
_Weaknesses:_ Requires more data; prone to overfitting with many features.  
_Applications:_ When class covariances are significantly different.  
**Partial Least Squares (PLS)**  
_Description:_ Finds latent variables explaining variance in predictors and response.  
_Weaknesses:_ Hard to interpret; requires careful tuning.  
_Applications:_ Chemometrics, bioinformatics, financial modeling.  
### 2.  6 Prediction

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
