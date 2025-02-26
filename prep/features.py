import joblib
import pandas as pd


# Load the saved model
model_filename = 'best_model.pkl'
loaded_model = joblib.load(model_filename)

# Extract feature names and importance
if hasattr(loaded_model, 'feature_names_in_'):
    feature_names = loaded_model.feature_names_in_
    print("Feature Names:", feature_names)

if hasattr(loaded_model, 'feature_importances_'):
    feature_importance = loaded_model.feature_importances_
    print("Feature Importance:", feature_importance)

    # Combine feature names with importance scores
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    print("Feature Importance with Names:", feature_importance_dict)