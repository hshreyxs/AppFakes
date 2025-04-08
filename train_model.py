import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import os

# ✅ Load Dataset
print("🔹 Loading dataset...")
df = pd.read_csv("train_cleaned_final.csv")

# ✅ Ensure the dataset contains OUTPUT column
if 'OUTPUT' not in df.columns:
    raise KeyError("The dataset must contain the 'OUTPUT' column for labels.")

# ✅ Separate features and labels
X = df.drop('OUTPUT', axis=1)
y = df['OUTPUT']

print(f"✅ Dataset loaded with {X.shape[0]} rows and {X.shape[1]} features.")

# ✅ Apply SMOTE to balance classes
print("🔹 Applying SMOTE for class balancing...")
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"✅ After SMOTE: {X_resampled.shape[0]} samples")

# ✅ Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Define the RandomForestClassifier and parameter grid
print("🔹 Training RandomForest model with hyperparameter tuning...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# ✅ Train the model
grid_search.fit(X_train, y_train)

# ✅ Get the best model
best_model = grid_search.best_estimator_

# ✅ Evaluate the model
y_pred = best_model.predict(X_test)

print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Save the trained model
model_path = "rf_model.pkl"
print("\n🔹 Saving the model...")

try:
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"✅ Model saved successfully as '{model_path}' with size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# ✅ Save feature columns
feature_path = "feature_columns.pkl"
print("\n🔹 Saving feature columns...")

try:
    with open(feature_path, "wb") as f:
        pickle.dump(X.columns.tolist(), f)
    print(f"✅ Feature columns saved successfully as '{feature_path}'")
except Exception as e:
    print(f"❌ Error saving feature columns: {e}")
