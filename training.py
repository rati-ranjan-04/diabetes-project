import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

df = pd.read_csv("diabetes.csv")
print(df.head(5))
print("Dataset Shape: ", df.shape)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]
print("Feature Matrix Shape: ", X.shape)
print("Target Vector Shape: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

def print_metrics(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(title)
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Precision: {prec:.2%}")
    print(f"  Recall: {rec:.2%}")
    print(f"  F1-score: {f1:.2%}")
    print()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Define the hyperparameter grid to search
# The keys are prefixed with the name of the pipeline step ('classifier')
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs']
}

# Set up GridSearchCV (cv=5 means 5-fold cross-validation)
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)

# Training the model
print("Starting hyperparameter tuning with GridSearchCV...")
grid_search.fit(X_train, y_train)

print("\nBest hyperparameters found:")
print(grid_search.best_params_)

y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)

y_test_prob = grid_search.predict_proba(X_test)[:, 1]

print_metrics("TRAIN METRICS", y_train, y_train_pred)
print_metrics("TEST METRICS", y_test, y_test_pred)

cm = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix (Test Data)")
plt.show()

dump(grid_search, "model_dir/diabetes_model.joblib")
print("Model saved successfully to model_dir/diabetes_model.joblib")