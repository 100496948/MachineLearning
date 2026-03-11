"""
January 2026 - Machine Learning Classes
University Carlos III of Madrid

This template scikit to train a decision tree classifier
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1) Load CSV
DATA_PATH = "lunar_lander_data.csv"  # change to your dataset
df = pd.read_csv(DATA_PATH)

# 2) Separate inputs (X) and label (y)
# Assume the label column is named 'action'
y = df["action"]
X = df.drop(columns=["action"])

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Train model (simple baseline)
clf = DecisionTreeClassifier(random_state=42)   # optional: max_depth=5
clf.fit(X_train, y_train)

# 5) Evaluate
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 6) Save model
joblib.dump(clf, "lunarlander_tree.pkl")
print("Saved model to lunarlander_tree.pkl")