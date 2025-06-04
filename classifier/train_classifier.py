# classifier/train_classifier.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Train and evaluate the model
def train_and_evaluate(X, y, save_path='classifier/mlp_model.joblib'):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Train/Test Split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    #Train MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)

    #Evaluate the model
    y_pred = clf.predict(X_val)
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

    #Save model & scaler
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(clf, save_path)
    joblib.dump(scaler, save_path.replace('mlp_model', 'scaler'))

    return clf, scaler
