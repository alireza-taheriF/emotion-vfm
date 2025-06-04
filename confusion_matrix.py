# confusion_matrix.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib
import torch # Add torch for checking num_workers if needed

# Load dataset
from utils.dataset_loader import get_dataloaders

# Load model
from models.feature_extractor import load_dino_model, extract_features

# Load classifier
from classifier.train_classifier import train_and_evaluate

# Main
if __name__ == '__main__':
    # Load dataset
    # If you use num_workers > 0 in get_dataloaders, you may need to change it to 0 or put this part inside the if __name__ == '__main__': block.
    # It is recommended to check num_workers in get_dataloaders first.
    # We assume that get_dataloaders is set up correctly or works with num_workers=0.
    train_loader, _, class_names = get_dataloaders("data/FER2013")

    # Load model
    model = load_dino_model()
    X, y = extract_features(model, train_loader)

    # Load classifier
    clf, _ = train_and_evaluate(X, y)

    # Predict
    y_pred = clf.predict(X)

    # Draw confusion matrix
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
