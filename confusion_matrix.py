# confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib
import torch # اضافه کردن torch برای بررسی num_workers در صورت لزوم

# بارگذاری داده
from utils.dataset_loader import get_dataloaders

# بارگذاری مدل
from models.feature_extractor import load_dino_model, extract_features

# بارگذاری طبقه‌بند
from classifier.train_classifier import train_and_evaluate

# --- اینجا تغییر اصلی شروع میشه ---
if __name__ == '__main__':
    # بارگذاری داده
    # اگر در get_dataloaders از num_workers > 0 استفاده می‌کنید، ممکن است نیاز باشد
    # num_workers را به 0 تغییر دهید یا این قسمت را داخل بلاک if __name__ == '__main__': قرار دهید.
    # توصیه می‌شود ابتدا num_workers در get_dataloaders بررسی شود.
    # فرض می‌کنیم get_dataloaders خودش به درستی تنظیم شده یا با num_workers=0 کار می‌کند.
    train_loader, _, class_names = get_dataloaders("data/FER2013")

    # بارگذاری مدل
    model = load_dino_model()
    X, y = extract_features(model, train_loader)

    # بارگذاری طبقه‌بند
    clf, _ = train_and_evaluate(X, y)

    # پیش‌بینی
    y_pred = clf.predict(X)

    # رسم ماتریس
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
