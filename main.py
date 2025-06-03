# main.py

from utils.dataset_loader import get_dataloaders
from models.feature_extractor import load_dino_model, extract_features
from classifier.train_classifier import train_and_evaluate

def main():
    print("🚀 Loading dataset...")
    train_loader, _, class_names = get_dataloaders("data/FER2013", batch_size=64)

    print("🧠 Loading DINOv2 model...")
    model = load_dino_model()

    print("🔍 Extracting features...")
    X, y = extract_features(model, train_loader)

    print("📊 Training classifier...")
    clf, scaler = train_and_evaluate(X, y)

    print("✅ Done. Model trained and saved.")
    print("📂 Classes:", class_names)

if __name__ == "__main__":
    main()
