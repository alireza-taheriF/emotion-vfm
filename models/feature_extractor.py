# models/feature_extractor.py

import torch
import torch.nn as nn
import timm 

# Load DINOv2 model
def load_dino_model(model_name='vit_small_patch16_224', pretrained=True): 
    model = timm.create_model(model_name, pretrained=True)
    model.reset_classifier(0) # 0 is the default value for the number of classes in the model
    return model

# Extract features from the model
def extract_features(model, dataloader, device='cude' if torch.cuda.is_available() else 'cpu'): 
    model.to(device)
    features = []
    labels = []

    with torch.no_grad(): 
        for images, targets in dataloader:
            images = images.to(device)
            output = model(images)
            features.append(output.cpu())
            labels.append(targets)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    return features, labels
