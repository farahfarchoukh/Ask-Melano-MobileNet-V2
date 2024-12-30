import os
import torch
import albumentations
from torchvision import models
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import model_selection
from sklearn import metrics
from torch.utils.data import DataLoader

# Define your ClassificationLoader here or import it if it's defined elsewhere
# from your_module import ClassificationLoader

class MobileNetV2Wrapper(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Wrapper, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)  # For binary classification
    
    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.model.classifier(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss

def train(fold):
    input_path = "/home/askmelano/workspace/melanoma/input/"
    df = pd.read_csv(os.path.join(input_path, "train.csv"))
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=10)
    for fold_, (_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = fold_

    training_data_path = "/home/askmelano/workspace/melanoma/input/jpeg/train224/"
    model_path = "/home/askmelano/workspace/melanoma-deep-learning/model.pth"  # Save model as .pth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose([albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])
    valid_aug = albumentations.Compose([albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(image_paths=train_images, targets=train_targets, resize=None, augmentations=train_aug)
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4)

    valid_dataset = ClassificationLoader(image_paths=valid_images, targets=valid_targets, resize=None, augmentations=valid_aug)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=4)

    model = MobileNetV2Wrapper(pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode="max")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, loss = model(images, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        predictions = []
        valid_loss = 0.0
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                outputs, loss = model(images, targets)
                valid_loss += loss.item()
                predictions.append(outputs.sigmoid().cpu().numpy())  # Use sigmoid for binary classification
avg_valid_loss = valid_loss / len(valid_loader)
        predictions = np.concatenate(predictions).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)  # Calculate AUC
        scheduler.step(auc)  # Update the learning rate based on AUC

        print(f"Validation Loss: {avg_valid_loss:.4f}, AUC: {auc:.4f}")

        # Save the model if the AUC improves
        torch.save(model.state_dict(), model_path)

# Call the train function for a specific fold
if __name__ == "__main__":
    train(fold=0)  # You can change the fold number as needed
