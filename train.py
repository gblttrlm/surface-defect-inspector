import os
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ----------------------------
    # Data transforms
    # ----------------------------
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ----------------------------
    # Datasets
    # ----------------------------
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Using device: {DEVICE}")

    # ----------------------------
    # Model: ResNet18 transfer learning
    # ----------------------------
    model = models.resnet18(weights=None)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)

            train_loss += loss.item() * inputs.size(0)
            train_correct += (preds == labels).sum().item()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = train_correct / len(train_dataset)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_correct / len(val_dataset)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f} | Val   Acc: {epoch_val_acc:.4f}")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best model updated.")

    # ----------------------------
    # Save best model
    # ----------------------------
    model.load_state_dict(best_model_wts)

    save_path = os.path.join(MODEL_DIR, "defect_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE
    }, save_path)

    print(f"\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()