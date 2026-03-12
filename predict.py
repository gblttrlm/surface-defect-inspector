import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path="models/defect_model.pth"):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, transform, class_names


def predict_image(image_path):
    model, transform, class_names = load_model()

    image = Image.open(image_path).convert("L")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    return pred_class, confidence


if __name__ == "__main__":
    img = "data/test/pitted_surface/Ps_34.bmp"   # change if needed
    label, conf = predict_image(img)
    print("Prediction:", label)
    print("Confidence:", conf)