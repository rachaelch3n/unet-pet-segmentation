import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

from main import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same image transform as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# load test dataset
test_dataset = datasets.OxfordIIITPet(
    root="./oxford_pet_data",
    split="test",
    download=False,
    transform=transform,
    target_types="segmentation"
)

# load trained model
model = UNet(in_channels=3, num_classes=3).to(device)

model.load_state_dict(torch.load("unet_model.pth"))

model.eval()

# create predictions folder
os.makedirs("predictions", exist_ok=True)

with torch.no_grad():

    for idx in range(len(test_dataset)):

        image, _ = test_dataset[idx]

        image = image.unsqueeze(0).to(device)

        logits = model(image)

        pred = torch.argmax(logits, dim=1)

        pred = pred.squeeze(0).cpu().numpy()

        # save prediction
        image_id = test_dataset.images[idx].split("/")[-1].split(".")[0]

        np.save(f"predictions/{image_id}.npy", pred)

print("Predictions saved!")