import os
import pandas as pd
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from main import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# LOAD MODEL
model = UNet(in_channels=3, num_classes=3).to(device)

model.load_state_dict(torch.load("unet_model.pth"))

model.eval()

# CREATE PREDICTIONS FOLDER
os.makedirs("predictions", exist_ok=True)

# READ SAMPLE SUBMISSION
df = pd.read_csv("sample_submission.csv")

with torch.no_grad():

    for image_id in df["id"]:

        # LOAD IMAGE
        image_path = f"test_images/{image_id}.jpg"

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0).to(device)

        # PREDICT
        logits = model(image)

        pred = torch.argmax(logits, dim=1)

        pred = pred.squeeze(0).cpu().numpy()

        # SAVE .NPY
        np.save(f"predictions/{image_id}.npy", pred)

print("Predictions saved!")