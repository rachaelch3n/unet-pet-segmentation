import os
import pandas as pd
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

import segmentation_models_pytorch as smp

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
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=3,
).to(device)

model.load_state_dict(torch.load("best_unetplusplus_model.pth", map_location=device))

model.eval()

# CREATE PREDICTIONS FOLDER
os.makedirs("predictions", exist_ok=True)

# READ SAMPLE SUBMISSION
df = pd.read_csv("sample_submission.csv")

with torch.no_grad():
    for _, row in df.iterrows():
        image_id = row["id"]
        H = int(row["height"])
        W = int(row["width"])

        image_path = f"test_images/{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        logits = model(image_tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")

        # resize from 224x224 back to original resolution
        pred_img = Image.fromarray(pred)
        pred_img = pred_img.resize((W, H), resample=Image.NEAREST)
        pred = np.array(pred_img).astype("uint8")

        np.save(f"predictions/{image_id}.npy", pred)

print("Predictions saved!")