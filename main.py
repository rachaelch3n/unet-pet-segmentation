import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

from data import train_loader, val_loader

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """UNet architecture for image segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        # TODO: define encoder (contracting path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # TODO: define bottleneck (e.g., 512 -> 1024)
        self.bottleneck = DoubleConv(512, 1024)
        # TODO: define decoder (expanding path)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        # TODO: define final 1x1 conv (out_channels = num_classes)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: encoder forward, store feature maps for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # TODO: bottleneck forward
        b = self.bottleneck(self.pool(e4))
        # TODO: decoder forward, concat with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        # TODO: return segmentation logits via final 1x1 conv
        return self.final_conv(d1)


class SegmentationLoss(nn.Module):
    """Loss function for segmentation (e.g., BCE / CrossEntropy / Dice / combined)."""

    def __init__(self):
        super().__init__()
        # TODO: define the loss to use
        #   - BCEWithLogitsLoss for binary segmentation
        #   - CrossEntropyLoss for multi-class segmentation
        #   - optionally combine with Dice loss
        weights = torch.tensor([1.0, 1.0, 3.0])
        self.register_buffer("weights", weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO: compute and return loss from logits and targets
        targets = targets.long()
        return self.loss_fn(logits, targets)


class Trainer:
    """Training / validation loop wrapper."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        # TODO: iterate over (images, masks) batches from loader
        #   1) move tensors to device
        #   2) optimizer.zero_grad()
        #   3) forward -> compute loss
        #   4) loss.backward() -> optimizer.step()
        #   5) accumulate and return average loss
        total_loss = 0

        for images, masks in loader:

            images = images.to(self.device)
            masks = masks.to(self.device).long()

            self.optimizer.zero_grad()

            logits = self.model(images)

            loss = self.criterion(logits, masks)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def mean_iou(self, preds, masks, num_classes=3):
        ious = []

        for cls in range(num_classes):
            pred_cls = preds == cls
            mask_cls = masks == cls

            intersection = (pred_cls & mask_cls).sum().item()
            union = (pred_cls | mask_cls).sum().item()

            if union > 0:
                ious.append(intersection / union)

        return sum(ious) / len(ious)

    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        # TODO: run forward only and compute loss / metrics (IoU, Dice, etc.)
        total_loss = 0
        total_iou = 0

        for images, masks in loader:

            images = images.to(self.device)
            masks = masks.to(self.device).long()

            logits = self.model(images)

            loss = self.criterion(logits, masks)

            total_loss += loss.item()

            # PREDICTED CLASS FOR EACH PIXEL
            preds = torch.argmax(logits, dim=1)

            # COMPUTE IoU
            iou = self.mean_iou(preds, masks, num_classes=3)

            total_iou += iou

        avg_loss = total_loss / len(loader)
        avg_iou = total_iou / len(loader)

        print(f"mIoU: {avg_iou:.4f}")

        return avg_loss, avg_iou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: set hyperparameters (lr, num_epochs, num_classes, etc.)
    num_epochs = 50
    learning_rate = 1e-4
    num_classes = 3

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    ).to(device)

    criterion = SegmentationLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    trainer = Trainer(model, criterion, optimizer, device)

    best_miou = 0.0
    
    for epoch in range(num_epochs):

        train_loss = trainer.train_one_epoch(train_loader)

        val_loss, val_miou = trainer.validate(val_loader)

        scheduler.step(val_miou)

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f}"
        )
        # TODO: save best model checkpoint / visualize predictions / report metrics
        if val_miou > best_miou:
            best_miou = val_miou

            torch.save(model.state_dict(), "best_unetplusplus_model.pth")

if __name__ == "__main__":
    main()
