import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.degan_generator_dynamic import DEGANGeneratorDynamic
from utils.dataset_brats2015 import Brats2015Dataset
from utils.pdf_transform import pdf_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation dataset
val_dataset = Brats2015Dataset("/path/to/data")
val_loader  = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Load dynamic generator
G = DEGANGeneratorDynamic(w=64).to(device)

ckpt_path = "/path/to/data"
state = torch.load(ckpt_path, map_location=device)
G.load_state_dict(state)
G.eval()

mae_loss = nn.L1Loss()

total_mae = 0.0
count = 0

with torch.no_grad():
    for x_flair, global_mean, class_means, seg in val_loader:

        x_flair = x_flair.to(device)          # (B,1,H,W)
        global_mean = global_mean.to(device)  # (B,1,1,1)
        class_means = class_means.to(device)  # (B,3,1,1)
        seg = seg.to(device)                  # (B,H,W)

        # PDF-enhanced target (class-aware + label-guided)
        y_real = pdf_transform(
            x_flair,
            global_mean,
            class_means,
            seg
        )  # (B,1,H,W)

        # Dynamic model prediction
        y_fake = G(x_flair)  # (B,1,H,W)

        # MAE
        loss = mae_loss(y_fake, y_real)
        total_mae += loss.item()
        count += 1

avg_mae = total_mae / count
print(f"Validation MAE (DE-GAN): {avg_mae:.6f}")
