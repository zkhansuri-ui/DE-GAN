import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dyngan_generator_dynamic import DynGANGeneratorDynamic
from utils.dataset_brats2015 import Brats2015Dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "/path/to/data"

    # -----------------------------
    # 1. Correct dataset arguments
    # -----------------------------
    test_subjects_file = "/path/to/data")

    # IMPORTANT: your test subjects live inside testing/HGG_LGG
    root_dir="/path/to/data")

    test_dataset = Brats2015Dataset(
        list_file=test_subjects_file,
        root_dir=root_dir
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------
    # 2. Load model
    # -----------------------------
    G = DynGANGeneratorDynamic()
    G.to(device)

    print(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state["G_state_dict"] if "G_state_dict" in state else state)
    G.eval()

    # -----------------------------
    # 3. Test loop
    # -----------------------------
    mae_sum = 0.0
    n_pixels = 0

    mae_loss = nn.L1Loss(reduction="sum")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):

            # Your dataset returns: (x2d_resized, class_means)
            x, class_means = batch

            x = x.to(device)                # shape: (B, 4, 64, 64)
            class_means = class_means.to(device)

            # Dynamic model prediction
            y_fake = G(x)

            # PDF target = class_means broadcast to image size
            # Use only the FLAIR mean (channel 0)
            flair_mean = class_means[:, 0:1, :, :]        # shape: (B, 1, 1, 1)

            # Expand to match generator output
            y_real = flair_mean.expand_as(y_fake)         # shape: (B, 1, 64, 64)

            # Compute MAE
            mae = mae_loss(y_fake, y_real).item()
            mae_sum += mae
            n_pixels += y_real.numel()

            if (batch_idx + 1) % 10 == 0:
                print(f"[{batch_idx+1}/{len(test_loader)}] partial MAE: {mae_sum / n_pixels:.6f}")

    test_mae = mae_sum / n_pixels
    print(f"\nTest MAE (DynGAN): {test_mae:.6f}")

if __name__ == "__main__":
    main()