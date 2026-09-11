# train_dyngan.py - Training script for Enhanced DynGAN Parameters

import os
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.dyngan_generator_dynamic import DynGANGeneratorDynamic
from models.dyngan_discriminator import MultiscaleDiscriminator
from utils.pdf_transform import pdf_transform

alpha = 0.1
beta = 1.0
CHECKPOINT_DIR = "trained_model"
CHECKPOINT_BASENAME = "dyngan_dynamic_checkpoint"
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dyngan_dynamic_latest.pt")


def ensure_checkpoint_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def checkpoint_path_for_epoch(epoch_number):
    return os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_BASENAME}_epoch{epoch_number}.pt")


def find_latest_checkpoint():
    if os.path.exists(LATEST_CHECKPOINT_PATH):
        return LATEST_CHECKPOINT_PATH

    if not os.path.isdir(CHECKPOINT_DIR):
        return None

    checkpoint_pattern = re.compile(rf"{re.escape(CHECKPOINT_BASENAME)}_epoch(\d+)\.pt$")
    latest_epoch = -1
    latest_path = None

    for filename in os.listdir(CHECKPOINT_DIR):
        match = checkpoint_pattern.match(filename)
        if match is None:
            continue

        epoch_number = int(match.group(1))
        if epoch_number > latest_epoch:
            latest_epoch = epoch_number
            latest_path = os.path.join(CHECKPOINT_DIR, filename)

    return latest_path


def save_checkpoint(epoch_number, G, D, opt_G, opt_D):
    ensure_checkpoint_dir()

    checkpoint = {
        "epoch": epoch_number,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "optimizer_G_state_dict": opt_G.state_dict(),
        "optimizer_D_state_dict": opt_D.state_dict(),
    }

    epoch_checkpoint_path = checkpoint_path_for_epoch(epoch_number)
    torch.save(checkpoint, epoch_checkpoint_path)
    torch.save(checkpoint, LATEST_CHECKPOINT_PATH)
    print(f"  → Saved checkpoint: {epoch_checkpoint_path}")
    print(f"  → Updated latest checkpoint: {LATEST_CHECKPOINT_PATH}")


def load_checkpoint(checkpoint_path, G, D, opt_G, opt_D, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        G.load_state_dict(checkpoint["generator_state_dict"])
        D.load_state_dict(checkpoint["discriminator_state_dict"])
        opt_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        opt_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed training from checkpoint: {checkpoint_path} (epoch {start_epoch})")
        return start_epoch

    G.load_state_dict(checkpoint)
    print(f"Loaded generator weights from legacy checkpoint: {checkpoint_path}")
    return 0

def resolve_device(preferred_device=None):
    requested_device = preferred_device
    if requested_device is None or str(requested_device) == "auto":
        try:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            requested_device = "cpu"

    if str(requested_device).startswith("cuda"):
        try:
            torch.empty(1, device="cuda")
            return torch.device(requested_device)
        except Exception as exc:
            print(f"CUDA unavailable ({exc}); falling back to CPU.")
            return torch.device("cpu")

    return torch.device("cpu")


def train_dyngan(
    train_loader,
    epochs=400,
    device=None,
    checkpoint_interval=10,
    resume_checkpoint=None,
    auto_resume=True,
):
    """
    Training function for DynGAN
    
    Args:
        train_loader: DataLoader with training data
        epochs: Number of training epochs
        device: 'cuda' or 'cpu'
    """
    
    device = resolve_device(device)

    # Initialize generator and discriminator
    G = DynGANGeneratorDynamic(w=128, base_ch=32, use_coord_conv=True).to(device)
    D = MultiscaleDiscriminator().to(device)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Loss functions
    adv_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()

    ensure_checkpoint_dir()

    start_epoch = 0
    if resume_checkpoint is None and auto_resume:
        resume_checkpoint = find_latest_checkpoint()

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        start_epoch = load_checkpoint(resume_checkpoint, G, D, opt_G, opt_D, device)
        if start_epoch >= epochs:
            print(f"Checkpoint epoch {start_epoch} is already at or beyond requested epochs ({epochs}).")
            return G
        start_epoch += 1

    print(f"Training Enhanced DynGAN with Dynamic Parameters")
    print(f"Generator: {G.__class__.__name__}")
    print(f"Using device: {device}")
    print(f"Total Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
    print()

    for epoch in range(start_epoch, epochs):
        total_loss_G = 0.0
        total_loss_D = 0.0
        batch_count = 0

        for x, class_means in train_loader:

            x = x.to(device)
            class_means = class_means.to(device)

            y_real = pdf_transform(x, class_means)

            # -------------------------
            # Train Generator
            # -------------------------
            opt_G.zero_grad()

            y_fake = G(x)

            pred_fake = D(x, y_fake)
            
            loss_G_adv = sum([adv_loss(pf, torch.ones_like(pf)) for pf in pred_fake])
            loss_G_mae = mae_loss(y_fake, y_real)

            loss_G = alpha * loss_G_adv + beta * loss_G_mae
            loss_G.backward()
            opt_G.step()

            # -------------------------
            # Train Discriminator
            # -------------------------
            opt_D.zero_grad()

            pred_real = D(x, y_real)
            pred_fake = D(x, y_fake.detach())

            loss_D_real = sum([adv_loss(pr, torch.ones_like(pr)) for pr in pred_real])
            loss_D_fake = sum([adv_loss(pf, torch.zeros_like(pf)) for pf in pred_fake])

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_D.step()

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            batch_count += 1

        avg_loss_G = total_loss_G / batch_count
        avg_loss_D = total_loss_D / batch_count
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: G_loss={avg_loss_G:.4f}, D_loss={avg_loss_D:.4f}")

        # Save checkpoint periodically and on the final epoch.
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            save_checkpoint(epoch + 1, G, D, opt_G, opt_D)

    print("Training completed!")
    return G


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Enhanced DynGAN with Dynamic Parameters')
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save a checkpoint every N epochs')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Path to a checkpoint to resume from')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='Disable automatic resume from the latest checkpoint')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: train for only 1 epoch with 2 batches')
    
    args = parser.parse_args()
    
    # Test mode settings
    if args.test:
        args.epochs = 1
        print("Running in TEST mode (1 epoch, 2 batches)")
    
    # Set device
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    resume_checkpoint = args.resume_checkpoint
    if not args.no_auto_resume and resume_checkpoint is None:
        resume_checkpoint = find_latest_checkpoint()
    
    # Load dataset
    from utils.dataset import BraTS2021Dataset
    
    print("Initializing DataLoader for BraTS...")
    train_dataset = BraTS2021Dataset(
        txt_path="config/train_name_all.txt", 
        data_root="/path/to/data"),
        samples_per_volume=5
    )
    
    # In test mode, use smaller dataset
    if args.test:
        train_dataset.patient_ids = train_dataset.patient_ids[:2]
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    print(f"Dataset size: {len(train_dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Train model
    model = train_dyngan(
        train_loader,
        epochs=args.epochs,
        device=device,
        checkpoint_interval=args.checkpoint_interval,
        resume_checkpoint=resume_checkpoint,
        auto_resume=not args.no_auto_resume,
    )
    
    # Save final model
    torch.save(model.state_dict(), "trained_model/dyngan_dynamic_final.pt")
    print("Final model saved: trained_model/dyngan_dynamic_final.pt")
