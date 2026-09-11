import os
import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from models.dyngan_generator_dynamic import DynGANGeneratorDynamic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/path/to/data"
OUT_ROOT  = "/path/to/data"
CKPT_PATH = "/path/to/data"

os.makedirs(OUT_ROOT, exist_ok=True)

G = DynGANGeneratorDynamic(w=128, base_ch=32, use_coord_conv=True).to(device)
G.load_state_dict(torch.load(CKPT_PATH, map_location=device))
G.eval()

PATCH = 128

def extract_patches(vol):
    H, W, D = vol.shape
    patches, coords = [], []
    for z in range(D):
        slice_ = vol[:, :, z]
        for i in range(0, H - PATCH + 1, PATCH):
            for j in range(0, W - PATCH + 1, PATCH):
                p = slice_[i:i+PATCH, j:j+PATCH]
                patches.append(p)
                coords.append((i, j, z))
    return patches, coords

def reconstruct(patches, coords, shape):
    out = np.zeros(shape, dtype=np.float32)
    for p, (i, j, z) in zip(patches, coords):
        out[i:i+PATCH, j:j+PATCH, z] = p
    return out

subjects = sorted(
    list(Path(DATA_ROOT).glob("**/*_flair.nii")) +
    list(Path(DATA_ROOT).glob("**/*_flair.nii.gz"))
)


for flair_path in subjects:
    case_dir = flair_path.parent
    case_id = case_dir.name
    print("Processing:", case_id, flush=True)

    flair = nib.load(str(case_dir / f"{case_id}_flair.nii")).get_fdata()
    
    patches, coords = extract_patches(flair)
    syn_patches = []

    for p in patches:
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
        p_tensor = torch.tensor(p_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            syn = G(p_tensor).cpu().numpy()[0, 0]

        syn_patches.append(syn)

    syn_vol = reconstruct(syn_patches, coords, flair.shape)

    out_path = f"{OUT_ROOT}/{case_id}_dynamic_flair.nii.gz"
    nib.save(nib.Nifti1Image(syn_vol, np.eye(4)), out_path)

    print("Done:", case_id, flush=True)
