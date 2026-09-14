import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import cv2

import SimpleITK as sitk

def load_volume(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # shape: [slices, H, W]
    arr = arr.transpose(1, 2, 0)       # convert to [H, W, slices]
    return arr

class Brats2015Dataset(Dataset):
    def __init__(self, root_dir, split_file):
        self.root_dir = root_dir

        with open(split_file, "r") as f:
            self.case_ids = [line.strip() for line in f if line.strip()]

        self.samples = []

        for cid in self.case_ids:

            # Search in all possible subfolders
            possible_dirs = [
                os.path.join(root_dir, "training", "HGG", cid),
                os.path.join(root_dir, "training", "LGG", cid),
                os.path.join(root_dir, "testing", "HGG_LGG", cid)
            ]

            case_dir = None
            for d in possible_dirs:
                if os.path.isdir(d):
                    case_dir = d
                    break

            if case_dir is None:
                print(f"[WARNING] Case folder not found: {cid}")
                continue

            # Dynamically detect modality filenames
            files = os.listdir(case_dir)

            flair_file = next((f for f in files if "flair" in f.lower()), None)
            t1ce_file  = next((f for f in files if "t1ce" in f.lower() or "t1c" in f.lower()), None)

            if flair_file is None or t1ce_file is None:
                print(f"[WARNING] Missing modalities for case: {cid}")
                continue

            flair_path = os.path.join(case_dir, flair_file)
            t1ce_path  = os.path.join(case_dir, t1ce_file)

            self.samples.append((flair_path, t1ce_path))

        print(f"Loaded {len(self.samples)} cases from Brats2015")

    def __len__(self):
        return len(self.samples) * 155

    def __getitem__(self, idx):
        case_idx  = idx // 155
        slice_idx = idx % 155

        flair_path, t1ce_path = self.samples[case_idx]

        # Load .mha using SimpleITK
        flair_vol = load_volume(flair_path)
        t1ce_vol  = load_volume(t1ce_path)

        flair_slice = flair_vol[:, :, slice_idx]
        t1ce_slice  = t1ce_vol[:, :, slice_idx]

        flair_slice = cv2.resize(flair_slice, (128, 128))
        t1ce_slice  = cv2.resize(t1ce_slice, (128, 128))

        flair_slice = (flair_slice - flair_slice.mean()) / (flair_slice.std() + 1e-8)
        t1ce_slice  = (t1ce_slice - t1ce_slice.mean()) / (t1ce_slice.std() + 1e-8)

        flair_tensor = torch.tensor(flair_slice, dtype=torch.float32).unsqueeze(0)
        t1ce_tensor  = torch.tensor(t1ce_slice, dtype=torch.float32).unsqueeze(0)

        return flair_tensor, t1ce_tensor
