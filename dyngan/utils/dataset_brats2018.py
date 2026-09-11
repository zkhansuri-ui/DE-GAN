import os
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Brats2018Dataset(Dataset):
    def __init__(self, root_dir, split_file):
        self.root_dir = root_dir

        with open(split_file, "r") as f:
            self.samples = [line.strip() for line in f]

        self.train_root = os.path.join(root_dir, "MICCAI_BraTS_2018_Data_Training")
        self.val_root   = os.path.join(root_dir, "MICCAI_BraTS_2018_Data_Validation")

    def __len__(self):
        return len(self.samples) * 155

    def __getitem__(self, idx):
        case_idx = idx // 155
        slice_idx = idx % 155

        case_id = self.samples[case_idx]

        # Determine correct folder
        case_path_train = os.path.join(self.train_root, case_id)
        case_path_val   = os.path.join(self.val_root, case_id)

        if os.path.exists(case_path_train):
            case_path = case_path_train
        elif os.path.exists(case_path_val):
            case_path = case_path_val
        else:
            raise FileNotFoundError(f"Case {case_id} not found in training or validation folders.")

        # Correct filenames (.nii, not .nii.gz)
        flair_path = os.path.join(case_path, f"{case_id}_flair.nii")
        t1ce_path  = os.path.join(case_path, f"{case_id}_t1ce.nii")

        flair_img = sitk.ReadImage(flair_path)
        t1ce_img  = sitk.ReadImage(t1ce_path)

        # Extract slice
        flair_slice = sitk.GetArrayFromImage(flair_img)[slice_idx]
        t1ce_slice  = sitk.GetArrayFromImage(t1ce_img)[slice_idx]

        # Convert to tensor
        flair = torch.tensor(flair_slice, dtype=torch.float32).unsqueeze(0)
        t1ce  = torch.tensor(t1ce_slice, dtype=torch.float32).unsqueeze(0)

        # Resize to 128×128 (required for EnhGAN-Dynamic)
        flair = F.interpolate(flair.unsqueeze(0), size=(128,128), mode="bilinear", align_corners=False).squeeze(0)
        t1ce  = F.interpolate(t1ce.unsqueeze(0), size=(128,128), mode="bilinear", align_corners=False).squeeze(0)

        return flair, t1ce, case_id
