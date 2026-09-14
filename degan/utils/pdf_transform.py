# utils/pdf_transform.py

import torch


def pdf_transform(x, global_mean, class_means, seg, sigma=0.1):
    """
    Class-aware, label-guided PDF transform.

    Args:
        x: input patch (B,1,H,W) - FLAIR
        global_mean: (B,1,1,1) or (1,1,1) global mean intensity
        class_means: (B,3,1,1) or (3,1,1) for WT, TC, ET
        seg: (B,H,W) or (H,W) label map with BraTS labels (0,1,2,3)

    Returns:
        y_real: enhanced target (B,1,H,W)
    """
    if global_mean.dim() == 3:
        global_mean = global_mean.unsqueeze(0)  # (1,1,1,1)
    if class_means.dim() == 3:
        class_means = class_means.unsqueeze(0)  # (1,3,1,1)
    if seg.dim() == 2:
        seg = seg.unsqueeze(0)  # (1,H,W)

    B, C, H, W = x.shape
    y = torch.zeros_like(x)

    # Normalize x
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    std = x.std(dim=[1, 2, 3], keepdim=True) + 1e-8
    x_norm = (x - mean) * (sigma / std)

    # Masks
    seg = seg.to(x.device)
    wt_mask = (seg > 0)          # WT
    tc_mask = torch.isin(seg, torch.tensor([1, 3], device=x.device))  # TC
    et_mask = (seg == 3)         # ET
    bg_mask = (seg == 0)

    # Broadcast means
    wt_mean = class_means[:, 0:1, :, :].to(x.device)  # (B,1,1,1)
    tc_mean = class_means[:, 1:2, :, :].to(x.device)
    et_mean = class_means[:, 2:3, :, :].to(x.device)
    g_mean = global_mean.to(x.device)

    # Apply class-aware targets
    for b in range(B):
        # background
        y[b, 0][bg_mask[b]] = (g_mean[b, 0, 0, 0] + x_norm[b, 0][bg_mask[b]])
        # WT
        y[b, 0][wt_mask[b]] = (wt_mean[b, 0, 0, 0] + x_norm[b, 0][wt_mask[b]])
        # TC
        y[b, 0][tc_mask[b]] = (tc_mean[b, 0, 0, 0] + x_norm[b, 0][tc_mask[b]])
        # ET
        y[b, 0][et_mask[b]] = (et_mean[b, 0, 0, 0] + x_norm[b, 0][et_mask[b]])

    return y


def histogram_separation_loss(class_means, margin=0.05):
    """
    Class-conditional histogram separation:
    encourage WT, TC, ET means to be separated by at least 'margin'.

    Args:
        class_means: (B,3,1,1) or (3,1,1)
    """
    if class_means.dim() == 3:
        class_means = class_means.unsqueeze(0)  # (1,3,1,1)

    # Use batch-averaged means
    mu = class_means.mean(dim=0).view(3)  # (3,)
    wt, tc, et = mu[0], mu[1], mu[2]

    loss = 0.0
    for a, b in [(wt, tc), (wt, et), (tc, et)]:
        diff = torch.abs(a - b)
        loss = loss + torch.relu(margin - diff)

    return loss


def label_guided_enhancement_loss(y_fake, class_means, seg):
    """
    Label-guided enhancement loss:
    enforce that enhanced intensities in each region are close to class-specific means.

    Args:
        y_fake: (B,1,H,W)
        class_means: (B,3,1,1) or (3,1,1)
        seg: (B,H,W) or (H,W)
    """
    if class_means.dim() == 3:
        class_means = class_means.unsqueeze(0)
    if seg.dim() == 2:
        seg = seg.unsqueeze(0)

    B, _, H, W = y_fake.shape
    seg = seg.to(y_fake.device)
    class_means = class_means.to(y_fake.device)

    wt_mean = class_means[:, 0:1, :, :]
    tc_mean = class_means[:, 1:2, :, :]
    et_mean = class_means[:, 2:3, :, :]

    loss = 0.0
    for b in range(B):
        yf = y_fake[b, 0]

        wt_mask = (seg[b] > 0)
        tc_mask = torch.isin(seg[b], torch.tensor([1, 3], device=y_fake.device))
        et_mask = (seg[b] == 3)

        if wt_mask.sum() > 0:
            loss = loss + torch.mean(torch.abs(yf[wt_mask] - wt_mean[b, 0, 0, 0]))
        if tc_mask.sum() > 0:
            loss = loss + torch.mean(torch.abs(yf[tc_mask] - tc_mean[b, 0, 0, 0]))
        if et_mask.sum() > 0:
            loss = loss + torch.mean(torch.abs(yf[et_mask] - et_mean[b, 0, 0, 0]))

    return loss / B
