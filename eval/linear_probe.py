"""Linear probe for PASTIS — the headline downstream metric.

PASTIS is SEGMENTATION, so the probe is dense, not a single label per image (Common Mistake #9).
Recipe: FREEZE the pretrained encoder, attach a single 1x1 conv (linear) head on the per-pixel
feature map, train ONLY the head, report mIoU. Compare against supervised U-TAE (63.1 mIoU).

Sanity checks define whether the probe itself works:
    - probe on a SUPERVISED-trained encoder -> approaches the U-TAE ballpark.
    - probe on a RANDOM-INIT encoder -> near chance (else the head is leaking / too powerful).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def extract_dense_features(encoder, batch, img_size=128):
    """Run the FROZEN encoder, temporally pool, upsample token features to (B, D, H, W).

    Uses the spatiotemporal path: encode_temporal -> (B, T, N, D); masked-mean over real frames
    -> (B, N, D); reshape to the (H', W') token grid -> bilinear upsample to full resolution.
    """
    encoder.eval()
    data, dates, pad_mask = batch["data"], batch["dates"], batch["pad_mask"]
    tok = encoder.encode_temporal(data, dates, pad_mask)          # (B, T, N, D)
    m = pad_mask.float()[:, :, None, None]                        # (B, T, 1, 1)
    pooled = (tok * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)   # (B, N, D)
    B, N, D = pooled.shape
    Hp, Wp = encoder.grid_hw
    feat = pooled.transpose(1, 2).reshape(B, D, Hp, Wp)           # (B, D, H', W')
    return F.interpolate(feat, size=(img_size, img_size), mode="bilinear", align_corners=False)


def _confusion(pred, label, num_classes, ignore_index):
    mask = label != ignore_index
    p = pred[mask].view(-1)
    t = label[mask].view(-1)
    k = (t * num_classes + p)
    return torch.bincount(k, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def miou_from_confusion(conf, ignore_index=None):
    inter = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter / union.clamp_min(1)
    valid = union > 0
    if ignore_index is not None and 0 <= ignore_index < conf.shape[0]:
        valid[ignore_index] = False
    return iou[valid].mean().item(), iou


def linear_probe_segmentation(encoder, train_loader, val_loader, num_classes=19,
                              ignore_index=0, epochs=20, lr=1e-3, device=None):
    """Freeze encoder; train a 1x1-conv head -> per-pixel logits; report mIoU.

    Returns: dict(miou=..., per_class_iou=tensor).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # infer feature dim from one batch
    sample = next(iter(train_loader))
    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    D = extract_dense_features(encoder, sample).shape[1]
    head = nn.Conv2d(D, num_classes, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    for ep in range(epochs):
        head.train()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            feat = extract_dense_features(encoder, batch)
            logits = head(feat)
            loss = F.cross_entropy(logits, batch["label"], ignore_index=ignore_index)
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            feat = extract_dense_features(encoder, batch)
            pred = head(feat).argmax(1).cpu()
            conf += _confusion(pred, batch["label"].cpu(), num_classes, ignore_index)
    miou, per_class = miou_from_confusion(conf, ignore_index)
    return {"miou": miou, "per_class_iou": per_class}
