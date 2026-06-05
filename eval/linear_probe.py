"""Linear probe for PASTIS — the headline downstream metric.

PASTIS is SEGMENTATION, so the probe is dense, not a single label per image (Common Mistake #9).
Recipe: FREEZE the pretrained encoder, attach a single 1x1 conv (linear) head on the per-pixel
feature map, train ONLY the head, report mIoU. Compare against supervised U-TAE (63.1 mIoU).

The probe measures representation quality: a strong frozen encoder + tiny linear head should
recover much of the supervised performance. Sanity checks define whether the probe itself works:
    - probe on a SUPERVISED-trained encoder -> approaches the U-TAE ballpark.
    - probe on a RANDOM-INIT encoder -> near chance (else the head is leaking / too powerful).
"""
from __future__ import annotations


def extract_dense_features(encoder, batch):
    """Run the FROZEN encoder, upsample token features back to (B, D, H, W) pixel grid.

    Note: features live on the P×P token grid; you must upsample (e.g. nearest/bilinear) to
    full resolution to align with per-pixel labels. TODO: implement; encoder in eval, no grad.
    """
    raise NotImplementedError("M3")


def linear_probe_segmentation(encoder, train_loader, val_loader, num_classes=19,
                              ignore_index=0, epochs=20):
    """Freeze encoder; train a 1x1-conv head -> per-pixel logits; report mIoU.

    TODO
        - freeze encoder params; head = nn.Conv2d(D, num_classes, 1).
        - train head with cross-entropy (ignore background if you chose to).
        - evaluate mIoU over classes (decide background handling, be consistent).
    Returns: dict(miou=..., per_class_iou=...).
    """
    raise NotImplementedError("M3")
