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
def extract_dense_features(encoder, batch, img_size=128, use_temporal=True, train_encoder=False):
    """Run the encoder (frozen by default), temporally pool, upsample tokens to (B, D, H, W).

    use_temporal=True  : spatiotemporal path (encode_temporal) — for methods that trained the
                         temporal encoder (temporal/spatial JEPA).
    use_temporal=False : spatial-only path (encode_full per frame) then masked-mean over time —
                         for baselines (MAE/BYOL/SimCLR) that train only the spatial encoder, so
                         their untrained temporal encoder doesn't corrupt the probe (fair eval).
    Then reshape to the (H',W') token grid -> bilinear upsample to full resolution.

    train_encoder=True leaves the encoder in whatever mode the caller set and is ONLY for the
    end-to-end supervised reference; every frozen-probe number in the paper is produced with the
    default, which forces eval() here so a caller cannot accidentally probe a training-mode encoder.
    """
    if not train_encoder:
        encoder.eval()
    data, dates, pad_mask = batch["data"], batch["dates"], batch["pad_mask"]
    if use_temporal:
        tok = encoder.encode_temporal(data, dates, pad_mask)     # (B, T, N, D)
    else:
        B, T = data.shape[:2]
        flat = data.reshape(B * T, *data.shape[2:])
        tok = encoder.encode_full(flat).reshape(B, T, encoder.num_patches, encoder.embed_dim)
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


def _sanitize_labels(label, num_classes, ignore_index):
    """Map any label outside [0, num_classes) to ignore_index. PASTIS void (19) and any stray
    out-of-range value become 'ignore' instead of triggering a CUDA device-side assert in
    cross_entropy (which would poison the CUDA context for the rest of the process)."""
    return torch.where((label >= 0) & (label < num_classes), label,
                       torch.full_like(label, ignore_index))


PROBE_HIDDEN = 256          # hidden width shared by every non-linear probe head
PROBE_RF = {"linear": 1, "conv": 3, "rf1": 1, "rf3": 3, "rf5": 5, "rf9": 9}


def _build_head(D, num_classes, head):
    """Probe head, graded by SPATIAL RECEPTIVE FIELD.

    The capacity axis that matters for the floor comparison is how much spatial mixing the probe
    can supply for itself. Raw bands carry no spatial context, so a 1x1 probe cannot recover any;
    a randomly initialised encoder supplies mixing through attention before the probe ever sees
    the features. Sweeping the receptive field is therefore the direct test of which floor binds.

    Ladder (n stacked 3x3 convs then a 1x1 classifier) gives RF = 2n+1:
      'linear' / 'rf1'  1x1                      RF 1   the strict linear-probe convention
      'conv'   / 'rf3'  3x3 -> 1x1               RF 3   the light dense decoder
      'rf5'             (3x3)x2 -> 1x1           RF 5
      'rf9'             (3x3)x4 -> 1x1           RF 9
    'linear' and 'conv' are kept as the canonical names for RF 1 and RF 3, so numbers reported
    before this sweep existed remain exactly the RF 1 and RF 3 points of the curve.
    """
    if head in ("linear", "rf1"):
        return nn.Conv2d(D, num_classes, 1)
    if head in ("conv", "rf3", "rf5", "rf9"):
        n_layers = {"conv": 1, "rf3": 1, "rf5": 2, "rf9": 4}[head]
        layers, c_in = [], D
        for _ in range(n_layers):
            layers += [nn.Conv2d(c_in, PROBE_HIDDEN, 3, padding=1), nn.GELU()]
            c_in = PROBE_HIDDEN
        layers.append(nn.Conv2d(c_in, num_classes, 1))
        return nn.Sequential(*layers)
    raise ValueError(f"unknown head {head!r}")


def linear_probe_segmentation(encoder, train_loader, val_loader, num_classes=20,
                              ignore_index=19, epochs=20, lr=1e-3, device=None,
                              use_temporal=True, head="linear", seed=0, freeze=True):
    """Train a head on encoder features -> per-pixel logits; report mIoU.

    freeze=True (default) is the frozen probe used for every cell in the paper: the encoder is
    eval() and its parameters get no gradient. freeze=False trains encoder and head together and
    exists for one purpose, the end-to-end SUPERVISED REFERENCE, which asks whether the protocol
    itself is capable of producing a usable segmenter at this budget. It is not a probe and must
    never be pooled with the frozen cells.

    use_temporal selects the feature pathway (see extract_dense_features): True for JEPA,
    False for spatial-only baselines. head: 'linear' (strict probe) or 'conv' (light decoder).
    `seed` fixes the head init + data-order RNG so the reported mIoU is reproducible (otherwise it
    varies ~±1 mIoU run-to-run — fine for the conclusion, bad for a results table).

    Returns: dict(miou=..., per_class_iou=tensor).
    """
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval() if freeze else encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(not freeze)

    # infer feature dim from one batch
    sample = next(iter(train_loader))
    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    with torch.no_grad():
        D = extract_dense_features(encoder, sample, use_temporal=use_temporal).shape[1]
    head = _build_head(D, num_classes, head).to(device)
    params = list(head.parameters()) + ([] if freeze else list(encoder.parameters()))
    opt = torch.optim.AdamW(params, lr=lr)

    for ep in range(epochs):
        head.train()
        if not freeze:
            encoder.train()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            feat = extract_dense_features(encoder, batch, use_temporal=use_temporal,
                                          train_encoder=not freeze)
            logits = head(feat)
            label = _sanitize_labels(batch["label"], num_classes, ignore_index)
            loss = F.cross_entropy(logits, label, ignore_index=ignore_index)
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    encoder.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            feat = extract_dense_features(encoder, batch, use_temporal=use_temporal)
            pred = head(feat).argmax(1).cpu()
            label = _sanitize_labels(batch["label"].cpu(), num_classes, ignore_index)
            conf += _confusion(pred, label, num_classes, ignore_index)
    miou, per_class = miou_from_confusion(conf, ignore_index)
    return {"miou": miou, "per_class_iou": per_class}
