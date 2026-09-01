#!/usr/bin/env python3
"""Measure collapse diagnostics on the SAVED encoders, VICReg on vs off.

The per-step traces printed during training were not written to a file, so we measure
the same quantities directly from the checkpoints. This is a defensible substitute and
arguably the more relevant measurement: it evaluates the encoder that is actually
probed (the EMA target encoder), on held-out data, under an identical protocol for
both cells.

Reports per-dimension std, effective rank (of 512) and off-diagonal covariance for
tjepa_h1 (VICReg on) and tjepa_noreg (VICReg off), both seed 0, both batch 12 x 16.

    python REO-2/figures/collapse_diagnostics.py
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
from torch.utils.data import DataLoader
from data.pastis_dataset import PASTIS, collate_variable_length
from data.transforms import compute_band_stats
from engine.diagnostics import per_dim_std, effective_rank, offdiag_covariance
from models.jepa import build_model
from utils.config import load_yaml
from utils.device import resolve_device

CKPTS = [("VICReg on  (tjepa_h1, seed 0)",  "runs/matrix/tjepa_h1__s0.pt"),
         ("VICReg off (tjepa_noreg)",       "runs/matrix/tjepa_noreg.pt")]
N_PATCHES = 96


def main():
    os.environ.setdefault("TJEPA_MMAP", "1")
    dev = resolve_device("cuda")
    dcfg = load_yaml("configs/data/pastis.yaml")
    msl = dcfg.get("max_seq_len")

    train = PASTIS(dcfg["root"], folds=dcfg["train_folds"], return_label=False,
                   max_seq_len=msl, subsample_train=True)
    mean, std = compute_band_stats(train, max_samples=200)
    ev = PASTIS(dcfg["root"], folds=dcfg["val_folds"], return_label=False,
                norm_mean=mean, norm_std=std, max_seq_len=msl)
    loader = DataLoader(ev, batch_size=4, collate_fn=collate_variable_length, num_workers=2)

    out = {}
    for label, path in CKPTS:
        st = torch.load(path, map_location="cpu", weights_only=False)
        cfg = st["cfg"]
        cfg["objective"] = "temporal_jepa"
        m = build_model(cfg)
        enc = m.target_encoder
        enc.load_state_dict(st["encoder"])
        enc = enc.to(dev).eval()

        embs, seen = [], 0
        with torch.no_grad():
            for b in loader:
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                with torch.autocast("cuda", enabled=True):
                    # Same pathway the frozen probe uses for JEPA cells
                    # (eval/linear_probe.extract_dense_features, use_temporal=True),
                    # so the diagnostic describes the representation actually evaluated.
                    tok = enc.encode_temporal(b["data"], b["dates"], b["pad_mask"])  # (B,T,N,D)
                m = b["pad_mask"].float()[:, :, None, None]
                pooled = (tok.float() * m).sum(1) / m.sum(1).clamp_min(1.0)          # (B,N,D)
                embs.append(pooled.flatten(0, 1).cpu())
                seen += b["data"].shape[0]
                if seen >= N_PATCHES:
                    break
        Z = torch.cat(embs, 0)
        r = {"per_dim_std": round(per_dim_std(Z), 4),
             "effective_rank": round(effective_rank(Z), 2),
             "offdiag_cov": round(offdiag_covariance(Z), 5),
             "dim": Z.shape[-1], "n_tokens": Z.shape[0]}
        out[label] = r
        print(f"{label}\n    per-dim std {r['per_dim_std']:.4f}   "
              f"effective rank {r['effective_rank']:.2f} of {r['dim']}   "
              f"off-diag cov {r['offdiag_cov']:.5f}   ({r['n_tokens']} tokens)")
        del m, enc
        torch.cuda.empty_cache()

    with open("REO-2/evidence/collapse_diagnostics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[saved] REO-2/evidence/collapse_diagnostics.json")


if __name__ == "__main__":
    main()
