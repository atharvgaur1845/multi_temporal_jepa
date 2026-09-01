"""Probe-capacity sweep: which trivial floor binds, as a function of probe receptive field.

The paper reports a reversal between the two floors at RF 1 (linear) and RF 3 (conv). The
mechanism we propose is that raw bands carry no spatial context, so a probe that cannot mix
spatially cannot use them, while a randomly initialised encoder supplies mixing through attention
before the probe sees anything. If that is the mechanism, the two floor curves must CROSS at an
identifiable receptive field. This sweeps RF in {1,3,5,9} and locates the crossing.

Encoders are rebuilt from each saved checkpoint exactly as scripts/run_matrix.py builds them and
probed with the same linear_probe_segmentation call, so the RF 1 and RF 3 points reproduce the
published linear/conv columns rather than merely resembling them.

Usage:
  python REO-2/probe_capacity_sweep.py --cells raw_features,random --seeds 0,1,2 --heads rf1,rf3,rf5,rf9
"""
from __future__ import annotations

import argparse, copy, csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_encoder_from_ckpt(ckpt, data_cfg, device):
    """Reconstruct the frozen encoder a checkpoint came from. Mirrors run_matrix.py cell dispatch."""
    import torch
    from models.jepa import build_model
    cfg, obj = ckpt["cfg"], ckpt["objective"]
    if obj == "raw":
        from models.raw_encoder import RawPatchEncoder
        enc = RawPatchEncoder(patch_size=cfg["encoder"]["patch_size"],
                              in_chans=data_cfg.get("bands", 10)).to(device)
        return enc, False
    if obj in ("random", "temporal_jepa", "spatial_jepa"):
        # build_model asserts objective in {spatial_jepa, temporal_jepa}; 'random' is the same
        # architecture as temporal_jepa, never trained (run_matrix.py does this same swap).
        mcfg = copy.deepcopy(cfg)
        if obj == "random":
            mcfg["objective"] = "temporal_jepa"
        m = build_model(mcfg).to(device)
        enc = m.target_encoder
        m.context_encoder = None; m.predictor = None
        return enc, True
    raise SystemExit(f"[sweep] objective {obj!r} has no per-seed checkpoint path; skipping is safer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="raw_features,random,tjepa_h1,spatial_jepa")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--heads", default="rf1,rf3,rf5,rf9")
    ap.add_argument("--data", default="configs/data/pastis.yaml")
    ap.add_argument("--ckpt-dir", default="runs/matrix")
    ap.add_argument("--out", default="REO-2/evidence/probe_capacity.csv")
    ap.add_argument("--probe-epochs", type=int, default=15)
    # run_matrix.py never passes `seed=` to linear_probe_segmentation, so EVERY published probe
    # used head-init seed 0 regardless of the cell's training seed. Match that here or the RF 1 and
    # RF 3 points will not line up with the published linear and conv columns.
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from data.pastis_dataset import PASTIS, collate_variable_length
    from data.transforms import compute_band_stats
    from eval.linear_probe import linear_probe_segmentation, PROBE_RF
    from scripts.run_matrix import load_yaml
    from utils.device import resolve_device
    from utils.seed import seed_everything

    device = resolve_device(args.device)
    data_cfg = load_yaml(args.data)
    msl = data_cfg.get("max_seq_len")

    # identical split + normalisation pipeline to run_matrix.py
    train = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=False,
                   max_seq_len=msl, subsample_train=True)
    mean, std = compute_band_stats(train, max_samples=200)
    eval_ds = PASTIS(data_cfg["root"], folds=data_cfg["val_folds"], return_label=True,
                     norm_mean=mean, norm_std=std, max_seq_len=msl)
    probe_tr = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=True,
                      norm_mean=mean, norm_std=std, max_seq_len=msl)
    el = DataLoader(eval_ds, batch_size=8, collate_fn=collate_variable_length)
    tl = DataLoader(probe_tr, batch_size=8, shuffle=True, collate_fn=collate_variable_length)

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    done = set()
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    if not new:
        with open(args.out) as fh:
            for r in csv.DictReader(fh):
                done.add((r["cell"], r["seed"], r["head"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w" if new else "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["cell", "objective", "seed", "head", "receptive_field", "miou",
                        "probe_epochs", "probe_seed"])
            f.flush()
        for cell in cells:
            for seed in seeds:
                ck = os.path.join(args.ckpt_dir, f"{cell}__s{seed}.pt")
                if not os.path.exists(ck):
                    print(f"[sweep] MISSING {ck} -- skipped", flush=True)
                    continue
                ckpt = torch.load(ck, map_location=device, weights_only=False)
                for head in heads:
                    if (cell, str(seed), head) in done:
                        print(f"[sweep] {cell} s{seed} {head}: already done", flush=True)
                        continue
                    # rebuild per head so a head never inherits state from the previous probe
                    seed_everything(seed)
                    enc, use_temporal = build_encoder_from_ckpt(ckpt, data_cfg, device)
                    enc.load_state_dict(ckpt["encoder"])
                    res = linear_probe_segmentation(
                        enc, tl, el, num_classes=data_cfg["num_classes"],
                        ignore_index=data_cfg["ignore_index"], epochs=args.probe_epochs,
                        device=device, use_temporal=use_temporal, head=head,
                        seed=args.probe_seed)
                    # Two sweeps may run concurrently against one CSV; without a lock their
                    # appends interleave and truncate each other's rows. That already cost a set
                    # of results once in this project (see run_matrix.py).
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.seek(0, os.SEEK_END)
                        w.writerow([cell, ckpt["objective"], seed, head, PROBE_RF[head],
                                    round(res["miou"], 4), args.probe_epochs, args.probe_seed])
                        f.flush(); os.fsync(f.fileno())
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    print(f"[sweep] {cell} s{seed} {head} (RF{PROBE_RF[head]}): "
                          f"mIoU {res['miou']*100:.2f}", flush=True)
                    del enc
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                del ckpt


if __name__ == "__main__":
    main()
