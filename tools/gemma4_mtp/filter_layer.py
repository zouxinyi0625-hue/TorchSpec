#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
"""
Filter one layer out of the merged MaiProfile train/eval splits, and cap eval.

For a single-layer run (e.g. layer1_delta), pull that layer's rows from the
merged 90/10 split files (each row has "source_layer"), preserving the clean
split (no train/eval leakage). Eval is capped to --eval-cap rows.

Usage:
    MNT=$AZURE_ML_INPUT_ukwdata/maiprofile
    python tools/gemma4_mtp/filter_layer.py \
        --train $MNT/mtp_26b/split/train_maiprofile_26b.jsonl \
        --eval  $MNT/mtp_26b/split/eval_maiprofile_26b.jsonl \
        --layer layer1_delta \
        --out-dir ./data \
        --eval-cap 1000
"""
import argparse
import json
import os
import random


def _norm(name: str) -> str:
    for pre in ("train_", "eval_", "maiprofile_"):
        if name.startswith(pre):
            name = name[len(pre):]
    return name


def filter_layer(src, layer, cap=0, seed=42):
    rows = []
    total = bad = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                bad += 1
                continue
            if _norm(str(row.get("source_layer", ""))) == layer:
                rows.append(row)
    if cap and len(rows) > cap:
        random.Random(seed).shuffle(rows)
        rows = rows[:cap]
    return rows, total, bad


def write(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--layer", default="layer1_delta")
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--eval-cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tr, tr_tot, tr_bad = filter_layer(args.train, args.layer, cap=0, seed=args.seed)
    ev, ev_tot, ev_bad = filter_layer(args.eval, args.layer, cap=args.eval_cap, seed=args.seed)

    tr_path = os.path.join(args.out_dir, f"train_{args.layer}.jsonl")
    ev_path = os.path.join(args.out_dir, f"eval_{args.layer}_{args.eval_cap}.jsonl")
    write(tr, tr_path)
    write(ev, ev_path)

    print(f"train: scanned {tr_tot} ({tr_bad} bad) -> {len(tr)} '{args.layer}' rows -> {tr_path}")
    print(f"eval : scanned {ev_tot} ({ev_bad} bad) -> {len(ev)} '{args.layer}' rows "
          f"(capped at {args.eval_cap}) -> {ev_path}")


if __name__ == "__main__":
    main()
