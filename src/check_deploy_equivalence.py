"""Verify that the deployment conversion (BN folding + rFFT -> constant DFT matmul) preserves the
outputs of the *trained* LamyNet checkpoints, and record the evidence for the paper.

For every runs/litedpas_v5_*_1000ep*/best.pt the trained model and its converted copy are compared on
the first 2,000 COST2100 test samples of the matching scenario (max absolute output difference and
the test NMSE of both models). Results go to source/deploy_equivalence.json.

Usage: python -m src.check_deploy_equivalence --data-dir <COST2100 dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.cost2100 import COST2100Dataset
from src.deploy import to_deploy
from src.metrics import sample_nmse_linear
from src.train_baseline import build_model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=root / "runs")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=root / "source" / "deploy_equivalence.json")
    args = parser.parse_args()

    torch.set_num_threads(2)
    records = []
    for run_dir in sorted(args.runs_dir.glob("litedpas_v5_*_1000ep*")):
        best = run_dir / "best.pt"
        if not best.is_file() or "_abl-" in run_dir.name:
            continue
        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        model = build_model(ckpt["model_name"], root, encoded_dim=int(ckpt["encoded_dim"]), model_kwargs=ckpt.get("model_kwargs") or {})
        model.load_state_dict(ckpt["model"])
        model.eval()
        deploy = to_deploy(model).eval()
        scenario = ckpt.get("scenario", "indoor")
        suffix = "in" if scenario == "indoor" else "out"
        data = COST2100Dataset(args.data_dir / f"DATA_Htest{suffix}.mat").data[: args.samples]
        max_diff, ratios_m, ratios_d = 0.0, [], []
        with torch.no_grad():
            for i in range(0, data.shape[0], 200):
                x = data[i : i + 200]
                ym, yd = model(x), deploy(x)
                max_diff = max(max_diff, float((ym - yd).abs().max()))
                ratios_m.append(sample_nmse_linear(x, ym))
                ratios_d.append(sample_nmse_linear(x, yd))
        nm = 10 * np.log10(torch.cat(ratios_m).mean().item())
        nd = 10 * np.log10(torch.cat(ratios_d).mean().item())
        rec = {
            "run": run_dir.name, "scenario": scenario, "compression_ratio": 2048 // int(ckpt["encoded_dim"]),
            "samples": int(data.shape[0]), "max_abs_output_diff": max_diff,
            "test_nmse_db_trained": round(float(nm), 4), "test_nmse_db_deployed": round(float(nd), 4),
            "parameters_trained": sum(p.numel() for p in model.parameters()),
            "parameters_deployed": sum(p.numel() for p in deploy.parameters()),
        }
        records.append(rec)
        print(f"{run_dir.name}: max |diff| = {max_diff:.2e}, NMSE {nm:.4f} -> {nd:.4f} dB, params {rec['parameters_trained']} -> {rec['parameters_deployed']}")
    args.out.write_text(json.dumps({"description": "trained LamyNet checkpoints vs. deployment conversion (src.deploy.to_deploy)", "results": records}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}; overall max |diff| = {max(r['max_abs_output_diff'] for r in records):.2e}")


if __name__ == "__main__":
    main()
