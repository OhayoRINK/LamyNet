"""Evaluate every finished run under ``runs/`` on the COST2100 test split.

Usage::

    python -m src.evaluate_runs --data-dir <COST2100 dir> [--runs-dir runs] [--out source/experiment_results.json]

For each run folder containing ``summary.json`` and ``best.pt`` the model is
rebuilt from the stored configuration (model name, encoded_dim, ablation
``model_kwargs``), evaluated on the validation and test splits (NMSE in dB and
CsiNet-style cosine similarity rho on the test split), and profiled (parameters, Conv/Linear MACs). The result
is one JSON record per run plus a flat CSV, so ablation / baseline / seed tables
can be assembled without re-running anything. Test data is read only here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cost2100 import COST2100Dataset
from src.metrics import sample_nmse_linear
from src.models.lite_dpas import count_parameters
from src.profile_lite_dpas import approximate_macs
from src.train_baseline import build_model


def load_raw_test(data_dir: Path, scenario: str) -> torch.Tensor:
    """Full spatial-frequency test channels ``HF_all`` as complex [N, 32 antennas, 125 subcarriers]."""
    import scipy.io

    suffix = "in" if scenario == "indoor" else "out"
    raw = scipy.io.loadmat(data_dir / f"DATA_HtestF{suffix}_all.mat", variable_names=["HF_all"])["HF_all"]
    return torch.from_numpy(np.asarray(raw, dtype=np.complex64)).reshape(raw.shape[0], 32, 125)


def rho_sum(estimate: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """Sum over the batch of the CsiNet cosine similarity rho.

    Follows the public CsiNet/CRNet evaluator: the reconstructed angular-delay
    matrix (offset 0.5 removed) is zero-padded along the delay axis to 257
    taps, transformed back to the frequency domain with an unnormalised FFT,
    truncated to the first 125 subcarriers, and compared with the raw channel
    per subcarrier; rho is the mean over subcarriers of
    |h_hat^H h| / (||h_hat|| ||h||).
    """
    pred = torch.complex(estimate[:, 0] - 0.5, estimate[:, 1] - 0.5)  # [B, 32 ant, 32 delay]
    pred = torch.nn.functional.pad(pred, (0, 257 - 32))
    pred_f = torch.fft.fft(pred, dim=2)[:, :, :125]  # [B, 32, 125]
    cross = (pred_f.conj() * raw).sum(dim=1).abs()  # [B, 125]
    norm = pred_f.abs().square().sum(dim=1).sqrt() * raw.abs().square().sum(dim=1).sqrt()
    return (cross / norm.clamp_min(1e-12)).mean(dim=1).sum()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, raw: torch.Tensor | None = None
) -> tuple[float, float | None]:
    """Return (NMSE dB, rho). rho needs the raw frequency-domain channels (test split only)."""
    ratios, rho_total, offset = [], 0.0, 0
    for x in loader:
        x = x.to(device)
        y = model(x)
        ratios.append(sample_nmse_linear(x, y).cpu())
        if raw is not None:
            rho_total += float(rho_sum(y, raw[offset : offset + x.shape[0]].to(device)))
        offset += x.shape[0]
    nmse_db = 10.0 * np.log10(max(torch.cat(ratios).mean().item(), 1e-12))
    return float(nmse_db), (rho_total / offset if raw is not None else None)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=root / "runs")
    parser.add_argument("--out", type=Path, default=root / "source" / "experiment_results.json")
    parser.add_argument("--only", type=str, default=None, help="substring filter on run folder names")
    parser.add_argument("--skip-existing", action="store_true", help="reuse records already in --out")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders: dict[tuple[str, str], DataLoader] = {}

    def loader(scenario: str, split: str) -> DataLoader:
        key = (scenario, split)
        if key not in loaders:
            suffix = "in" if scenario == "indoor" else "out"
            ds = COST2100Dataset(args.data_dir / f"DATA_H{split}{suffix}.mat")
            loaders[key] = DataLoader(ds, batch_size=500, shuffle=False, num_workers=0)
        return loaders[key]

    raws: dict[str, torch.Tensor] = {}

    def raw_test(scenario: str) -> torch.Tensor:
        if scenario not in raws:
            raws[scenario] = load_raw_test(args.data_dir, scenario)
        return raws[scenario]

    existing: dict[str, dict] = {}
    if args.skip_existing and args.out.is_file():
        for rec in json.loads(args.out.read_text(encoding="utf-8")):
            existing[rec["run"]] = rec

    records = []
    for run_dir in sorted(args.runs_dir.iterdir()):
        summary_path, best_path = run_dir / "summary.json", run_dir / "best.pt"
        if not (summary_path.is_file() and best_path.is_file()):
            continue
        if args.only and args.only not in run_dir.name:
            continue
        if run_dir.name in existing:
            records.append(existing[run_dir.name])
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        name = ckpt["model_name"]
        encoded_dim = int(ckpt["encoded_dim"])
        model_kwargs = ckpt.get("model_kwargs") or summary.get("model_kwargs") or {}
        model = build_model(name, root, encoded_dim=encoded_dim, model_kwargs=model_kwargs)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        scenario = summary.get("scenario", "indoor")

        val_db, _ = evaluate(model, loader(scenario, "val"), device)
        test_db, test_rho = evaluate(model, loader(scenario, "test"), device, raw_test(scenario))
        sample = torch.zeros(1, 2, 32, 32, device=device)
        record = {
            "run": run_dir.name,
            "group": summary.get("group"),
            "model": name,
            "scenario": scenario,
            "compression_ratio": 2048 // encoded_dim,
            "encoded_dim": encoded_dim,
            "epochs": summary.get("epochs"),
            "seed": summary.get("seed"),
            "loss_mode": summary.get("loss_mode"),
            "scheduler": summary.get("scheduler"),
            "model_kwargs": model_kwargs,
            "best_epoch": int(ckpt.get("epoch", -1)),
            "val_nmse_db": round(val_db, 3),
            "test_nmse_db": round(test_db, 3),
            "test_rho": round(test_rho, 5),
            "parameters": count_parameters(model),
            "approx_macs": approximate_macs(model, sample),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "seconds_per_epoch": summary.get("seconds_per_epoch"),
            "device": summary.get("device"),
        }
        records.append(record)
        print(
            f"{run_dir.name}: test NMSE {test_db:.3f} dB, rho {test_rho:.4f}, "
            f"params {record['parameters']:,}, best epoch {record['best_epoch']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    if records:
        fields = [k for k in records[0] if k != "model_kwargs"] + ["model_kwargs"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in records:
                row = dict(rec)
                row["model_kwargs"] = json.dumps(rec["model_kwargs"], sort_keys=True)
                writer.writerow(row)
    print(f"Wrote {len(records)} records to {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
