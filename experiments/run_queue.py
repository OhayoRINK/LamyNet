"""Run the paper-revision experiment queue on one GPU.

Jobs are executed in the listed order (ablation -> CsiNet reproduction ->
CRNet/CLNet reproduction -> LamyNet re-run on this machine) with ``--workers`` concurrent processes. A job whose run
folder already holds ``summary.json`` is skipped, so the queue can be restarted
after an interruption; a run folder that exists without ``summary.json`` is
resumed from its newest ``checkpoint_epoch_*.pt`` when one exists, otherwise it
is removed and started again.

Usage::

    python experiments/run_queue.py --data-dir <COST2100 dir> --workers 2
    python experiments/run_queue.py --list          # print the queue only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LOGS = ROOT / "experiments" / "logs"

ABLATION_EPOCHS = 300
FULL_EPOCHS = 1000
ABLATION_SETTINGS = [("indoor", 16), ("indoor", 64), ("outdoor", 16), ("outdoor", 64)]
ALL_SETTINGS = [(s, cr) for cr in (16, 64, 4, 32) for s in ("indoor", "outdoor")]

# tag -> extra CLI flags for src.train_lite_dpas
ABLATION_VARIANTS = {
    "full": [],
    "nogate": ["--no-spectral-gate"],
    "nodelay": ["--no-delay-path"],
    "k5": ["--delay-kernel", "5"],
    "latent8": ["--latent-grid", "8"],
    "bilinear": ["--upsample", "bilinear"],
}


def run_name(model: str, scenario: str, cr: int, epochs: int, seed: int, tag: str | None) -> str:
    name = f"{model}_{scenario}_cr{cr}_{epochs}ep"
    if seed != 42:
        name += f"_seed{seed}"
    if tag:
        name += f"_{tag}"
    return name


def build_jobs() -> list[dict]:
    jobs = []
    # 1. Ablation (300 epochs, CR 1/16 and 1/64, indoor + outdoor)
    for tag, flags in ABLATION_VARIANTS.items():
        for scenario, cr in ABLATION_SETTINGS:
            jobs.append(
                {
                    "phase": "1_ablation",
                    "model": "litedpas_v5",
                    "scenario": scenario,
                    "cr": cr,
                    "epochs": ABLATION_EPOCHS,
                    "seed": 42,
                    "tag": f"abl-{tag}",
                    "flags": flags + ["--group", "ablation"],
                }
            )
    # 2. CsiNet reproduced with the LamyNet recipe (NMSE loss, ReduceLROnPlateau, 1000 epochs)
    for scenario, cr in ALL_SETTINGS:
        jobs.append(
            {
                "phase": "2_csinet",
                "model": "csinet",
                "scenario": scenario,
                "cr": cr,
                "epochs": FULL_EPOCHS,
                "seed": 42,
                "tag": None,
                "flags": ["--scheduler", "on", "--group", "csinet_repro"],
            }
        )
    # 3. CRNet / CLNet (authors' code) reproduced with the LamyNet recipe; high compression ratios first
    for scenario, cr in [(s, cr) for cr in (64, 16, 32, 4) for s in ("indoor", "outdoor")]:
        for model in ("crnet", "clnet"):
            jobs.append(
                {
                    "phase": "3_baselines",
                    "model": model,
                    "scenario": scenario,
                    "cr": cr,
                    "epochs": FULL_EPOCHS,
                    "seed": 42,
                    "tag": None,
                    "flags": ["--scheduler", "on", "--group", "baseline_repro"],
                }
            )
    # 4. LamyNet itself re-trained on this machine (same code and recipe as the Colab runs) so that
    #    every model in the comparison table comes from one hardware / library environment
    for scenario, cr in [(s, cr) for cr in (64, 16, 32, 4) for s in ("indoor", "outdoor")]:
        jobs.append(
            {
                "phase": "4_lamynet_local",
                "model": "litedpas_v5",
                "scenario": scenario,
                "cr": cr,
                "epochs": FULL_EPOCHS,
                "seed": 42,
                "tag": "local",
                "flags": ["--group", "lamynet_local"],
            }
        )
    # 5. Activation study: SiLU -> LeakyReLU(0.3), which ONNX Runtime fuses into the convolutions.
    #    Same 300-epoch settings as the ablation so it compares directly with the abl-full runs.
    for scenario, cr in ABLATION_SETTINGS:
        jobs.append(
            {
                "phase": "5_activation",
                "model": "litedpas_v5",
                "scenario": scenario,
                "cr": cr,
                "epochs": ABLATION_EPOCHS,
                "seed": 42,
                "tag": "abl-leaky",
                "flags": ["--activation", "leakyrelu", "--group", "activation_study"],
            }
        )
    for job in jobs:
        job["run"] = run_name(job["model"], job["scenario"], job["cr"], job["epochs"], job["seed"], job["tag"])
    return jobs


def command(job: dict, data_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.train_lite_dpas",
        "--data-dir", str(data_dir),
        "--model", job["model"],
        "--scenario", job["scenario"],
        "--compression-ratio", str(job["cr"]),
        "--epochs", str(job["epochs"]),
        "--seed", str(job["seed"]),
        "--save-every", "50",
    ]
    if job["tag"]:
        cmd += ["--run-tag", job["tag"]]
    run_dir = RUNS / job["run"]
    if run_dir.is_dir() and not (run_dir / "summary.json").is_file():
        checkpoints = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            cmd += ["--resume-from", str(checkpoints[-1])]
        else:
            shutil.rmtree(run_dir)
    return cmd + job["flags"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--phase", type=str, default=None, help="run only jobs whose phase starts with this")
    args = parser.parse_args()

    jobs = build_jobs()
    if args.phase:
        jobs = [j for j in jobs if j["phase"].startswith(args.phase)]
    if args.list:
        for i, job in enumerate(jobs, 1):
            print(f"{i:3d} {job['phase']:12s} {job['run']}")
        print(f"{len(jobs)} jobs")
        return
    if args.data_dir is None:
        parser.error("--data-dir is required")

    LOGS.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "queue_status.json"
    pending = [j for j in jobs if not (RUNS / j["run"] / "summary.json").is_file()]
    done = [j["run"] for j in jobs if j not in pending]
    running: dict[str, tuple[subprocess.Popen, dict, float]] = {}
    failed: list[str] = []
    print(f"{len(done)} already finished, {len(pending)} pending, workers={args.workers}", flush=True)

    def write_status() -> None:
        status_path.write_text(
            json.dumps(
                {
                    "updated": datetime.now().isoformat(timespec="seconds"),
                    "done": done,
                    "failed": failed,
                    "running": {k: {"started": v[2]} for k, v in running.items()},
                    "pending": [j["run"] for j in pending],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    while pending or running:
        while pending and len(running) < args.workers:
            job = pending.pop(0)
            log = (LOGS / f"{job['run']}.log").open("a", encoding="utf-8")
            log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} =====\n")
            log.flush()
            proc = subprocess.Popen(command(job, args.data_dir), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running[job["run"]] = (proc, job, time.time())
            print(f"[start] {job['run']}", flush=True)
            write_status()
        time.sleep(30)
        for run, (proc, job, started) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[run]
            hours = (time.time() - started) / 3600
            if code == 0 and (RUNS / run / "summary.json").is_file():
                done.append(run)
                print(f"[done ] {run} ({hours:.2f} h)", flush=True)
            else:
                failed.append(run)
                print(f"[FAIL ] {run} exit={code} ({hours:.2f} h) see {LOGS / (run + '.log')}", flush=True)
            write_status()
    print(f"Queue finished: {len(done)} done, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
