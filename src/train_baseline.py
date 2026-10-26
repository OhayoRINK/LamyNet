"""Shared train/validation-only runner for short COST2100 trend experiments."""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cost2100 import COST2100Dataset
from src.metrics import sample_nmse_linear
from src.models.csinet import CsiNet
from src.models.lite_dpas import LiteDPASCsiNet, LiteDPASCsiNetV2
from src.upstream_models import build_upstream_model


LEARNING_RATES = {
    "csinet": 1e-3,
    "crnet": 1e-3,
    "clnet": 1e-3,
    "transnet": 1e-4,
    "litedpas": 1e-3,
    "litedpas_v2": 1e-3,
    "litedpas_v4": 1e-3,
    "litedpas_v5": 1e-3,
}


def build_model(
    name: str,
    project_root: str | Path,
    encoded_dim: int = 512,
    model_kwargs: dict | None = None,
) -> torch.nn.Module:
    """Build a model; ``model_kwargs`` carries LamyNet (litedpas_v5) ablation options."""
    model_kwargs = dict(model_kwargs or {})
    if model_kwargs and name != "litedpas_v5":
        raise ValueError(f"model_kwargs are only supported for litedpas_v5, got {name}")
    if name == "csinet":
        return CsiNet(encoded_dim=encoded_dim)
    if name == "litedpas":
        return LiteDPASCsiNet(encoded_dim=encoded_dim)
    if name == "litedpas_v2":
        return LiteDPASCsiNetV2(encoded_dim=encoded_dim)
    if name == "litedpas_v4":
        from src.models.lite_dpas import LiteDPASCsiNetV4

        return LiteDPASCsiNetV4(encoded_dim=encoded_dim)
    if name == "litedpas_v5":
        from src.models.lite_dpas import LiteDPASCsiNetV5

        return LiteDPASCsiNetV5(encoded_dim=encoded_dim, **model_kwargs)
    return build_upstream_model(name, project_root, encoded_dim=encoded_dim)


def _load_dpas_teacher(
    teacher_repo: str | Path,
    teacher_checkpoint: str | Path,
    encoded_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load the public DPAS implementation and freeze its pretrained weights."""
    repo = Path(teacher_repo)
    candidates = [repo / "models" / "dpas_csinet_v5.py", repo / "models" / "dpas_csinet.py"]
    model_file = next((path for path in candidates if path.is_file()), None)
    if model_file is None:
        raise FileNotFoundError(
            "Could not find models/dpas_csinet_v5.py or models/dpas_csinet.py under "
            f"teacher_repo={repo}"
        )

    spec = importlib.util.spec_from_file_location("lite_dpas_teacher_arch", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import DPAS model definition: {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    teacher = module.DPASCsiNet(feedback_bits=encoded_dim)

    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Teacher checkpoint must contain a PyTorch state_dict")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    teacher.load_state_dict(state, strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_history(history: list[dict], output_dir: Path) -> None:
    """Atomically persist the epoch log so Colab disconnects lose little metadata."""
    if not history:
        return
    destination = output_dir / "history.csv"
    temporary = output_dir / "history.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(destination)


def _save_periodic_checkpoint(
    path: Path,
    *,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_linear: float,
    history: list[dict],
    encoded_dim: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    loss_mode: str,
    scenario: str,
    teacher_checkpoint: str | Path | None,
    distill_weight: float,
) -> None:
    """Save a resumable state via atomic replacement."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_name": name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_nmse_linear": best_linear,
            "history": history,
            "encoded_dim": encoded_dim,
            "batch_size": batch_size,
            "seed": seed,
            "learning_rate": learning_rate,
            "loss_mode": loss_mode,
            "scenario": scenario,
            "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else None,
            "distill_weight": distill_weight,
        },
        temporary,
    )
    temporary.replace(path)


def train_model(
    name: str,
    project_root: str | Path,
    data_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 200,
    seed: int = 42,
    encoded_dim: int = 512,
    run_tag: str | None = None,
    save_every: int = 100,
    resume_from: str | Path | None = None,
    scenario: str = "indoor",
    loss_mode: str = "mse",
    scheduler_patience: int = 25,
    teacher_checkpoint: str | Path | None = None,
    teacher_repo: str | Path | None = None,
    distill_weight: float = 0.25,
    model_kwargs: dict | None = None,
    use_scheduler: bool | None = None,
    group: str | None = None,
) -> Path:
    """Train from scratch, select by validation NMSE, and never read test data.

    ``use_scheduler=None`` keeps the historical behaviour (ReduceLROnPlateau only
    for litedpas_v4/v5); ``True``/``False`` forces it on or off for any model so
    that baselines can be trained with exactly the LamyNet recipe. ``group`` is a
    free-form label (e.g. "ablation", "csinet_repro", "seed") stored in the
    summary for later aggregation.
    """
    model_kwargs = dict(model_kwargs or {})
    if name not in LEARNING_RATES:
        raise ValueError(f"Expected one of {tuple(LEARNING_RATES)}, got {name}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if encoded_dim < 1 or encoded_dim > 2048:
        raise ValueError("encoded_dim must be in [1, 2048]")
    if save_every < 1:
        raise ValueError("save_every must be positive")
    if loss_mode not in ("nmse", "mse"):
        raise ValueError("loss_mode must be 'nmse' or 'mse'")
    if scenario not in ("indoor", "outdoor"):
        raise ValueError("scenario must be 'indoor' or 'outdoor'")
    if scheduler_patience < 1:
        raise ValueError("scheduler_patience must be positive")
    if not 0.0 <= distill_weight <= 2.0:
        raise ValueError("distill_weight must be in [0, 2]")
    if (teacher_checkpoint is None) != (teacher_repo is None):
        raise ValueError("Set both teacher_checkpoint and teacher_repo, or leave both unset")
    if teacher_checkpoint is not None and name not in ("litedpas_v4", "litedpas_v5"):
        raise ValueError("DPAS distillation is currently enabled only for litedpas_v4/v5")

    project_root, data_dir = Path(project_root), Path(data_dir)
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    scenario_suffix = "in" if scenario == "indoor" else "out"
    train_data = COST2100Dataset(data_dir / f"DATA_Htrain{scenario_suffix}.mat")
    val_data = COST2100Dataset(data_dir / f"DATA_Hval{scenario_suffix}.mat")
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=pin_memory,
    )

    model = build_model(
        name, project_root, encoded_dim=encoded_dim, model_kwargs=model_kwargs
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATES[name])
    scheduler = None
    if use_scheduler is None:
        use_scheduler = name in ("litedpas_v4", "litedpas_v5")
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=scheduler_patience,
            threshold=0.05, threshold_mode="abs", min_lr=1e-5,
        )
    teacher = None
    if teacher_checkpoint is not None:
        teacher = _load_dpas_teacher(teacher_repo, teacher_checkpoint, encoded_dim, device)
        print(f"Loaded frozen DPAS teacher from: {teacher_checkpoint}")
    compression_ratio = 2048 // encoded_dim
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        output_dir = resume_path.parent
    else:
        run_name = f"{name}_{scenario}_cr{compression_ratio}_{epochs}ep"
        if seed != 42:
            run_name = f"{run_name}_seed{seed}"
        if name in ("litedpas", "litedpas_v2", "litedpas_v4", "litedpas_v5"):
            run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
            if not run_tag.replace("_", "").replace("-", "").isalnum():
                raise ValueError("run_tag may contain only letters, digits, underscores, and hyphens")
            run_name = f"{run_name}_{run_tag}"
        output_dir = project_root / "runs" / run_name
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Run folder already contains files: {output_dir}. Choose a new --run-tag to preserve prior results."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
    best_linear = float("inf")
    history: list[dict] = []
    start_epoch = 1
    if resume_from is not None:
        resumed = torch.load(resume_path, map_location=device, weights_only=False)
        if resumed.get("model_name") != name or int(resumed.get("encoded_dim", -1)) != encoded_dim:
            raise ValueError("Resume checkpoint model name or encoded_dim does not match this run")
        if resumed.get("loss_mode", loss_mode) != loss_mode:
            raise ValueError("Resume checkpoint loss_mode does not match this run")
        if resumed.get("scenario", "indoor") != scenario:
            raise ValueError("Resume checkpoint scenario does not match this run")
        if resumed.get("teacher_checkpoint") != (str(teacher_checkpoint) if teacher_checkpoint else None):
            raise ValueError("Resume checkpoint teacher configuration does not match this run")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        if scheduler is not None and resumed.get("scheduler") is not None:
            scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed["epoch"]) + 1
        best_linear = float(resumed["best_val_nmse_linear"])
        history = list(resumed["history"])
        print(f"Resuming at epoch {start_epoch} from {resume_path}")
    start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_squared_error = 0.0
        train_objective = 0.0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            mse_loss = (prediction - x).square().mean()
            supervised_loss = sample_nmse_linear(x, prediction).mean()
            loss = supervised_loss if loss_mode == "nmse" else mse_loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_output = teacher(x)
                    teacher_prediction = (
                        teacher_output[0]
                        if isinstance(teacher_output, (tuple, list))
                        else teacher_output
                    )
                teacher_distance = (
                    (prediction - teacher_prediction).square().sum(dim=(1, 2, 3))
                    / (x - 0.5).square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
                ).mean()
                loss = loss + distill_weight * teacher_distance
            loss.backward()
            optimizer.step()
            train_squared_error += mse_loss.detach().item() * x.size(0)
            train_objective += loss.detach().item() * x.size(0)
        train_mse = train_squared_error / len(train_data)
        train_objective_mean = train_objective / len(train_data)

        model.eval()
        ratios = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                ratios.append(sample_nmse_linear(x, model(x)).cpu())
        val_linear = torch.cat(ratios).mean().item()
        val_db = 10.0 * np.log10(max(val_linear, 1e-12))
        if scheduler is not None:
            scheduler.step(val_db)
        record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "train_objective": train_objective_mean,
            "val_nmse_linear": val_linear,
            "val_nmse_db": float(val_db),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        _write_history(history, output_dir)

        if val_linear < best_linear:
            best_linear = val_linear
            torch.save(
                {
                    "model_name": name,
                    "encoded_dim": encoded_dim,
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_nmse_linear": val_linear,
                    "val_nmse_db": float(val_db),
                    "train_mse": train_mse,
                    "loss_mode": loss_mode,
                    "scenario": scenario,
                    "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else None,
                    "distill_weight": distill_weight if teacher is not None else 0.0,
                    "seed": seed,
                    "batch_size": batch_size,
                    "learning_rate": LEARNING_RATES[name],
                    "dataset": f"COST2100 {scenario}, CR=1/{compression_ratio}",
                    "model_kwargs": model_kwargs,
                },
                output_dir / "best.pt",
            )
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            best_db = 10.0 * np.log10(max(best_linear, 1e-12))
            print(
                f"epoch={epoch}/{epochs} train_mse={train_mse:.6g} "
                f"val_nmse_db={val_db:.3f} best_db={best_db:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        if epoch % save_every == 0 or epoch == epochs:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            _save_periodic_checkpoint(
                checkpoint_path,
                name=name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_linear=best_linear,
                history=history,
                encoded_dim=encoded_dim,
                batch_size=batch_size,
                seed=seed,
                learning_rate=LEARNING_RATES[name],
                loss_mode=loss_mode,
                scenario=scenario,
                teacher_checkpoint=teacher_checkpoint,
                distill_weight=distill_weight if teacher is not None else 0.0,
            )
            print(f"Saved resumable checkpoint: {checkpoint_path}")

    elapsed = time.perf_counter() - start
    _write_history(history, output_dir)
    summary = {
        "model_name": name,
        "encoded_dim": encoded_dim,
        "compression_ratio": compression_ratio,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "run_tag": run_tag,
        "save_every": save_every,
        "learning_rate": LEARNING_RATES[name],
        "loss_mode": loss_mode,
        "scenario": scenario,
        "scheduler": "ReduceLROnPlateau" if scheduler is not None else "none",
        "scheduler_patience": scheduler_patience,
        "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else None,
        "distill_weight": distill_weight if teacher is not None else 0.0,
        "best_epoch": int(torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)["epoch"]),
        "best_val_nmse_db": float(10.0 * np.log10(max(best_linear, 1e-12))),
        "elapsed_seconds": elapsed,
        "seconds_per_epoch": elapsed / max(epochs - start_epoch + 1, 1),
        "test_split_used": False,
        "model_kwargs": model_kwargs,
        "use_scheduler": bool(use_scheduler),
        "group": group,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved best checkpoint and validation history to: {output_dir}")
    return output_dir / "best.pt"
