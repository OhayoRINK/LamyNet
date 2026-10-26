"""Command-line entry point for the Lite-DPAS COST2100 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train_baseline import train_model


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--compression-ratio", type=int, choices=(4, 16, 32, 64), default=4)
    parser.add_argument(
        "--model",
        choices=("litedpas_v2", "litedpas_v4", "litedpas_v5", "csinet", "crnet", "clnet", "transnet"),
        default="litedpas_v5",
    )
    parser.add_argument("--scenario", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument(
        "--scheduler",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto = ReduceLROnPlateau for litedpas_v4/v5 only; on/off forces it for any model",
    )
    parser.add_argument("--group", type=str, default=None, help="label stored in summary.json")
    # LamyNet ablation options (litedpas_v5 only)
    parser.add_argument("--delay-kernel", type=int, default=None)
    parser.add_argument("--spectral-bands", type=int, default=None)
    parser.add_argument("--no-spectral-gate", action="store_true")
    parser.add_argument("--no-delay-path", action="store_true")
    parser.add_argument("--latent-grid", type=int, choices=(16, 8), default=None)
    parser.add_argument("--upsample", choices=("pixelshuffle", "bilinear"), default=None)
    parser.add_argument("--activation", choices=("silu", "leakyrelu", "relu"), default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--loss-mode", choices=("nmse", "mse"), default="nmse")
    parser.add_argument("--scheduler-patience", type=int, default=25)
    parser.add_argument("--teacher-repo", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--distill-weight", type=float, default=0.25)
    args = parser.parse_args()

    data_dir = args.data_dir or args.project_root / "Data"
    encoded_dim = 2048 // args.compression_ratio
    model_kwargs = {}
    if args.delay_kernel is not None:
        model_kwargs["delay_kernel"] = args.delay_kernel
    if args.spectral_bands is not None:
        model_kwargs["spectral_bands"] = args.spectral_bands
    if args.no_spectral_gate:
        model_kwargs["use_spectral_gate"] = False
    if args.no_delay_path:
        model_kwargs["use_delay_path"] = False
    if args.latent_grid is not None:
        model_kwargs["latent_grid"] = args.latent_grid
    if args.upsample is not None:
        model_kwargs["upsample"] = args.upsample
    if args.activation is not None:
        model_kwargs["activation"] = args.activation
    use_scheduler = None if args.scheduler == "auto" else args.scheduler == "on"
    checkpoint = train_model(
        args.model,
        args.project_root,
        data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        encoded_dim=encoded_dim,
        run_tag=args.run_tag,
        save_every=args.save_every,
        resume_from=args.resume_from,
        loss_mode=args.loss_mode,
        scheduler_patience=args.scheduler_patience,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_repo=args.teacher_repo,
        distill_weight=args.distill_weight,
        scenario=args.scenario,
        model_kwargs=model_kwargs,
        use_scheduler=use_scheduler,
        group=args.group,
    )
    print(f"Best validation checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
