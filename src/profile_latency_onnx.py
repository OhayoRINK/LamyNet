"""Batch-1 latency of LamyNet and the baselines (CsiNet, CRNet, CLNet, TransNet) after ONNX Runtime conversion.

LamyNet is converted with ``src.deploy.to_deploy`` (BN folding, DFT-matmul gate) because its
rFFT does not export; the other models are exported directly from the authors' PyTorch code
(ONNX Runtime fuses Conv+BN itself). Encoder (UE side) and decoder (BS side) are exported and
timed separately on the CPU execution provider with one intra-op thread and full graph
optimisation. Weights are random (latency is weight-independent).

Baseline code is loaded from ``--baselines`` (default: ``$LAMYNET_BASELINES`` or ``<repo>/baselines``).

Usage::

    python -m src.profile_latency_onnx [--out source/latency_onnx_results.json] [--repeats 500]
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from src.deploy import export_onnx, max_abs_diff, to_deploy
from src.models.lite_dpas import count_parameters
from src.train_baseline import build_model
from src.upstream_models import _load_model_file, baselines_dir


# ---- encoder / decoder wrappers for the authors' models ----
class _CRNetEnc(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        n = x.shape[0]
        out = torch.cat((self.m.encoder1(x), self.m.encoder2(x)), dim=1)
        out = self.m.encoder_conv(out)
        return self.m.encoder_fc(out.reshape(n, -1))

class _CRNetDec(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, z):
        out = self.m.decoder_fc(z).reshape(-1, 2, 32, 32)
        return self.m.sigmoid(self.m.decoder_feature(out))

class _TransNetEnc(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        n = x.shape[0]
        memory = self.m.encoder(x.reshape(n, self.m.feature_shape[0], self.m.feature_shape[1]))
        return self.m.fc_encoder(memory.reshape(n, -1))

class _TransNetDec(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, z):
        md = self.m.fc_decoder(z).reshape(-1, self.m.feature_shape[0], self.m.feature_shape[1])
        return self.m.decoder(md, md).reshape(-1, 2, 32, 32)

class _Gen(nn.Module):
    def __init__(self, f): super().__init__(); self.f = f
    def forward(self, x): return self.f(x)


def encoder_decoder(name: str, model: nn.Module):
    """Return (encoder_module, decoder_module) for a model."""
    if name in ("LamyNet", "LamyNet-LReLU", "CsiNet"):
        return _Gen(model.encode), _Gen(model.decode)
    if name == "CRNet":
        return _CRNetEnc(model), _CRNetDec(model)
    if name == "CLNet":
        return model.encoder, model.decoder
    if name == "TransNet":
        return _TransNetEnc(model), _TransNetDec(model)
    raise ValueError(name)


def build(name: str, cr: int, root: Path, baselines: Path) -> nn.Module:
    if name == "LamyNet":
        return to_deploy(build_model("litedpas_v5", root, encoded_dim=2048 // cr))
    if name == "LamyNet-LReLU":  # activation study: every SiLU replaced by LeakyReLU(0.3)
        return to_deploy(build_model("litedpas_v5", root, encoded_dim=2048 // cr, model_kwargs={"activation": "leakyrelu"}))
    if name == "CsiNet":
        return build_model("csinet", root, encoded_dim=2048 // cr).eval()
    if name == "CRNet":
        return _load_model_file("up_crnet", baselines / "CRNet/models/crnet.py").crnet(reduction=cr).eval()
    if name == "CLNet":
        return _load_model_file("up_clnet", baselines / "CLNet/models/clnet.py").clnet(reduction=cr).eval()
    if name == "TransNet":
        return _load_model_file("up_transnet", baselines / "TransNet/models/TransNet.py").transnet(reduction=cr, d_model=64).eval()
    raise ValueError(name)


def time_session(sess: ort.InferenceSession, feed: dict, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t) * 1000.0)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "source" / "latency_onnx_results.json")
    parser.add_argument("--baselines", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=["LamyNet", "CsiNet", "CRNet", "CLNet", "TransNet"])
    parser.add_argument("--crs", nargs="+", type=int, default=[4, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    if args.baselines is None and any(m in ("CRNet", "CLNet", "TransNet") for m in args.models):
        args.baselines = baselines_dir(root)

    torch.set_num_threads(1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    report = {
        "cpu": platform.processor() or platform.machine(),
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
        "provider": "CPUExecutionProvider",
        "intra_op_threads": 1,
        "batch_size": 1,
        "note": "LamyNet: BN folded + DFT-matmul gate (src.deploy). Baselines exported from the authors' code. ONNX opset 17, ORT_ENABLE_ALL.",
        "results": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.models:
            for cr in args.crs:
                model = build(name, cr, root, args.baselines)
                for p in model.parameters():
                    p.requires_grad_(False)
                enc_m, dec_m = encoder_decoder(name, model)
                enc_m.eval(); dec_m.eval()  # wrappers must stay in eval mode (BN running stats) after export
                x = torch.rand(1, 2, 32, 32)
                with torch.no_grad():
                    z = enc_m(x)
                enc_path, dec_path = Path(tmp) / f"{name}_cr{cr}_enc.onnx", Path(tmp) / f"{name}_cr{cr}_dec.onnx"
                try:
                    torch.onnx.export(enc_m, (x,), str(enc_path), input_names=["csi"], output_names=["codeword"], opset_version=17, dynamo=False)
                    torch.onnx.export(dec_m, (z,), str(dec_path), input_names=["codeword"], output_names=["csi"], opset_version=17, dynamo=False)
                except Exception as e:  # noqa: BLE001
                    print(f"{name} CR 1/{cr}: ONNX export failed: {type(e).__name__}: {str(e)[:120]}")
                    continue
                enc = ort.InferenceSession(str(enc_path), so, providers=["CPUExecutionProvider"])
                dec = ort.InferenceSession(str(dec_path), so, providers=["CPUExecutionProvider"])
                xn = x.numpy()
                zn = enc.run(None, {"csi": xn})[0]
                enc_m.eval(); dec_m.eval()
                with torch.no_grad():
                    ref = dec_m(enc_m(x)).numpy()
                err = float(np.abs(dec.run(None, {"codeword": zn})[0] - ref).max())
                entry = {
                    "model": name,
                    "compression_ratio": cr,
                    "parameters": count_parameters(model),
                    "encoder_parameters": sum(p.numel() for p in enc_m.parameters()),
                    "onnx_vs_torch_max_abs_error": err,
                    "onnx_nodes": {"encoder": len(onnx.load(str(enc_path)).graph.node), "decoder": len(onnx.load(str(dec_path)).graph.node)},
                    "encoder": time_session(enc, {"csi": xn}, args.warmup, args.repeats),
                    "decoder": time_session(dec, {"codeword": zn}, args.warmup, args.repeats),
                }
                report["results"].append(entry)
                print(
                    f"{name:8s} CR 1/{cr:<3d} params {entry['parameters']:>10,}  enc {entry['encoder']['median_ms']:.3f} ms  "
                    f"dec {entry['decoder']['median_ms']:.3f} ms  err {err:.1e}  nodes {entry['onnx_nodes']}"
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
