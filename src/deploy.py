"""Deployment-form conversion of LamyNet / CsiNet: BN folding, DFT-matmul spectral gate, ONNX export.

Why: the BandwiseSpectralGate uses ``torch.fft.rfft``/``irfft``, which the ONNX
exporter does not support. For the fixed widths used here (W = 32 or 16) the
real FFT is a small fixed linear map, so it is replaced by matmuls with
constant DFT matrices (numerically identical, max error ~1e-8). Because
irfft(rfft(x)) = x, the gate's residual blend x + a * (irfft(S * G) - x) equals
irfft(S * G') with G' = (1 - a) + a * G, so the blend is folded into the gate
gains (one small op on the [B, C, bands] gate tensor instead of three on the
feature map). BatchNorm layers are folded into the preceding convolutions,
including the BN that follows PixelShuffle (channel c takes the scale of conv
channels c*r^2 .. c*r^2 + r^2 - 1). The resulting module graph exports to ONNX
and lets runtimes such as ONNX Runtime fuse the many small operators.

Usage::

    python -m src.deploy --checkpoint runs/<run>/best.pt --out source/onnx/<name>
    # writes <name>_encoder.onnx and <name>_decoder.onnx and checks equivalence
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from src.models.lite_dpas import BandwiseSpectralGate


class DFTSpectralGate(nn.Module):
    """Inference-only BandwiseSpectralGate with rFFT/irFFT expressed as constant matmuls."""

    def __init__(self, gate: BandwiseSpectralGate, width: int) -> None:
        super().__init__()
        self.gate = gate.gate
        self.bands, self.channels = gate.bands, gate.channels
        self.register_buffer("scale", gate.residual_scale.detach().clone())
        n, bins = width, width // 2 + 1
        k, t = torch.arange(bins).float(), torch.arange(n).float()
        ang = 2 * math.pi * torch.outer(t, k) / n  # [n, bins]
        self.register_buffer("C", torch.cos(ang) / math.sqrt(n))  # Re{rfft(x)} = x @ C   (norm="ortho")
        self.register_buffer("S", -torch.sin(ang) / math.sqrt(n))  # Im{rfft(x)} = x @ S
        w = torch.full((bins,), 2.0)
        w[0] = 1.0
        if n % 2 == 0:
            w[-1] = 1.0
        self.register_buffer("Ci", (w[:, None] * torch.cos(ang.T)) / math.sqrt(n))  # irfft = Re @ Ci + Im @ Si
        self.register_buffer("Si", (-w[:, None] * torch.sin(ang.T)) / math.sqrt(n))
        # adaptive_avg_pool1d(bins -> bands) uses overlapping windows [floor(i*bins/bands), ceil((i+1)*bins/bands))
        pool = torch.zeros(bins, self.bands)
        for b in range(self.bands):
            start, end = (b * bins) // self.bands, -((-(b + 1) * bins) // self.bands)
            pool[start:end, b] = 1.0 / (end - start)
        self.register_buffer("pool", pool)
        band_ids = torch.div(torch.arange(bins) * self.bands, bins, rounding_mode="floor").clamp_max(self.bands - 1)
        self.register_buffer("expand", F.one_hot(band_ids, self.bands).float().T)  # per-bin gate = gates @ expand

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        re, im = x @ self.C, x @ self.S
        magnitude = torch.sqrt(re * re + im * im).mean(dim=2)  # [B, C, bins]
        band_energy = magnitude @ self.pool
        gates = self.gate(band_energy.flatten(start_dim=1)).view(x.shape[0], self.channels, self.bands)
        # residual blend folded into the gains: x + a * (irfft(S*G) - x) == irfft(S * ((1 - a) + a * G))
        per_bin = (((1.0 - self.scale) + self.scale * gates) @ self.expand).unsqueeze(2)
        return (re * per_bin) @ self.Ci + (im * per_bin) @ self.Si


def _replace_gates(model: nn.Module, sample: torch.Tensor) -> None:
    widths: dict[int, int] = {}
    hooks = [
        g.register_forward_hook(lambda mod, inp, out, g=g: widths.__setitem__(id(g), inp[0].shape[-1]))
        for g in model.modules()
        if isinstance(g, BandwiseSpectralGate)
    ]
    with torch.no_grad():
        model(sample)
    for h in hooks:
        h.remove()

    def rec(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, BandwiseSpectralGate):
                setattr(module, name, DFTSpectralGate(child, widths[id(child)]))
            else:
                rec(child)

    rec(model)


def _fuse_conv_pixelshuffle_bn(conv: nn.Conv2d, shuffle: nn.PixelShuffle, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold BN(PixelShuffle(conv(x))) into the conv: output channel c of the shuffle comes from conv channels c*r^2 + i."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    shift = bn.bias - bn.running_mean * scale
    r2 = shuffle.upscale_factor ** 2
    scale, shift = scale.repeat_interleave(r2), shift.repeat_interleave(r2)
    fused = copy.deepcopy(conv)
    bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
    fused.weight = nn.Parameter(conv.weight * scale[:, None, None, None])
    fused.bias = nn.Parameter(bias * scale + shift)
    return fused


def _fold_batchnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            mods, out, i = list(child.children()), [], 0
            while i < len(mods):
                if i + 1 < len(mods) and isinstance(mods[i], nn.Conv2d) and isinstance(mods[i + 1], nn.BatchNorm2d):
                    out.append(fuse_conv_bn_eval(mods[i], mods[i + 1]))
                    i += 2
                elif (i + 2 < len(mods) and isinstance(mods[i], nn.Conv2d) and isinstance(mods[i + 1], nn.PixelShuffle)
                      and isinstance(mods[i + 2], nn.BatchNorm2d)):
                    out += [_fuse_conv_pixelshuffle_bn(mods[i], mods[i + 1], mods[i + 2]), mods[i + 1]]
                    i += 3
                else:
                    _fold_batchnorm(mods[i])
                    out.append(mods[i])
                    i += 1
            setattr(module, name, nn.Sequential(*out))
        else:
            _fold_batchnorm(child)


def to_deploy(model: nn.Module, input_shape: tuple[int, ...] = (1, 2, 32, 32)) -> nn.Module:
    """Return an eval-mode deep copy with BN folded and spectral gates in DFT-matmul form."""
    deploy = copy.deepcopy(model).eval().cpu()
    _replace_gates(deploy, torch.zeros(*input_shape))
    _fold_batchnorm(deploy)
    for p in deploy.parameters():
        p.requires_grad_(False)
    return deploy


class _Encoder(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.encode(x)


class _Decoder(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.m.decode(z)


def export_onnx(deploy: nn.Module, out_prefix: Path, opset: int = 17) -> tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 2, 32, 32)
    with torch.no_grad():
        z = deploy.encode(x)
    enc_path = out_prefix.with_name(out_prefix.name + "_encoder.onnx")
    dec_path = out_prefix.with_name(out_prefix.name + "_decoder.onnx")
    torch.onnx.export(
        _Encoder(deploy), (x,), str(enc_path), input_names=["csi"], output_names=["codeword"], opset_version=opset, dynamo=False
    )
    torch.onnx.export(
        _Decoder(deploy), (z,), str(dec_path), input_names=["codeword"], output_names=["csi"], opset_version=opset, dynamo=False
    )
    return enc_path, dec_path


def max_abs_diff(model: nn.Module, deploy: nn.Module, n: int = 16) -> float:
    torch.manual_seed(0)
    x = torch.rand(n, 2, 32, 32)
    with torch.no_grad():
        return float((model.eval().cpu()(x) - deploy(x)).abs().max())


def main() -> None:
    from src.train_baseline import build_model

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output prefix, e.g. source/onnx/lamynet_in_cr16")
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(
        ckpt["model_name"], root, encoded_dim=int(ckpt["encoded_dim"]), model_kwargs=ckpt.get("model_kwargs") or {}
    )
    model.load_state_dict(ckpt["model"])
    deploy = to_deploy(model)
    print(f"max |original - deploy| over 16 random inputs: {max_abs_diff(model, deploy):.2e}")
    enc, dec = export_onnx(deploy, args.out)
    print(f"wrote {enc} and {dec}")


if __name__ == "__main__":
    main()
