"""Compact dual-path CSI autoencoder with a band-wise spectral gate.

The input/output convention matches the project's CsiNet port:
float32 [B, 2, 32, 32], real/imaginary channels normalized to [0, 1].
``encoded_dim`` counts real-valued latent coefficients, as in the baseline.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualAxisLiteBlock(nn.Module):
    """Depthwise spatial/delay paths followed by pointwise feature fusion."""

    def __init__(
        self, channels: int, delay_kernel: int = 5, use_delay_path: bool = True
    ) -> None:
        super().__init__()
        if delay_kernel % 2 != 1:
            raise ValueError("delay_kernel must be odd")
        self.use_delay_path = use_delay_path
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        if use_delay_path:
            self.delay = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, delay_kernel),
                    padding=(0, delay_kernel // 2),
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            fuse_in = 2 * channels
        else:
            # Ablation: spatial-only block (single 3x3 depthwise path).
            self.delay = None
            fuse_in = channels
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.delay is None:
            features = self.spatial(x)
        else:
            features = torch.cat((self.spatial(x), self.delay(x)), dim=1)
        return self.activation(x + self.fuse(features))


class BandwiseSpectralGate(nn.Module):
    """Apply learned, input-conditioned gains to grouped delay-axis FFT bins."""

    def __init__(self, channels: int, bands: int = 4, reduction: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.bands = bands
        hidden = max(4, channels * bands // reduction)
        self.gate = nn.Sequential(
            nn.Linear(channels * bands, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels * bands),
            nn.Sigmoid(),
        )
        # Start close to an identity map so early training is stable.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected [B, {self.channels}, H, W], got {tuple(x.shape)}")
        original_dtype = x.dtype
        # Keep FFT operations in float32 for reliable autocast/CUDA support.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            spectrum = torch.fft.rfft(x32, dim=-1, norm="ortho")
            magnitude = spectrum.abs().mean(dim=2)  # [B, C, W//2+1]
            band_energy = F.adaptive_avg_pool1d(magnitude, self.bands)
            gates = self.gate(band_energy.flatten(start_dim=1)).view(
                x.shape[0], self.channels, self.bands
            )

            bin_count = spectrum.shape[-1]
            band_ids = torch.div(
                torch.arange(bin_count, device=x.device) * self.bands,
                bin_count,
                rounding_mode="floor",
            ).clamp_max(self.bands - 1)
            per_bin_gate = gates.index_select(dim=-1, index=band_ids).unsqueeze(2)
            refined = torch.fft.irfft(
                spectrum * per_bin_gate,
                n=x.shape[-1],
                dim=-1,
                norm="ortho",
            )
            result = x32 + self.residual_scale * (refined - x32)
        return result.to(dtype=original_dtype)


class LiteDPASCsiNet(nn.Module):
    """Lite-DPAS test model for COST2100 32x32 CSI and selectable latent size."""

    INPUT_SHAPE = (2, 32, 32)
    FEATURE_CHANNELS = 8
    LATENT_GRID = (8, 8)

    def __init__(
        self,
        encoded_dim: int = 512,
        delay_kernel: int = 5,
        spectral_bands: int = 4,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        if not 1 <= encoded_dim <= 2048:
            raise ValueError("encoded_dim must be in [1, 2048]")
        if blocks < 1:
            raise ValueError("blocks must be at least 1")
        channels = self.FEATURE_CHANNELS
        self.encoded_dim = encoded_dim
        self.encoder = nn.Sequential(
            ConvNormAct(2, channels),
            *(DualAxisLiteBlock(channels, delay_kernel) for _ in range(blocks)),
            nn.AdaptiveAvgPool2d(self.LATENT_GRID),
        )
        latent_features = channels * self.LATENT_GRID[0] * self.LATENT_GRID[1]
        self.to_codeword = nn.Linear(latent_features, encoded_dim)
        self.from_codeword = nn.Linear(encoded_dim, latent_features)
        self.decoder_low = DualAxisLiteBlock(channels, delay_kernel)
        self.upsample = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1),
            DualAxisLiteBlock(channels, delay_kernel),
        )
        self.spectral_gate = BandwiseSpectralGate(channels, bands=spectral_bands)
        self.output = nn.Sequential(nn.Conv2d(channels, 2, 3, padding=1), nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != self.INPUT_SHAPE:
            raise ValueError(f"Expected [B, 2, 32, 32], got {tuple(x.shape)}")
        features = self.encoder(x)
        return self.to_codeword(features.flatten(start_dim=1))

    def decode(self, codeword: torch.Tensor) -> torch.Tensor:
        if codeword.ndim != 2 or codeword.shape[1] != self.encoded_dim:
            raise ValueError(
                f"Expected [B, {self.encoded_dim}] codeword, got {tuple(codeword.shape)}"
            )
        batch = codeword.shape[0]
        channels = self.FEATURE_CHANNELS
        low = self.from_codeword(codeword).reshape(batch, channels, *self.LATENT_GRID)
        low = self.decoder_low(low)
        high = F.interpolate(low, size=(32, 32), mode="bilinear", align_corners=False)
        high = self.upsample(high)
        high = self.spectral_gate(high)
        return self.output(high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def count_parameters(model: nn.Module) -> int:
    """Count all trainable and non-trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


class LearnedDownsample(nn.Module):
    """Learned stride-2 analysis transform followed by dual-axis processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        delay_kernel: int,
        use_delay_path: bool = True,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.refine = DualAxisLiteBlock(out_channels, delay_kernel, use_delay_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class LearnedPixelShuffleUp(nn.Module):
    """Learned 2x synthesis transform; avoids fixed bilinear reconstruction."""

    def __init__(
        self, channels: int, delay_kernel: int, use_delay_path: bool = True
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.refine = DualAxisLiteBlock(channels, delay_kernel, use_delay_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


class BilinearUp(nn.Module):
    """Ablation alternative to LearnedPixelShuffleUp: fixed bilinear 2x upsampling."""

    def __init__(
        self, channels: int, delay_kernel: int, use_delay_path: bool = True
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.refine = DualAxisLiteBlock(channels, delay_kernel, use_delay_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.refine(self.up(high))


class LiteDPASCsiNetV3(nn.Module):
    """DPAS V3: wider low-resolution latent, earlier spectral selection.

    Requires the existing helpers in lite_dpas.py:
    ConvNormAct, DualAxisLiteBlock, LearnedDownsample,
    LearnedPixelShuffleUp, BandwiseSpectralGate.
    """

    INPUT_SHAPE = (2, 32, 32)
    LATENT_GRID = (8, 8)
    LATENT_CHANNELS = 20
    DECODER_CHANNELS = 16

    def __init__(
        self,
        encoded_dim: int = 512,
        delay_kernel: int = 5,
        spectral_bands: int = 4,
    ) -> None:
        super().__init__()

        if not 1 <= encoded_dim <= 2048:
            raise ValueError("encoded_dim must be in [1, 2048]")

        latent_channels = self.LATENT_CHANNELS
        decoder_channels = self.DECODER_CHANNELS
        self.encoded_dim = encoded_dim

        # Analysis path: keep early, high-resolution channel widths small.
        # Frequency selection happens at 8x8 before the feedback projection.
        self.encoder_features = nn.Sequential(
            ConvNormAct(2, 8),
            DualAxisLiteBlock(8, delay_kernel),
            LearnedDownsample(8, 12, delay_kernel),  # 32x32 -> 16x16
            LearnedDownsample(12, latent_channels, delay_kernel),  # 16x16 -> 8x8
            BandwiseSpectralGate(latent_channels, bands=spectral_bands),
            DualAxisLiteBlock(latent_channels, delay_kernel),
        )

        feature_dim = latent_channels * self.LATENT_GRID[0] * self.LATENT_GRID[1]
        self.to_codeword = nn.Linear(feature_dim, encoded_dim)

        # UE-to-BS feedback remains the same size for a given compression ratio.
        self.from_codeword = nn.Linear(encoded_dim, feature_dim)
        self.decoder_low = DualAxisLiteBlock(latent_channels, delay_kernel)

        # Spend the extra latent width at 8x8, then reduce channels before
        # higher-resolution synthesis to control MACs.
        self.to_decoder_channels = ConvNormAct(
            latent_channels, decoder_channels, kernel_size=1
        )

        self.decoder_up1 = LearnedPixelShuffleUp(
            decoder_channels, delay_kernel
        )  # 8x8 -> 16x16
        self.decoder_spectral_gate = BandwiseSpectralGate(
            decoder_channels, bands=spectral_bands
        )  # spectral refinement at 16x16, not 32x32
        self.decoder_up2 = LearnedPixelShuffleUp(
            decoder_channels, delay_kernel
        )  # 16x16 -> 32x32

        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels, 2, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != self.INPUT_SHAPE:
            raise ValueError(f"Expected [B, 2, 32, 32], got {tuple(x.shape)}")

        features = self.encoder_features(x)
        return self.to_codeword(features.flatten(start_dim=1))

    def decode(self, codeword: torch.Tensor) -> torch.Tensor:
        if codeword.ndim != 2 or codeword.shape[1] != self.encoded_dim:
            raise ValueError(
                f"Expected [B, {self.encoded_dim}] codeword, "
                f"got {tuple(codeword.shape)}"
            )

        batch = codeword.shape[0]
        low = self.from_codeword(codeword).reshape(
            batch,
            self.LATENT_CHANNELS,
            self.LATENT_GRID[0],
            self.LATENT_GRID[1],
        )

        low = self.decoder_low(low)
        high = self.to_decoder_channels(low)
        high = self.decoder_up1(high)
        high = self.decoder_spectral_gate(high)
        high = self.decoder_up2(high)

        return self.output(high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# V4 adds a low-cost spectral selection stage at 16x16, before the second
# learned downsample. This protects delay-domain cues before the 8x8 bottleneck.
class LiteDPASCsiNetV4(LiteDPASCsiNetV3):
    """V3 plus encoder-side spectral refinement at the 16x16 feature scale."""

    def __init__(
        self,
        encoded_dim: int = 512,
        delay_kernel: int = 5,
        spectral_bands: int = 4,
    ) -> None:
        super().__init__(
            encoded_dim=encoded_dim,
            delay_kernel=delay_kernel,
            spectral_bands=spectral_bands,
        )
        self.encoder_features = nn.Sequential(
            ConvNormAct(2, 8),
            DualAxisLiteBlock(8, delay_kernel),
            LearnedDownsample(8, 12, delay_kernel),  # 32x32 -> 16x16
            BandwiseSpectralGate(12, bands=spectral_bands),
            LearnedDownsample(12, self.LATENT_CHANNELS, delay_kernel),  # -> 8x8
            BandwiseSpectralGate(self.LATENT_CHANNELS, bands=spectral_bands),
            DualAxisLiteBlock(self.LATENT_CHANNELS, delay_kernel),
        )


def _replace_silu(module: nn.Module, activation: str) -> None:
    """Swap every nn.SiLU below ``module`` for LeakyReLU(0.3) (slope used by CsiNet/CRNet) or ReLU."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(0.3, inplace=True) if activation == "leakyrelu" else nn.ReLU(inplace=True))
        else:
            _replace_silu(child, activation)


class LiteDPASCsiNetV5(nn.Module):
    """Spatially preserved, long-delay Lite-DPAS without teacher distillation.

    V5 preserves 16x16 spatial resolution, processes it with 8 channels, then
    projects to a 4-channel latent before feedback. It uses depthwise long-delay
    operators, while the real-valued feedback dimension remains ``encoded_dim``.
    """

    INPUT_SHAPE = (2, 32, 32)
    LATENT_GRID = (16, 16)
    PROCESS_CHANNELS = 8
    LATENT_CHANNELS = 4

    def __init__(
        self,
        encoded_dim: int = 512,
        delay_kernel: int = 15,
        spectral_bands: int = 4,
        use_spectral_gate: bool = True,
        use_delay_path: bool = True,
        latent_grid: int = 16,
        upsample: str = "pixelshuffle",
        activation: str = "silu",
    ) -> None:
        """Default arguments reproduce the paper model (LamyNet).

        The remaining keyword arguments exist for ablation studies only:
        ``use_spectral_gate=False`` removes both spectral gates,
        ``use_delay_path=False`` reduces every dual-axis block to its 3x3 path,
        ``latent_grid=8`` adds a second stride-2 stage so the latent is
        16 x 8 x 8 (same 1,024-dim FC input), and ``upsample="bilinear"``
        replaces the learned PixelShuffle synthesis with bilinear interpolation.
        """
        super().__init__()
        if not 1 <= encoded_dim <= 2048:
            raise ValueError("encoded_dim must be in [1, 2048]")
        if delay_kernel % 2 != 1:
            raise ValueError("delay_kernel must be odd")
        if latent_grid not in (16, 8):
            raise ValueError("latent_grid must be 16 or 8")
        if upsample not in ("pixelshuffle", "bilinear"):
            raise ValueError("upsample must be 'pixelshuffle' or 'bilinear'")

        self.encoded_dim = encoded_dim
        self.config = {
            "delay_kernel": delay_kernel,
            "spectral_bands": spectral_bands,
            "use_spectral_gate": use_spectral_gate,
            "use_delay_path": use_delay_path,
            "latent_grid": latent_grid,
            "upsample": upsample,
            "activation": activation,
        }
        if activation not in ("silu", "leakyrelu", "relu"):
            raise ValueError("activation must be 'silu', 'leakyrelu' or 'relu'")
        process_channels = self.PROCESS_CHANNELS
        # Keep the FC input at 1,024 real values for both latent grids.
        latent_channels = self.LATENT_CHANNELS if latent_grid == 16 else 16
        self.latent_grid = (latent_grid, latent_grid)
        self.latent_channels = latent_channels
        dp = use_delay_path

        def gate(channels: int) -> nn.Module:
            if use_spectral_gate:
                return BandwiseSpectralGate(channels, bands=spectral_bands)
            return nn.Identity()

        def up(channels: int) -> nn.Module:
            if upsample == "pixelshuffle":
                return LearnedPixelShuffleUp(channels, delay_kernel, dp)
            return BilinearUp(channels, delay_kernel, dp)

        encoder = [
            ConvNormAct(2, process_channels),
            DualAxisLiteBlock(process_channels, delay_kernel, dp),
            LearnedDownsample(process_channels, process_channels, delay_kernel, dp),
            gate(process_channels),
            DualAxisLiteBlock(process_channels, delay_kernel, dp),
        ]
        if latent_grid == 8:
            encoder.append(
                LearnedDownsample(process_channels, process_channels, delay_kernel, dp)
            )
        encoder.append(ConvNormAct(process_channels, latent_channels, kernel_size=1))
        self.encoder_features = nn.Sequential(*encoder)

        feature_dim = latent_channels * latent_grid * latent_grid
        self.to_codeword = nn.Linear(feature_dim, encoded_dim)
        self.from_codeword = nn.Linear(encoded_dim, feature_dim)
        self.decoder_low = DualAxisLiteBlock(latent_channels, delay_kernel, dp)
        self.to_decoder_channels = ConvNormAct(
            latent_channels, process_channels, kernel_size=1
        )
        self.decoder_spectral_gate = gate(process_channels)
        if latent_grid == 8:
            self.decoder_up = nn.Sequential(up(process_channels), up(process_channels))
        else:
            self.decoder_up = up(process_channels)
        self.output = nn.Sequential(
            nn.Conv2d(process_channels, 2, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        if activation != "silu":
            # Latency study: ONNX Runtime fuses ReLU / LeakyReLU into the preceding convolution but not SiLU.
            # Every SiLU of the network (blocks, resampling stages and the gate MLP) is replaced.
            _replace_silu(self, activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != self.INPUT_SHAPE:
            raise ValueError(f"Expected [B, 2, 32, 32], got {tuple(x.shape)}")
        features = self.encoder_features(x)
        return self.to_codeword(features.flatten(start_dim=1))

    def decode(self, codeword: torch.Tensor) -> torch.Tensor:
        if codeword.ndim != 2 or codeword.shape[1] != self.encoded_dim:
            raise ValueError(
                f"Expected [B, {self.encoded_dim}] codeword, got {tuple(codeword.shape)}"
            )
        batch = codeword.shape[0]
        low = self.from_codeword(codeword).reshape(
            batch, self.latent_channels, *self.latent_grid
        )
        low = self.to_decoder_channels(self.decoder_low(low))
        low = self.decoder_spectral_gate(low)
        high = self.decoder_up(low)
        return self.output(high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# Preserve old import names. V4 is selected explicitly by the new experiment.
LiteDPASCsiNetV2 = LiteDPASCsiNetV3
