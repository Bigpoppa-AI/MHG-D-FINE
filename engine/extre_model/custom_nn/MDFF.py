import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def frequency_mirror(x):
    h, w = x.shape[-2:]

    index_h = (-torch.arange(h, device=x.device)) % h
    index_w = (-torch.arange(w, device=x.device)) % w

    return x.index_select(-2, index_h).index_select(-1, index_w)


class LowFreqSemantic(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()

        if reduction <= 0:
            raise ValueError("reduction must be greater than 0.")

        hidden = max(dim // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
        )

    def forward(self, band_fft):
        magnitude = band_fft.abs()
        attention = torch.sigmoid(self.mlp(self.pool(magnitude)))

        return band_fft * attention


class MidFreqDirectional(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "The mid-frequency strip convolution kernel size "
                "must be a positive odd integer."
            )

        padding = kernel_size // 2

        self.dw_h = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=dim,
            bias=False,
        )
        self.dw_v = nn.Conv2d(
            dim,
            dim,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=dim,
            bias=False,
        )
        self.fuse = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=False,
        )

    def forward(self, band_fft):
        magnitude = torch.fft.fftshift(
            band_fft.abs(),
            dim=(-2, -1),
        )

        directional_response = (
            self.dw_h(magnitude) + self.dw_v(magnitude)
        )
        attention = torch.sigmoid(self.fuse(directional_response))

        attention = torch.fft.ifftshift(
            attention,
            dim=(-2, -1),
        )

        attention = 0.5 * (
            attention + frequency_mirror(attention)
        )

        return band_fft * attention


class HighFreqSparse(nn.Module):
    def __init__(self, dim, threshold_init=1e-3, eps=1e-6):
        super().__init__()

        if threshold_init <= 0:
            raise ValueError("threshold_init must be greater than 0.")
        if eps <= 0:
            raise ValueError("eps must be greater than 0.")

        self.eps = eps

        raw_init = (
            threshold_init
            + math.log(-math.expm1(-threshold_init))
        )

        self.raw_threshold = nn.Parameter(
            torch.full((1, dim, 1, 1), raw_init)
        )

        self.gain_logits = nn.Parameter(
            torch.zeros(1, dim, 1, 1)
        )

    def forward(self, band_fft):
        magnitude = band_fft.abs()

        threshold = F.softplus(self.raw_threshold)

        soft_mask = (
            F.relu(magnitude - threshold)
            / magnitude.clamp_min(self.eps)
        )

        gain = 2.0 * torch.sigmoid(self.gain_logits)

        return band_fft * soft_mask * gain


class SpatialMultiConv(nn.Module):
    def __init__(self, dim, large_kernel=19):
        super().__init__()

        if dim < 3:
            raise ValueError(
                "The spatial branch must have at least 3 channels."
            )
        if large_kernel <= 0 or large_kernel % 2 == 0:
            raise ValueError(
                "large_kernel must be a positive odd integer."
            )

        branch_channels = dim // 3
        bypass_channels = dim - 2 * branch_channels

        self.split_sizes = (
            bypass_channels,
            branch_channels,
            branch_channels,
        )

        self.local_convs = nn.ModuleList([
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=kernel,
                padding=padding,
                groups=branch_channels,
                bias=False,
            )
            for kernel, padding in [
                ((1, 3), (0, 1)),
                ((3, 1), (1, 0)),
                ((3, 3), (1, 1)),
            ]
        ])

        p = large_kernel // 2

        self.large_convs = nn.ModuleList([
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=kernel,
                padding=padding,
                groups=branch_channels,
                bias=False,
            )
            for kernel, padding in [
                ((1, large_kernel), (0, p)),
                ((large_kernel, 1), (p, 0)),
                ((large_kernel, large_kernel), (p, p)),
            ]
        ])

    def forward(self, x):
        x_bypass, x_local, x_large = torch.split(
            x,
            self.split_sizes,
            dim=1,
        )

        local = F.relu(
            sum(conv(x_local) for conv in self.local_convs)
        )
        large = F.relu(
            sum(conv(x_large) for conv in self.large_convs)
        )

        return torch.cat(
            [x_bypass, local, large],
            dim=1,
        )


class MDFF(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        reduction=8,
        large_kernel=19,
        mid_kernel=5,
        threshold_init=1e-3,
        band_sigma=0.18,
        branch_scale_init=0.1,
    ):
        super().__init__()

        if c1 <= 0:
            raise ValueError("c1 must be greater than 0.")
        if c2 < 3:
            raise ValueError("c2 must be at least 3.")
        if band_sigma <= 0:
            raise ValueError("band_sigma must be greater than 0.")

        self.c1 = c1
        self.c2 = c2
        self.band_sigma = band_sigma

        self.input_proj = ConvBNAct(
            c1, c2, kernel_size=1, act=True
        )

        self.spatial_branch = SpatialMultiConv(
            c2,
            large_kernel=large_kernel,
        )

        self.low_freq = LowFreqSemantic(
            c2,
            reduction=reduction,
        )
        self.mid_freq = MidFreqDirectional(
            c2,
            kernel_size=mid_kernel,
        )
        self.high_freq = HighFreqSparse(
            c2,
            threshold_init=threshold_init,
        )

        self.band_logits = nn.Parameter(torch.zeros(3))

        self.freq_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.lambda_spatial = nn.Parameter(
            torch.tensor(float(branch_scale_init))
        )
        self.lambda_frequency = nn.Parameter(
            torch.tensor(float(branch_scale_init))
        )

        self.output_proj = ConvBNAct(
            c2, c2, kernel_size=1, act=False
        )

    def _create_band_masks(self, h, w, device):
        freq_h = torch.fft.fftfreq(
            h,
            device=device,
            dtype=torch.float32,
        ) * h

        freq_w = torch.fft.fftfreq(
            w,
            device=device,
            dtype=torch.float32,
        ) * w

        yy, xx = torch.meshgrid(
            freq_h,
            freq_w,
            indexing="ij",
        )

        radius = torch.sqrt(yy.square() + xx.square())
        radius_max = radius.max().clamp_min(1.0)
        normalized_radius = radius / radius_max

        centers = torch.tensor(
            [0.0, 0.5, 1.0],
            device=device,
            dtype=torch.float32,
        ).view(3, 1, 1)

        masks = torch.exp(
            -0.5 * (
                (normalized_radius.unsqueeze(0) - centers)
                / self.band_sigma
            ).square()
        )

        masks = masks / masks.sum(
            dim=0,
            keepdim=True,
        ).clamp_min(1e-12)

        return masks[:, None, None, :, :]

    def _frequency_forward(self, x):
        original_dtype = x.dtype

        if x.device.type in ("cuda", "cpu"):
            amp_context = torch.autocast(
                device_type=x.device.type,
                enabled=False,
            )
        else:
            amp_context = nullcontext()

        with amp_context:
            x_float = x.float()
            h, w = x_float.shape[-2:]

            spectrum = torch.fft.fft2(
                x_float,
                dim=(-2, -1),
                norm="ortho",
            )

            masks = self._create_band_masks(
                h, w, x.device
            )

            spectrum_low = spectrum * masks[0]
            spectrum_mid = spectrum * masks[1]
            spectrum_high = spectrum * masks[2]

            enhanced_low = self.low_freq(spectrum_low)
            enhanced_mid = self.mid_freq(spectrum_mid)
            enhanced_high = self.high_freq(spectrum_high)

            weights = torch.softmax(
                self.band_logits.float(),
                dim=0,
            )

            enhanced_spectrum = (
                weights[0] * enhanced_low
                + weights[1] * enhanced_mid
                + weights[2] * enhanced_high
            )

            reconstructed = torch.fft.ifft2(
                enhanced_spectrum,
                dim=(-2, -1),
                norm="ortho",
            ).real

            channel_weight = self.freq_channel_attention(
                reconstructed.abs()
            )

            frequency_output = reconstructed * channel_weight

        return frequency_output.to(dtype=original_dtype)

    def forward(self, x):
        # Support [D2, P3] input.
        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError(
                    "The multi-input format for MDFF must be [D2, P3]."
                )

            d2, p3 = x

            if d2.ndim != 4 or p3.ndim != 4:
                raise ValueError(
                    "Both D2 and P3 must be four-dimensional BCHW tensors."
                )
            if d2.shape[0] != p3.shape[0]:
                raise ValueError(
                    "D2 and P3 must have the same batch size."
                )
            if d2.shape[-2:] != p3.shape[-2:]:
                raise ValueError(
                    "D2 and P3 must have the same spatial dimensions; "
                    "downsample P2 using P2-DPD first."
                )
            if d2.device != p3.device or d2.dtype != p3.dtype:
                raise ValueError(
                    "D2 and P3 must have the same device and dtype."
                )

            x = torch.cat([d2, p3], dim=1)

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "MDFF input must be a tensor or a list/tuple "
                "containing two tensors."
            )
        if x.ndim != 4:
            raise ValueError(
                "MDFF input must be a four-dimensional BCHW tensor."
            )
        if x.shape[1] != self.c1:
            raise ValueError(
                f"MDFF expects {self.c1} total input channels, "
                f"but received {x.shape[1]}."
            )

        mapped = self.input_proj(x)

        # Process spatial and frequency features in parallel.
        spatial_output = self.spatial_branch(mapped)
        frequency_output = self._frequency_forward(mapped)

        fused = F.relu(
            mapped
            + self.lambda_spatial * spatial_output
            + self.lambda_frequency * frequency_output
        )

        return self.output_proj(fused)


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MDFF(
        c1=256,
        c2=128,
        large_kernel=19,
        mid_kernel=5,
        threshold_init=1e-3,
    ).to(device)

    d2 = torch.randn(
        2, 128, 31, 45,
        device=device,
        requires_grad=True,
    )
    p3 = torch.randn(
        2, 128, 31, 45,
        device=device,
        requires_grad=True,
    )

    output = model([d2, p3])

    assert output.shape == (2, 128, 31, 45)
    assert torch.isfinite(output).all()

    loss = output.square().mean()
    loss.backward()

    threshold_grad = model.high_freq.raw_threshold.grad

    assert threshold_grad is not None
    assert torch.isfinite(threshold_grad).all()

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("Device:", device)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", parameter_count)
    print(
        "Threshold gradient mean:",
        threshold_grad.abs().mean().item(),
    )
    print(
        "Band weights:",
        torch.softmax(
            model.band_logits.detach(), dim=0
        ).cpu().tolist(),
    )
