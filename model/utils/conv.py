import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from typing import Optional

def fft_conv2d(
    x: torch.Tensor,    # [B, C, H, W]
    k: torch.Tensor,    # [C, H, W] or [C, Hk, Wk], shape depends on use case
    dropout_mask: Optional[torch.Tensor] = None,
    gelu: bool = True,
    residual: bool = True,
):
    """
    2D FFT-based convolution on x (depth/channel-wise or matching-k-channels).
    Tweak residual / activation (gelu) as needed.

    Args:
        x: [B, C, H, W]
        k: [C, Hk, Wk]; for "full-size" convolution Hk=H, Wk=W (or pad manually outside).
        dropout_mask: shape [B, C], channel-wise dropout (optional).
        gelu: whether to apply gelu activation (optional).
        residual: whether to add the x residual (optional).

    Returns:
        out: [B, C, H, W], same shape as x.
    """

    B, C, H, W = x.shape
    # assume k.shape = [C, Hk, Wk]
    Hk, Wk = k.shape[-2], k.shape[-1]

    # pick FFT size that covers at least H+Hk-1, W+Wk-1
    # simple choice: 2x; tune for efficiency if needed
    fft_height = 2 * max(H, Hk)
    fft_width = 2 * max(W, Wk)

    # zero-pad kernel k first
    # depthwise case: k shape is [C, something, something]
    # this impl assumes per-channel / depthwise-separable
    k_f = torch.fft.rfft2(k.to(x.dtype), s=(fft_height, fft_width))  # [C, fft_height, fft_width//2+1]
    # note: the (fft_height * fft_width) divisor can be moved to after the multiply with x_f
    # (equivalent); pre-dividing is also fine — depends on style

    # FFT input x
    x_f = torch.fft.rfft2(x, s=(fft_height, fft_width))  # [B, C, fft_height, fft_width//2+1]

    # per-channel pointwise multiply
    # x_f: [B, C, Hf, Wf], k_f: [C, Hf, Wf] -> broadcasting
    y_f = x_f * k_f.unsqueeze(0)
    # inverse transform
    # irfft2 output size is s=(fft_height, fft_width); crop back to original
    y = torch.fft.irfft2(
        y_f,
        s=(fft_height, fft_width),
        norm="forward"  # or "backward" depending on convention
    )
    # crop to original size
    # for full-conv / same-conv: crop differently as needed
    y = y[..., :H, :W]  # [B, C, H, W]

    # optional: add residual + activation
    if residual:
        out = y + x
    else:
        out = y
    if gelu:
        out = F.gelu(out)

    # optional channel-wise dropout_mask
    # dropout_mask: [B, C], broadcast to [B, C, 1, 1]
    if dropout_mask is not None:
        out = out * dropout_mask.unsqueeze(-1).unsqueeze(-1)

    return out


class ShortConvolution2D(nn.Conv2d):
    """
    Short convolution for images (depthwise-separable), nn.Conv2d-style.
    Default groups = hidden_size, i.e. per-channel convolution.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        bias: bool = False,
        activation: Optional[str] = 'silu',
    ):
        # padding strategy is tweakable per need
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,     # depthwise-separable: groups = hidden_size
            bias=bias,
            padding=kernel_size // 2  # 'same' convolution
        )

        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            if activation not in ['silu', 'swish', 'relu', 'gelu']:
                raise ValueError(f"Activation `{activation}` not supported yet.")
            self.activation = activation

    def forward(
        self,
        x: torch.Tensor,               # [B, C, H, W]
        mask: Optional[torch.Tensor] = None,  # optional mask
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            mask: [B, H, W] or [B, 1, H, W], indicating valid positions.
                  broadcast to [B, C, H, W] as needed.

        Returns:
            out: [B, C, H, W]
        """
        if mask is not None:
            # mask must be compatible with [B, 1, H, W] or [B, C, H, W]
            x = x * mask

        # standard 2D conv
        out = self._conv_forward(x, self.weight, self.bias)

        # activation
        if self.activation == 'silu' or self.activation == 'swish':
            out = F.silu(out)
        elif self.activation == 'relu':
            out = F.relu(out)
        elif self.activation == 'gelu':
            out = F.gelu(out)

        return out

class LongConvolution2D(nn.Module):
    """
    "Long convolution" on images:
    - directly learn a [C, H, W] or [C, H_max, W_max] filter (depends on max resolution)
    - forward uses 2D FFT for the convolution
    """

    def __init__(
        self,
        hidden_size: int,
        max_h: int,
        max_w: int,
    ):
        """
        Args:
            hidden_size: channel count C.
            max_h, max_w: maximum H/W (assume training images don't exceed).
        """
        super().__init__()
        # filter shape is [C, max_h, max_w]
        # for more flexibility (per-channel filter sizes), extend as needed
        self.hidden_size = hidden_size
        self.filter = nn.Parameter(
            torch.randn(self.hidden_size, max_h, max_w),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            y: [B, C, H, W]
        """
        # 2D FFT convolution
        y = fft_conv2d(x, self.filter, dropout_mask=None, gelu=False, residual=True)
        return y

import math

class PositionalEmbedding2D(nn.Module):
    """
    Example 2D positional embedding. Simplest: trig-expand (x/H, y/W).
    The below is just a demo; design freely for your needs.
    """

    def __init__(self, emb_dim: int, max_h: int, max_w: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_h = max_h
        self.max_w = max_w

        # pre-store a 2D grid: [H, W, 2] -> (row, col)
        # then expand with frequencies
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, max_h),
            torch.linspace(0, 1, max_w),
            indexing='ij'
        )
        # grid: [max_h, max_w, 2]
        grid = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]
        # base (x, y) only; add more frequencies (sin(2pi k x), cos(...)) as needed
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, h: int, w: int):
        """
        Return positional encoding for the first h, w cells. Shape: [h, w, ...]
        """
        # simplest: return self.grid[:h, :w] as a bare "positional encoding"
        # add linear/nonlinear mapping here if you want
        return self.grid[:h, :w]


class ImplicitLongConvolution2D(nn.Module):
    """
    Use an MLP to implicitly generate a 2D convolution kernel, then FFT-convolve.
    """

    def __init__(
        self,
        hidden_size: int,   # channel count C
        max_h: int,
        max_w: int,
        d_emb: int = 4,     # positional-embed dim (demo)
        d_hidden: int = 16, # MLP hidden size
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_emb = d_emb

        # 2D positional embedding
        self.pos_emb = PositionalEmbedding2D(d_emb, max_h, max_w)

        # simple MLP: (d_emb) -> (d_hidden) -> (C)
        # we want the MLP to map every grid pixel independently, then assemble
        # into a [C, h, w] filter
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_emb, d_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(d_hidden, hidden_size),
        )

    def generate_filter(self, h: int, w: int):
        """
        Dynamically generate a [C, h, w] filter for input size h, w.
        """
        # positional encoding: [h, w, d_emb]
        pe = self.pos_emb(h, w)  # [h, w, 2] (or more)
        # reshape: [h*w, d_emb]
        pe_flat = pe.view(-1, self.d_emb)

        # through MLP: [h*w, hidden_size]
        out_flat = self.mlp(pe_flat)  # [h*w, C]

        # reshape back to [C, h, w]
        k = out_flat.transpose(0, 1).reshape(self.hidden_size, h, w)
        return k

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # generate filter of matching size: [C, H, W]
        k = self.generate_filter(H, W)

        # FFT convolution
        y = fft_conv2d(
            x, k,
            dropout_mask=None,
            gelu=False,
            residual=True
        )
        return y
