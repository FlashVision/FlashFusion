"""FlashFusion neural network building blocks."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """Standard convolution block: Conv2d + BatchNorm + Activation.

    A reusable building block that packages a convolution layer with optional
    batch normalization and activation function.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding. If None, auto-computed for 'same' output.
        groups: Number of convolution groups.
        bias: Whether to include bias (disabled when using BatchNorm).
        activation: Activation type ('relu', 'silu', 'leaky_relu', None).
        use_bn: Whether to include batch normalization.

    Example:
        >>> conv = ConvModule(64, 128, kernel_size=3, stride=1)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = conv(x)  # shape: (1, 128, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: int = 1,
        bias: bool = False,
        activation: Optional[str] = "silu",
        use_bn: bool = True,
    ):
        super().__init__()

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = self._build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    @staticmethod
    def _build_activation(activation: Optional[str]) -> nn.Module:
        if activation is None:
            return nn.Identity()
        activations = {
            "relu": nn.ReLU(inplace=True),
            "silu": nn.SiLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
            "hardswish": nn.Hardswish(inplace=True),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}. Available: {list(activations.keys())}")
        return activations[activation]


class DepthwiseConvModule(nn.Module):
    """Depthwise separable convolution: Depthwise Conv + Pointwise Conv.

    Efficient convolution block that factorizes a standard convolution into
    a depthwise spatial convolution followed by a pointwise (1x1) convolution.
    Reduces parameters and FLOPs significantly.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the depthwise convolution.
        stride: Stride for the depthwise convolution.
        padding: Padding for the depthwise convolution. Auto if None.
        activation: Activation type ('relu', 'silu', 'leaky_relu', None).
        use_bn: Whether to include batch normalization.

    Example:
        >>> dw_conv = DepthwiseConvModule(64, 128, kernel_size=3)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = dw_conv(x)  # shape: (1, 128, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        activation: Optional[str] = "silu",
        use_bn: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)

        self.depthwise = ConvModule(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels,
            activation=activation, use_bn=use_bn,
        )

        self.pointwise = ConvModule(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            groups=1, activation=activation, use_bn=use_bn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


__all__ = ["ConvModule", "DepthwiseConvModule"]
