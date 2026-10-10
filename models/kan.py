"""
KAN (Kolmogorov-Arnold Network) implementation.
Adapted from U-KAN: https://github.com/CUHK-AIM-Group/U-KAN

This is the Efficient-KAN version with:
- base_weight: Linear residual path (like ResNet)
- spline_weight: B-spline learnable functions
- spline_scaler: Per-edge scaling for spline output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    Efficient KAN Linear layer with B-spline basis functions.
    
    Combines a base linear transformation with learnable spline functions.
    output = base_activation(x) @ base_weight + splines(x) @ (spline_weight * spline_scaler)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        grid_size: Number of grid intervals for B-splines (default: 5)
        spline_order: Order of B-spline (default: 3, cubic)
        scale_noise: Noise scale for spline initialization
        scale_base: Scale for base weight initialization
        scale_spline: Scale for spline weight initialization
        enable_standalone_scale_spline: Whether to use separate spline scaler
        base_activation: Activation for base path (default: SiLU)
        grid_eps: Interpolation factor for adaptive grid update
        grid_range: Range for grid points (default: [-1, 1])
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create grid for B-spline evaluation
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Base weight (linear residual path)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Spline weight
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        
        # Optional spline scaler
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions for input x.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Tensor of shape (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid  # (in_features, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)
        
        # Initial basis (order 0)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # Recursive B-spline computation
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)])
                / (grid[:, k:-1] - grid[:, :-(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute spline coefficients that interpolate the given points.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            y: Output tensor of shape (batch_size, in_features, out_features)
            
        Returns:
            Coefficients tensor of shape (out_features, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Base path: activation(x) @ base_weight
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Spline path: splines(x) @ scaled_spline_weight
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Update grid based on input data distribution (adaptive grid).
        Call this periodically during training for better performance.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # (batch, in, out)

        # Sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0):
        """
        Compute regularization loss for sparsity.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'grid_size={self.grid_size}, spline_order={self.spline_order}'
        )


class KAN(nn.Module):
    """
    Multi-layer KAN network.
    
    Args:
        layers_hidden: List of layer sizes, e.g., [in_features, hidden, out_features]
        grid_size: Number of grid intervals for B-splines
        spline_order: Order of B-spline
        scale_noise: Noise scale for spline initialization
        scale_base: Scale for base weight initialization
        scale_spline: Scale for spline weight initialization
        base_activation: Activation for base path
        grid_eps: Interpolation factor for adaptive grid update
        grid_range: Range for grid points
    """
    def __init__(
        self,
        layers_hidden: list,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: type = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid: bool = False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
