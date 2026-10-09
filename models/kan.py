"""
Simple KAN (Kolmogorov-Arnold Network) implementation.
Based on original KAN paper: https://arxiv.org/abs/2404.19756

This is a simplified version without base_weight and spline_scaler,
resulting in fewer parameters while maintaining the core KAN functionality.

Params per layer: in_features × out_features × grid_size
(vs Efficient-KAN: in_features × out_features × (2 + grid_size + spline_order))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    Simple KAN Linear layer with B-spline basis functions.
    
    Each edge in the network is a learnable univariate function (B-spline).
    No base_weight or spline_scaler - pure spline transformation.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        grid_size: Number of grid intervals for B-splines (default: 3)
        spline_order: Order of B-spline (default: 2, quadratic)
        grid_range: Range for grid points (default: [-1, 1])
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
        spline_order: int = 2,
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Number of spline coefficients per edge
        self.num_coeffs = grid_size + spline_order
        
        # Create grid for B-spline evaluation
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        
        # Only spline weights - the core of KAN
        # Shape: [out_features, in_features, num_coeffs]
        # Total params: out_features × in_features × num_coeffs
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.num_coeffs)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize spline weights with small values
        nn.init.xavier_uniform_(self.spline_weight, gain=0.1)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions for input x.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Tensor of shape (batch_size, in_features, num_coeffs)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid  # (in_features, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)
        
        # Initial basis (order 0)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # Recursive B-spline computation
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :-(k + 1)]
            left_den = grid[:, k:-1] - grid[:, :-(k + 1)]
            right_num = grid[:, k + 1:] - x
            right_den = grid[:, k + 1:] - grid[:, 1:-k]
            
            # Avoid division by zero
            left = torch.where(left_den != 0, left_num / left_den, torch.zeros_like(left_num))
            right = torch.where(right_den != 0, right_num / right_den, torch.zeros_like(right_num))
            
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)  # [batch, in_features]
        
        # Compute B-spline basis: [batch, in_features, num_coeffs]
        splines = self.b_splines(x)
        
        # Memory-efficient: reshape and use linear
        # splines: [batch, in_features * num_coeffs]
        # weight: [out_features, in_features * num_coeffs]
        batch_size = splines.size(0)
        splines_flat = splines.reshape(batch_size, -1)  # [batch, in * coeffs]
        weight_flat = self.spline_weight.reshape(self.out_features, -1)  # [out, in * coeffs]
        
        output = F.linear(splines_flat, weight_flat)  # [batch, out_features]
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, grid_size={self.grid_size}, spline_order={self.spline_order}'


class KAN(nn.Module):
    """
    Multi-layer KAN network.
    
    Args:
        layers_hidden: List of layer sizes, e.g., [in_features, hidden, out_features]
        grid_size: Number of grid intervals for B-splines
        spline_order: Order of B-spline
        grid_range: Range for grid points
    """
    def __init__(
        self,
        layers_hidden: list,
        grid_size: int = 3,
        spline_order: int = 2,
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
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
