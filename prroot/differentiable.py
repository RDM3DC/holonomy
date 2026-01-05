"""
PR-Root Differentiable Module v1.0.1

PyTorch-compatible differentiable phase-resolved arithmetic.
Enables gradient flow through PR-Root operations for neural network integration.

Requires: torch (optional dependency)
"""
from __future__ import annotations
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
    import torch.nn as nn
    from torch.autograd import Function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = None

from .config import PRConfig

__all__ = [
    "PRTensor",
    "DifferentiablePRSqrt",
    "DifferentiablePRMul",
    "DifferentiablePRDiv",
    "pr_sqrt_diff",
    "pr_mul_diff",
    "pr_div_diff",
    "pr_unwrap_diff",
    "soft_parity",
    "soft_sign_from_parity",
    "TORCH_AVAILABLE",
]


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for differentiable PR-Root. "
            "Install with: pip install torch"
        )


# =============================================================================
# Soft/Relaxed Discrete Operations
# =============================================================================

def soft_parity(w: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Soft approximation of parity b = w mod 2.
    
    Uses: b_soft = 0.5 * (1 - cos(π * w))
    
    This is continuous and differentiable, equals 0 at even integers
    and 1 at odd integers.
    
    Args:
        w: Winding number (can be continuous during training)
        temperature: Sharpness of the approximation (lower = sharper)
    
    Returns:
        Soft parity in [0, 1]
    """
    _require_torch()
    import math
    return 0.5 * (1.0 - torch.cos(math.pi * w / temperature))


def soft_sign_from_parity(b_soft: Tensor) -> Tensor:
    """
    Convert soft parity to soft sign: (-1)^b ≈ 1 - 2*b_soft
    
    Returns value in [-1, 1].
    """
    _require_torch()
    return 1.0 - 2.0 * b_soft


def soft_winding(theta: Tensor, theta_R: Tensor, two_pi: float) -> Tensor:
    """
    Soft winding number: w = (θ - θ_R) / 2π
    
    This is the continuous relaxation of round((θ - θ_R) / 2π).
    """
    _require_torch()
    return (theta - theta_R) / two_pi


# =============================================================================
# PRTensor: Differentiable Phase-Resolved State
# =============================================================================

@dataclass
class PRTensor:
    """
    Differentiable phase-resolved tensor state.
    
    Stores amplitude and lifted phase as PyTorch tensors with gradient tracking.
    Winding and parity are computed as soft (differentiable) approximations.
    
    Attributes:
        A: Amplitude tensor (batch_size,) or scalar
        theta: Lifted phase tensor (batch_size,) or scalar
        config: PR-Root configuration
    """
    A: Tensor
    theta: Tensor
    config: PRConfig
    
    def __post_init__(self):
        _require_torch()
        if not isinstance(self.A, Tensor):
            self.A = torch.tensor(self.A, dtype=torch.float32)
        if not isinstance(self.theta, Tensor):
            self.theta = torch.tensor(self.theta, dtype=torch.float32)
    
    @property
    def theta_R(self) -> Tensor:
        """Principal phase via differentiable modular reduction."""
        import math
        two_pi = 2.0 * self.config.pi_a
        # Use fmod for differentiability, then shift
        wrapped = torch.fmod(self.theta + self.config.pi_a, two_pi)
        # Handle negative values
        wrapped = torch.where(wrapped < 0, wrapped + two_pi, wrapped)
        return wrapped - self.config.pi_a
    
    @property
    def w_soft(self) -> Tensor:
        """Soft (differentiable) winding number."""
        return soft_winding(self.theta, self.theta_R, 2.0 * self.config.pi_a)
    
    @property
    def w(self) -> Tensor:
        """Hard winding number (integer, gradient via straight-through)."""
        w_continuous = self.w_soft
        # Straight-through estimator: forward uses round, backward uses identity
        return w_continuous + (torch.round(w_continuous) - w_continuous).detach()
    
    @property
    def b_soft(self) -> Tensor:
        """Soft (differentiable) parity in [0, 1]."""
        return soft_parity(self.w_soft)
    
    @property
    def b(self) -> Tensor:
        """Hard parity (0 or 1, gradient via straight-through)."""
        b_continuous = self.b_soft
        b_hard = torch.round(b_continuous)
        return b_continuous + (b_hard - b_continuous).detach()
    
    @property
    def z(self) -> Tensor:
        """Complex value as tensor (real, imag stacked or complex dtype)."""
        return self.A * torch.exp(1j * self.theta)
    
    @property
    def z_real(self) -> Tensor:
        """Real part of z."""
        return self.A * torch.cos(self.theta)
    
    @property
    def z_imag(self) -> Tensor:
        """Imaginary part of z."""
        return self.A * torch.sin(self.theta)
    
    @classmethod
    def from_complex_tensor(
        cls,
        z: Tensor,
        config: Optional[PRConfig] = None,
        theta_lift: Optional[Tensor] = None,
    ) -> "PRTensor":
        """
        Create PRTensor from complex tensor.
        
        Args:
            z: Complex tensor (or real tensor interpreted as real part)
            config: PR-Root configuration
            theta_lift: Optional explicit lift (if None, uses atan2)
        """
        _require_torch()
        cfg = config or PRConfig()
        
        # Handle both complex and real tensors
        if z.is_complex():
            A = torch.abs(z)
            if theta_lift is None:
                theta = torch.angle(z)
            else:
                theta = theta_lift
        else:
            # Assume real tensor, interpret as (real, imag) or just real
            A = torch.abs(z)
            if theta_lift is None:
                theta = torch.where(z >= 0, torch.zeros_like(z),
                                    torch.full_like(z, cfg.pi_a))
            else:
                theta = theta_lift
        
        return cls(A=A, theta=theta, config=cfg)
    
    @classmethod
    def from_polar(
        cls,
        A: Tensor,
        theta: Tensor,
        config: Optional[PRConfig] = None,
    ) -> "PRTensor":
        """Create PRTensor from amplitude and phase tensors."""
        _require_torch()
        return cls(A=A, theta=theta, config=config or PRConfig())
    
    def detach(self) -> "PRTensor":
        """Return a detached copy (no gradient tracking)."""
        return PRTensor(
            A=self.A.detach(),
            theta=self.theta.detach(),
            config=self.config,
        )
    
    def requires_grad_(self, requires_grad: bool = True) -> "PRTensor":
        """Set gradient tracking in-place."""
        self.A.requires_grad_(requires_grad)
        self.theta.requires_grad_(requires_grad)
        return self


# =============================================================================
# Custom Autograd Functions
# =============================================================================

if TORCH_AVAILABLE:
    class DifferentiablePRSqrt(Function):
        """
        Autograd function for phase-resolved square root.
        
        Forward: sqrt(A) * exp(i * θ/2)
        Backward: Standard chain rule through A and θ
        """
        
        @staticmethod
        def forward(ctx, A: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
            """
            Forward pass: compute √z in polar form.
            
            Returns:
                A_out: sqrt(A)
                theta_out: theta / 2
            """
            A_out = torch.sqrt(A)
            theta_out = theta / 2.0
            
            # Save for backward
            ctx.save_for_backward(A, theta)
            
            return A_out, theta_out
        
        @staticmethod
        def backward(ctx, grad_A_out: Tensor, grad_theta_out: Tensor) -> Tuple[Tensor, Tensor]:
            """
            Backward pass: gradients w.r.t. input A and θ.
            
            d(√A)/dA = 1/(2√A)
            d(θ/2)/dθ = 1/2
            """
            A, theta = ctx.saved_tensors
            
            # Gradient for amplitude: d(sqrt(A))/dA = 0.5 / sqrt(A)
            grad_A = grad_A_out * 0.5 / (torch.sqrt(A) + 1e-12)
            
            # Gradient for phase: d(theta/2)/dtheta = 0.5
            grad_theta = grad_theta_out * 0.5
            
            return grad_A, grad_theta

    class DifferentiablePRMul(Function):
        """
        Autograd function for phase-resolved multiplication.
        
        (A1, θ1) * (A2, θ2) = (A1*A2, θ1+θ2)
        """
        
        @staticmethod
        def forward(ctx, A1: Tensor, theta1: Tensor, A2: Tensor, theta2: Tensor) -> Tuple[Tensor, Tensor]:
            ctx.save_for_backward(A1, theta1, A2, theta2)
            return A1 * A2, theta1 + theta2
        
        @staticmethod
        def backward(ctx, grad_A_out: Tensor, grad_theta_out: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            A1, theta1, A2, theta2 = ctx.saved_tensors
            
            grad_A1 = grad_A_out * A2
            grad_A2 = grad_A_out * A1
            grad_theta1 = grad_theta_out
            grad_theta2 = grad_theta_out
            
            return grad_A1, grad_theta1, grad_A2, grad_theta2

    class DifferentiablePRDiv(Function):
        """
        Autograd function for phase-resolved division.
        
        (A1, θ1) / (A2, θ2) = (A1/A2, θ1-θ2)
        """
        
        @staticmethod
        def forward(ctx, A1: Tensor, theta1: Tensor, A2: Tensor, theta2: Tensor) -> Tuple[Tensor, Tensor]:
            ctx.save_for_backward(A1, theta1, A2, theta2)
            return A1 / (A2 + 1e-12), theta1 - theta2
        
        @staticmethod
        def backward(ctx, grad_A_out: Tensor, grad_theta_out: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            A1, theta1, A2, theta2 = ctx.saved_tensors
            
            grad_A1 = grad_A_out / (A2 + 1e-12)
            grad_A2 = -grad_A_out * A1 / (A2 ** 2 + 1e-12)
            grad_theta1 = grad_theta_out
            grad_theta2 = -grad_theta_out
            
            return grad_A1, grad_theta1, grad_A2, grad_theta2


# =============================================================================
# Functional API
# =============================================================================

def pr_sqrt_diff(x: PRTensor) -> Tuple[PRTensor, Tensor]:
    """
    Differentiable phase-resolved square root.
    
    Args:
        x: Input PRTensor
    
    Returns:
        (sqrt_result, soft_parity): Result PRTensor and soft parity value
    """
    _require_torch()
    A_out, theta_out = DifferentiablePRSqrt.apply(x.A, x.theta)
    result = PRTensor(A=A_out, theta=theta_out, config=x.config)
    return result, x.b_soft


def pr_mul_diff(a: PRTensor, b: PRTensor) -> PRTensor:
    """Differentiable phase-resolved multiplication."""
    _require_torch()
    if a.config != b.config:
        raise ValueError("Configs must match")
    A_out, theta_out = DifferentiablePRMul.apply(a.A, a.theta, b.A, b.theta)
    return PRTensor(A=A_out, theta=theta_out, config=a.config)


def pr_div_diff(a: PRTensor, b: PRTensor) -> PRTensor:
    """Differentiable phase-resolved division."""
    _require_torch()
    if a.config != b.config:
        raise ValueError("Configs must match")
    A_out, theta_out = DifferentiablePRDiv.apply(a.A, a.theta, b.A, b.theta)
    return PRTensor(A=A_out, theta=theta_out, config=a.config)


def pr_unwrap_diff(
    z_sequence: Tensor,
    config: Optional[PRConfig] = None,
) -> PRTensor:
    """
    Differentiable phase unwrapping for a sequence of complex values.
    
    This is a fully differentiable version of pr_unwrap_path that maintains
    gradient flow through the entire sequence.
    
    Args:
        z_sequence: Complex tensor of shape (seq_len,) or (batch, seq_len)
        config: PR-Root configuration
    
    Returns:
        PRTensor with unwrapped phase
    """
    _require_torch()
    import math
    
    cfg = config or PRConfig()
    
    # Get amplitude and raw phase
    A = torch.abs(z_sequence)
    theta_raw = torch.angle(z_sequence)
    
    # Differentiable unwrapping via cumulative sum of wrapped differences
    diff = theta_raw[..., 1:] - theta_raw[..., :-1]
    
    # Wrap differences to [-π, π] (differentiable)
    diff_wrapped = diff - 2 * math.pi * torch.round(diff / (2 * math.pi))
    
    # Cumulative sum to get unwrapped phase
    theta_cumsum = torch.cumsum(diff_wrapped, dim=-1)
    
    # Prepend initial phase
    theta_unwrapped = torch.cat([
        theta_raw[..., :1],
        theta_raw[..., :1] + theta_cumsum
    ], dim=-1)
    
    return PRTensor(A=A, theta=theta_unwrapped, config=cfg)


# =============================================================================
# Neural Network Modules
# =============================================================================

if TORCH_AVAILABLE:
    class PRSqrtLayer(nn.Module):
        """
        Neural network layer that applies phase-resolved square root.
        """
        
        def __init__(self, config: Optional[PRConfig] = None):
            super().__init__()
            self.config = config or PRConfig()
        
        def forward(self, x: PRTensor) -> Tuple[PRTensor, Tensor]:
            return pr_sqrt_diff(x)

    class PRUnwrapLayer(nn.Module):
        """
        Neural network layer that unwraps phase from complex input.
        """
        
        def __init__(self, config: Optional[PRConfig] = None):
            super().__init__()
            self.config = config or PRConfig()
        
        def forward(self, z: Tensor) -> PRTensor:
            return pr_unwrap_diff(z, config=self.config)

    class HolonomyEmbedding(nn.Module):
        """
        Embedding layer that encodes phase state (A, θ, w_soft, b_soft).
        
        Maps PRTensor to a dense embedding that captures both the value
        and the topological state.
        
        Args:
            embed_dim: Output embedding dimension
            config: PR-Root configuration
        """
        
        def __init__(self, embed_dim: int, config: Optional[PRConfig] = None):
            super().__init__()
            self.config = config or PRConfig()
            self.embed_dim = embed_dim
            
            # Linear projection from (A, θ_R, w_soft, b_soft) -> embed_dim
            self.proj = nn.Linear(4, embed_dim)
        
        def forward(self, x: PRTensor) -> Tensor:
            """
            Args:
                x: PRTensor (can be batched)
            
            Returns:
                Embedding tensor of shape (..., embed_dim)
            """
            features = torch.stack([
                x.A,
                x.theta_R,
                x.w_soft,
                x.b_soft,
            ], dim=-1)
            
            return self.proj(features)

    # Export nn.Module classes
    __all__.extend(["PRSqrtLayer", "PRUnwrapLayer", "HolonomyEmbedding"])
