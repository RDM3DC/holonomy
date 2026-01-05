"""
Tests for differentiable PR-Root operations.
"""
import math
import pytest

# Skip all tests if torch not available
torch = pytest.importorskip("torch")

from prroot import PRConfig
from prroot.differentiable import (
    PRTensor,
    pr_sqrt_diff,
    pr_mul_diff,
    pr_div_diff,
    pr_unwrap_diff,
    soft_parity,
    soft_sign_from_parity,
    DifferentiablePRSqrt,
    HolonomyEmbedding,
)


class TestSoftOperations:
    """Test soft/relaxed discrete operations."""

    def test_soft_parity_at_integers(self):
        """soft_parity should be ~0 at even integers, ~1 at odd integers."""
        w_even = torch.tensor([0.0, 2.0, 4.0, -2.0])
        w_odd = torch.tensor([1.0, 3.0, -1.0, -3.0])

        b_even = soft_parity(w_even)
        b_odd = soft_parity(w_odd)

        assert torch.allclose(b_even, torch.zeros_like(b_even), atol=1e-6)
        assert torch.allclose(b_odd, torch.ones_like(b_odd), atol=1e-6)

    def test_soft_parity_is_differentiable(self):
        """soft_parity should have non-zero gradients."""
        w = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
        b = soft_parity(w)
        loss = b.sum()
        loss.backward()

        assert w.grad is not None
        assert not torch.allclose(w.grad, torch.zeros_like(w.grad))

    def test_soft_sign_from_parity(self):
        """soft_sign should map 0->1, 1->-1."""
        b = torch.tensor([0.0, 1.0, 0.5])
        sign = soft_sign_from_parity(b)

        expected = torch.tensor([1.0, -1.0, 0.0])
        assert torch.allclose(sign, expected)


class TestPRTensor:
    """Test PRTensor differentiable state."""

    def test_from_complex_tensor(self):
        """Create PRTensor from complex tensor."""
        z = torch.tensor([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
        cfg = PRConfig()

        prt = PRTensor.from_complex_tensor(z, config=cfg)

        assert torch.allclose(prt.A, torch.ones(4))
        expected_theta = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2])
        assert torch.allclose(prt.theta, expected_theta, atol=1e-6)

    def test_from_polar(self):
        """Create PRTensor from polar coordinates."""
        A = torch.tensor([1.0, 2.0, 0.5])
        theta = torch.tensor([0.0, math.pi, -math.pi / 2])

        prt = PRTensor.from_polar(A, theta)

        assert torch.equal(prt.A, A)
        assert torch.equal(prt.theta, theta)

    def test_theta_R_is_principal(self):
        """theta_R should be in (-π, π]."""
        theta = torch.tensor([0.0, 2 * math.pi, 4 * math.pi, -2 * math.pi, 3 * math.pi])
        A = torch.ones_like(theta)

        prt = PRTensor.from_polar(A, theta)
        theta_R = prt.theta_R

        # All should be in (-π, π]
        assert (theta_R > -math.pi - 1e-6).all()
        assert (theta_R <= math.pi + 1e-6).all()

    def test_w_soft_continuous(self):
        """w_soft should be continuous approximation of winding."""
        theta = torch.tensor([0.0, 2 * math.pi, 4 * math.pi, -2 * math.pi])
        A = torch.ones_like(theta)

        prt = PRTensor.from_polar(A, theta)

        # w_soft should be close to integer values
        expected_w = torch.tensor([0.0, 1.0, 2.0, -1.0])
        assert torch.allclose(prt.w_soft, expected_w, atol=0.1)

    def test_z_roundtrip(self):
        """z property should reconstruct complex value."""
        z_orig = torch.tensor([1 + 2j, -1 + 0j, 0.5 - 0.5j])

        prt = PRTensor.from_complex_tensor(z_orig)
        z_reconstructed = prt.z

        assert torch.allclose(z_reconstructed, z_orig, atol=1e-6)

    def test_requires_grad(self):
        """PRTensor should support gradient tracking."""
        A = torch.tensor([1.0, 2.0], requires_grad=True)
        theta = torch.tensor([0.0, math.pi], requires_grad=True)

        prt = PRTensor.from_polar(A, theta)

        assert prt.A.requires_grad
        assert prt.theta.requires_grad


class TestDifferentiableSqrt:
    """Test differentiable square root."""

    def test_sqrt_values(self):
        """pr_sqrt_diff should compute correct values."""
        A = torch.tensor([4.0, 9.0, 1.0])
        theta = torch.tensor([0.0, math.pi, math.pi / 2])

        prt = PRTensor.from_polar(A, theta)
        result, parity = pr_sqrt_diff(prt)

        # Check amplitudes: sqrt([4, 9, 1]) = [2, 3, 1]
        assert torch.allclose(result.A, torch.tensor([2.0, 3.0, 1.0]))

        # Check phases: [0, π, π/2] / 2 = [0, π/2, π/4]
        assert torch.allclose(result.theta, torch.tensor([0.0, math.pi / 2, math.pi / 4]))

    def test_sqrt_gradient_flow(self):
        """Gradients should flow through pr_sqrt_diff."""
        A = torch.tensor([4.0], requires_grad=True)
        theta = torch.tensor([math.pi], requires_grad=True)

        prt = PRTensor.from_polar(A, theta)
        result, _ = pr_sqrt_diff(prt)

        # Compute loss and backprop
        loss = result.A.sum() + result.theta.sum()
        loss.backward()

        assert A.grad is not None
        assert theta.grad is not None

        # d(sqrt(4))/dA = 0.5/sqrt(4) = 0.25
        assert torch.allclose(A.grad, torch.tensor([0.25]), atol=1e-6)

        # d(θ/2)/dθ = 0.5
        assert torch.allclose(theta.grad, torch.tensor([0.5]), atol=1e-6)

    def test_sqrt_parity_tracking(self):
        """Parity should reflect winding."""
        # w=0 -> b=0
        prt0 = PRTensor.from_polar(torch.tensor([1.0]), torch.tensor([0.0]))
        _, parity0 = pr_sqrt_diff(prt0)
        assert parity0.item() < 0.1

        # w=1 (θ = 2π + small) -> b≈1
        prt1 = PRTensor.from_polar(
            torch.tensor([1.0]), torch.tensor([2 * math.pi + 0.01])
        )
        _, parity1 = pr_sqrt_diff(prt1)
        assert parity1.item() > 0.9


class TestDifferentiableMulDiv:
    """Test differentiable multiplication and division."""

    def test_mul_values(self):
        """pr_mul_diff should compute (A1*A2, θ1+θ2)."""
        a = PRTensor.from_polar(torch.tensor([2.0]), torch.tensor([math.pi / 4]))
        b = PRTensor.from_polar(torch.tensor([3.0]), torch.tensor([math.pi / 4]))

        result = pr_mul_diff(a, b)

        assert torch.allclose(result.A, torch.tensor([6.0]))
        assert torch.allclose(result.theta, torch.tensor([math.pi / 2]))

    def test_div_values(self):
        """pr_div_diff should compute (A1/A2, θ1-θ2)."""
        a = PRTensor.from_polar(torch.tensor([6.0]), torch.tensor([math.pi]))
        b = PRTensor.from_polar(torch.tensor([2.0]), torch.tensor([math.pi / 2]))

        result = pr_div_diff(a, b)

        assert torch.allclose(result.A, torch.tensor([3.0]))
        assert torch.allclose(result.theta, torch.tensor([math.pi / 2]))

    def test_mul_gradient_flow(self):
        """Gradients should flow through multiplication."""
        A1 = torch.tensor([2.0], requires_grad=True)
        theta1 = torch.tensor([0.0], requires_grad=True)
        A2 = torch.tensor([3.0], requires_grad=True)
        theta2 = torch.tensor([math.pi / 2], requires_grad=True)

        a = PRTensor.from_polar(A1, theta1)
        b = PRTensor.from_polar(A2, theta2)
        result = pr_mul_diff(a, b)

        loss = result.A.sum()
        loss.backward()

        # d(A1*A2)/dA1 = A2 = 3
        assert torch.allclose(A1.grad, torch.tensor([3.0]))
        # d(A1*A2)/dA2 = A1 = 2
        assert torch.allclose(A2.grad, torch.tensor([2.0]))


class TestDifferentiableUnwrap:
    """Test differentiable phase unwrapping."""

    def test_unwrap_circular_path(self):
        """Unwrapping circular path should accumulate phase."""
        n = 100
        t = torch.linspace(0, 2 * math.pi, n + 1)
        z = torch.exp(1j * t)

        result = pr_unwrap_diff(z)

        # Final phase should be close to 2π
        assert result.theta[-1].item() > 6.0  # > 2π - tolerance
        assert result.theta[-1].item() < 6.5  # < 2π + tolerance

    def test_unwrap_is_differentiable(self):
        """Phase unwrapping should allow gradient flow."""
        n = 50
        t = torch.linspace(0, 2 * math.pi, n + 1, requires_grad=True)
        z = torch.exp(1j * t)

        result = pr_unwrap_diff(z)
        loss = result.theta[-1]
        loss.backward()

        assert t.grad is not None


class TestHolonomyEmbedding:
    """Test neural network embedding layer."""

    def test_embedding_shape(self):
        """HolonomyEmbedding should produce correct output shape."""
        embed_dim = 64
        layer = HolonomyEmbedding(embed_dim)

        # Single sample
        prt = PRTensor.from_polar(torch.tensor([1.0]), torch.tensor([0.0]))
        out = layer(prt)
        assert out.shape == (1, embed_dim)

    def test_embedding_gradient_flow(self):
        """Gradients should flow through embedding."""
        embed_dim = 32
        layer = HolonomyEmbedding(embed_dim)

        A = torch.tensor([1.0, 2.0], requires_grad=True)
        theta = torch.tensor([0.0, math.pi], requires_grad=True)
        prt = PRTensor.from_polar(A, theta)

        out = layer(prt)
        loss = out.sum()
        loss.backward()

        assert A.grad is not None
        assert theta.grad is not None

    def test_embedding_differentiates_parity(self):
        """Embedding should encode different parities differently."""
        embed_dim = 16
        layer = HolonomyEmbedding(embed_dim)

        # Same z value, different history (w=0 vs w=1)
        prt0 = PRTensor.from_polar(torch.tensor([1.0]), torch.tensor([0.0]))
        prt1 = PRTensor.from_polar(torch.tensor([1.0]), torch.tensor([2 * math.pi]))

        emb0 = layer(prt0)
        emb1 = layer(prt1)

        # Embeddings should be different (due to different w_soft)
        assert not torch.allclose(emb0, emb1, atol=0.1)
