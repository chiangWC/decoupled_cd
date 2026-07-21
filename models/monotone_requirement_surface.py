from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


SURFACE_MODES = (
    "full",
    "pooled_direct",
    "capacity_additive",
    "standard_ncd",
)
_PROBABILITY_CLIP = 1e-6
_INITIAL_TAIL_LOGIT = -8.0


def _initialize_hll_gates(raw_gates: torch.Tensor) -> None:
    """Initialize an HLL near the constant 0.5 monotone surface."""
    with torch.no_grad():
        raw_gates.fill_(_INITIAL_TAIL_LOGIT)
        if raw_gates.ndim == 1:
            raw_gates[0] = 0.0
        elif raw_gates.ndim == 2:
            raw_gates[0, 0] = 0.0
        else:  # pragma: no cover
            raise ValueError("HLL gates must be one- or two-dimensional.")


def _seeded_normal_(
    parameter: torch.Tensor,
    *,
    seed: int,
    mean: float = -4.0,
    std: float = 0.5,
) -> None:
    """Initialize without consuming or depending on the global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(
        parameter.shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).mul(std).add(mean)
    with torch.no_grad():
        parameter.copy_(values.to(parameter))


class HierarchicalLattice1D(nn.Module):
    """Minimal all-increasing one-dimensional HLL."""

    def __init__(self, lattice_size: int) -> None:
        super().__init__()
        if lattice_size < 2:
            raise ValueError("lattice_size must be at least 2.")
        self.lattice_size = int(lattice_size)
        self.raw_gates = nn.Parameter(torch.empty(self.lattice_size))
        self.register_buffer(
            "grid",
            torch.linspace(0.0, 1.0, self.lattice_size),
            persistent=False,
        )
        _initialize_hll_gates(self.raw_gates)

    def vertex_values(self) -> torch.Tensor:
        gates = torch.sigmoid(self.raw_gates)
        values = [gates[0]]
        for index in range(1, self.lattice_size):
            lower = values[-1]
            values.append(lower + gates[index] * (1.0 - lower))
        return torch.stack(values)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 1:
            raise ValueError("HierarchicalLattice1D expects a 1D batch.")
        scaled = inputs * (self.lattice_size - 1)
        lower_index = torch.floor(scaled).to(torch.long)
        lower_index = lower_index.clamp(0, self.lattice_size - 2)
        fraction = scaled - lower_index.to(inputs.dtype)
        vertices = self.vertex_values()
        return torch.lerp(
            vertices[lower_index],
            vertices[lower_index + 1],
            fraction,
        )


class HierarchicalLattice2D(nn.Module):
    """Minimal all-increasing two-dimensional HLL."""

    def __init__(self, lattice_sizes: tuple[int, int]) -> None:
        super().__init__()
        if len(lattice_sizes) != 2 or min(lattice_sizes) < 2:
            raise ValueError("lattice_sizes must contain two values >= 2.")
        self.size_x, self.size_y = map(int, lattice_sizes)
        self.raw_gates = nn.Parameter(torch.empty(self.size_x, self.size_y))
        self.register_buffer(
            "grid_x",
            torch.linspace(0.0, 1.0, self.size_x),
            persistent=False,
        )
        self.register_buffer(
            "grid_y",
            torch.linspace(0.0, 1.0, self.size_y),
            persistent=False,
        )
        _initialize_hll_gates(self.raw_gates)

    def vertex_values(self) -> torch.Tensor:
        gates = torch.sigmoid(self.raw_gates)
        values: list[list[torch.Tensor | None]] = [
            [None] * self.size_y for _ in range(self.size_x)
        ]
        for rank in range(self.size_x + self.size_y - 1):
            for x_index in range(self.size_x):
                y_index = rank - x_index
                if not 0 <= y_index < self.size_y:
                    continue
                predecessors: list[torch.Tensor] = []
                if x_index:
                    value = values[x_index - 1][y_index]
                    assert value is not None
                    predecessors.append(value)
                if y_index:
                    value = values[x_index][y_index - 1]
                    assert value is not None
                    predecessors.append(value)
                lower = (
                    torch.stack(predecessors).amax()
                    if predecessors
                    else gates.new_zeros(())
                )
                values[x_index][y_index] = (
                    lower
                    + gates[x_index, y_index] * (1.0 - lower)
                )
        rows: list[torch.Tensor] = []
        for row in values:
            assert all(value is not None for value in row)
            rows.append(torch.stack([value for value in row if value is not None]))
        return torch.stack(rows)

    @staticmethod
    def _coordinate(
        inputs: torch.Tensor,
        lattice_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = inputs * (lattice_size - 1)
        lower_index = torch.floor(scaled).to(torch.long)
        lower_index = lower_index.clamp(0, lattice_size - 2)
        fraction = scaled - lower_index.to(inputs.dtype)
        return lower_index, fraction

    def forward(
        self,
        x_inputs: torch.Tensor,
        y_inputs: torch.Tensor,
    ) -> torch.Tensor:
        if x_inputs.ndim != 1 or x_inputs.shape != y_inputs.shape:
            raise ValueError("HierarchicalLattice2D expects equal 1D batches.")
        x_index, x_fraction = self._coordinate(x_inputs, self.size_x)
        y_index, y_fraction = self._coordinate(y_inputs, self.size_y)
        vertices = self.vertex_values()
        value_00 = vertices[x_index, y_index]
        value_10 = vertices[x_index + 1, y_index]
        value_01 = vertices[x_index, y_index + 1]
        value_11 = vertices[x_index + 1, y_index + 1]
        lower_y = torch.lerp(value_00, value_10, x_fraction)
        upper_y = torch.lerp(value_01, value_11, x_fraction)
        return torch.lerp(lower_y, upper_y, y_fraction)


class PositiveLinear(nn.Module):
    """Dense linear layer with strictly positive softplus weights."""

    def __init__(self, in_features: int, out_features: int, *, seed: int) -> None:
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        _seeded_normal_(self.raw_weight, seed=seed)

    def effective_weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.effective_weight(), self.bias)


class SparseQPositiveLinear(nn.Module):
    """Sparse equivalent of a dense positive layer on a Q-masked vector."""

    def __init__(self, in_features: int, out_features: int, *, seed: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        _seeded_normal_(self.raw_weight, seed=seed)

    def effective_weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight)

    def dense_forward(
        self,
        readiness: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked = readiness * q_mask.to(readiness.dtype)
        return F.linear(masked, self.effective_weight(), self.bias)

    def forward(
        self,
        readiness: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        if readiness.ndim != 2 or readiness.shape != q_mask.shape:
            raise ValueError("Expected matching readiness/q_mask [N, K].")
        row_indices, concept_indices = q_mask.nonzero(as_tuple=True)
        active = readiness[row_indices, concept_indices]
        weights = self.effective_weight()[:, concept_indices].transpose(0, 1)
        contributions = active.unsqueeze(1) * weights
        summed = readiness.new_zeros((readiness.size(0), self.out_features))
        summed.index_add_(0, row_indices, contributions)
        return summed + self.bias.unsqueeze(0)


class _FullSurface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lattice = HierarchicalLattice2D((4, 4))

    def forward(self, readiness, q_mask, bottleneck, aggregate):
        del readiness, q_mask
        return self.lattice(bottleneck, aggregate)


class _PooledDirectSurface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lattice = HierarchicalLattice1D(16)

    def forward(self, readiness, q_mask, bottleneck, aggregate):
        del readiness, q_mask, bottleneck
        return self.lattice(aggregate)


class _CapacityAdditiveSurface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bottleneck_lattice = HierarchicalLattice1D(7)
        self.aggregate_lattice = HierarchicalLattice1D(7)
        self.raw_mix = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, readiness, q_mask, bottleneck, aggregate):
        del readiness, q_mask
        bottleneck_prob = self.bottleneck_lattice(bottleneck)
        aggregate_prob = self.aggregate_lattice(aggregate)
        bottleneck_logit = torch.logit(
            bottleneck_prob.clamp(_PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)
        )
        aggregate_logit = torch.logit(
            aggregate_prob.clamp(_PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)
        )
        mix = torch.sigmoid(self.raw_mix)
        logits = (
            self.bias
            + mix * bottleneck_logit
            + (1.0 - mix) * aggregate_logit
        )
        return torch.sigmoid(logits)


class _StandardNCDSurface(nn.Module):
    def __init__(self, num_concepts: int, *, seed: int) -> None:
        super().__init__()
        self.input_layer = SparseQPositiveLinear(
            num_concepts,
            256,
            seed=seed + 101,
        )
        self.hidden_layer = PositiveLinear(256, 128, seed=seed + 102)
        self.output_layer = PositiveLinear(128, 1, seed=seed + 103)

    def forward(self, readiness, q_mask, bottleneck, aggregate):
        del bottleneck, aggregate
        hidden = torch.sigmoid(self.input_layer(readiness, q_mask))
        hidden = torch.sigmoid(self.hidden_layer(hidden))
        return torch.sigmoid(self.output_layer(hidden).squeeze(-1))


class MonotoneRequirementSurface(nn.Module):
    """Map readiness [N,K], Q mask [N,K], and item offset [N] to prob [N]."""

    def __init__(
        self,
        *,
        num_concepts: int,
        mode: str = "full",
        epsilon: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if num_concepts < 1:
            raise ValueError("num_concepts must be positive.")
        if mode not in SURFACE_MODES:
            raise ValueError(f"Unsupported requirement-surface mode: {mode}")
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1).")
        self.num_concepts = int(num_concepts)
        self.mode = mode
        self.epsilon = float(epsilon)
        self.seed = int(seed)

        self.unary_lattice = HierarchicalLattice1D(16)
        if mode == "full":
            self.variant = _FullSurface()
        elif mode == "pooled_direct":
            self.variant = _PooledDirectSurface()
        elif mode == "capacity_additive":
            self.variant = _CapacityAdditiveSurface()
        else:
            self.variant = _StandardNCDSurface(num_concepts, seed=self.seed)

    def variant_parameters(self) -> Iterable[nn.Parameter]:
        return self.variant.parameters()

    @property
    def active_variant_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.variant_parameters())

    def _validate_inputs(
        self,
        readiness: torch.Tensor,
        q_mask: torch.Tensor,
        item_offset: torch.Tensor,
    ) -> torch.Tensor:
        if readiness.ndim != 2:
            raise ValueError("readiness must have shape [N, K].")
        if readiness.size(1) != self.num_concepts:
            raise ValueError(
                f"Expected K={self.num_concepts}, got K={readiness.size(1)}."
            )
        if q_mask.shape != readiness.shape:
            raise ValueError("q_mask must have the same shape as readiness.")
        if item_offset.shape != readiness.shape[:1]:
            raise ValueError("item_offset must have shape [N].")
        if q_mask.dtype == torch.bool:
            boolean_mask = q_mask
        else:
            if not bool(torch.logical_or(q_mask == 0, q_mask == 1).all()):
                raise ValueError("q_mask must be boolean or binary.")
            boolean_mask = q_mask.to(torch.bool)
        counts = boolean_mask.sum(dim=1)
        if not bool((counts > 0).all()):
            raise ValueError("Every row must contain at least one Q concept.")
        active = readiness.masked_select(boolean_mask)
        if not bool(torch.isfinite(active).all()):
            raise ValueError("Active readiness values must be finite.")
        if not bool(torch.logical_and(active >= 0.0, active <= 1.0).all()):
            raise ValueError("Active readiness values must be in [0, 1].")
        if not bool(torch.isfinite(item_offset).all()):
            raise ValueError("item_offset values must be finite.")
        return boolean_mask

    @staticmethod
    def _summaries(
        readiness: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = q_mask.sum(dim=1)
        aggregate = (
            readiness * q_mask.to(readiness.dtype)
        ).sum(dim=1) / counts.to(readiness.dtype)
        bottleneck = readiness.masked_fill(~q_mask, 1.0).amin(dim=1)
        return bottleneck, aggregate, counts

    def _learned_surface(
        self,
        readiness: torch.Tensor,
        boolean_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck, aggregate, counts = self._summaries(
            readiness,
            boolean_mask,
        )
        unary_surface = self.unary_lattice(aggregate)
        variant_surface = self.variant(
            readiness,
            boolean_mask,
            bottleneck,
            aggregate,
        )
        return torch.where(counts == 1, unary_surface, variant_surface), aggregate

    def learned_surface(
        self,
        readiness: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mechanism surface before epsilon blending and item offset."""
        zeros = readiness.new_zeros(readiness.size(0))
        boolean_mask = self._validate_inputs(readiness, q_mask, zeros)
        learned, _ = self._learned_surface(readiness, boolean_mask)
        return learned

    def requirement_surface(
        self,
        readiness: torch.Tensor,
        q_mask: torch.Tensor,
        item_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if item_offset is None:
            item_offset = readiness.new_zeros(readiness.size(0))
        boolean_mask = self._validate_inputs(readiness, q_mask, item_offset)
        learned, aggregate = self._learned_surface(readiness, boolean_mask)
        return (1.0 - self.epsilon) * learned + self.epsilon * aggregate

    def forward(
        self,
        readiness: torch.Tensor,
        q_mask: torch.Tensor,
        item_offset: torch.Tensor,
    ) -> torch.Tensor:
        surface = self.requirement_surface(readiness, q_mask, item_offset)
        logits = torch.logit(
            surface.clamp(_PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)
        )
        return torch.sigmoid(item_offset + logits)
