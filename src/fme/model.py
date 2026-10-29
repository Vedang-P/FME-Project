"""Reliability-aware temporal peer graph model implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from .graphs import SnapshotGraph


class ReliabilityTemporalGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, relations: list[str], dropout: float = 0.2):
        super().__init__()
        self.relations = relations
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.message_projection = nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in relations})
        self.relation_gate = nn.Linear(hidden_dim, max(len(relations), 1))
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _aggregate(x: torch.Tensor, graph: SnapshotGraph, relation: str) -> torch.Tensor:
        relation_graph = graph.relations[relation]
        aggregate = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        if relation_graph.source.size == 0:
            return aggregate
        source = torch.as_tensor(relation_graph.source, dtype=torch.long, device=x.device)
        target = torch.as_tensor(relation_graph.target, dtype=torch.long, device=x.device)
        weight = torch.as_tensor(relation_graph.weight, dtype=x.dtype, device=x.device).unsqueeze(1)
        aggregate.index_add_(0, target, x[source] * weight)
        degree.index_add_(0, target, weight)
        return aggregate / degree.clamp_min(1e-8)

    def forward(self, sequences: torch.Tensor, graph: SnapshotGraph) -> torch.Tensor:
        _, (hidden, _) = self.encoder(sequences)
        temporal = hidden[-1]
        if not self.relations:
            combined = temporal
        else:
            gates = torch.softmax(self.relation_gate(temporal), dim=-1)
            messages = torch.zeros_like(temporal)
            for relation_idx, relation in enumerate(self.relations):
                aggregate = self._aggregate(temporal, graph, relation)
                messages = messages + gates[:, relation_idx : relation_idx + 1] * self.message_projection[relation](aggregate)
            combined = self.update(torch.cat([self.self_projection(temporal), self.dropout(messages)], dim=-1))
        return self.classifier(combined).squeeze(-1)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


class IndependentFeatureLSTM(nn.Module):
    """A separate recurrent head for every financial series.

    The feature dimension is treated as a batch of independent LSTMs with
    feature-specific weights. This is equivalent to running one small LSTM per
    input series, but vectorizes the recurrence over firms and features.
    Inputs have shape ``[batch, time, features, channels]``; channels can hold
    the normalized value and, optionally, a missingness indicator.
    """

    def __init__(self, n_features: int, input_channels: int = 1, hidden_dim: int = 8):
        super().__init__()
        self.n_features = n_features
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        gates = 4 * hidden_dim
        self.weight_ih = nn.Parameter(torch.empty(n_features, input_channels, gates))
        self.weight_hh = nn.Parameter(torch.empty(n_features, hidden_dim, gates))
        self.bias_ih = nn.Parameter(torch.zeros(n_features, gates))
        self.bias_hh = nn.Parameter(torch.zeros(n_features, gates))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for feature in range(self.n_features):
            nn.init.xavier_uniform_(self.weight_ih[feature])
            nn.init.orthogonal_(self.weight_hh[feature])
        with torch.no_grad():
            self.bias_ih[:, self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 4:
            raise ValueError("Expected [batch, time, features, channels] input.")
        batch, steps, features, channels = sequences.shape
        if features != self.n_features or channels != self.input_channels:
            raise ValueError("Input feature/channel dimensions do not match the LSTM heads.")
        hidden = sequences.new_zeros((batch, features, self.hidden_dim))
        cell = sequences.new_zeros((batch, features, self.hidden_dim))
        for step in range(steps):
            values = sequences[:, step]
            input_gates = torch.matmul(values.unsqueeze(-2), self.weight_ih).squeeze(-2)
            recurrent_gates = torch.matmul(hidden.unsqueeze(-2), self.weight_hh).squeeze(-2)
            gates = input_gates + recurrent_gates + self.bias_ih + self.bias_hh
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
            hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden


class MultiHeadTemporalClassifier(nn.Module):
    """Independent per-feature LSTM heads with concat or attention pooling."""

    def __init__(
        self,
        n_features: int,
        *,
        input_channels: int = 1,
        hidden_dim: int = 8,
        pooling: str = "concat",
        dropout: float = 0.25,
        input_skip: bool = False,
    ):
        super().__init__()
        if pooling not in {"concat", "attention"}:
            raise ValueError("pooling must be 'concat' or 'attention'.")
        self.encoder = IndependentFeatureLSTM(n_features, input_channels, hidden_dim)
        self.pooling = pooling
        self.input_skip = input_skip
        if pooling == "attention":
            self.feature_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            representation_dim = hidden_dim
        else:
            representation_dim = n_features * hidden_dim
        if input_skip:
            # Preserve the latest accounting state and its most recent change
            # alongside the recurrent summary for each feature.
            representation_dim += 2 * n_features * input_channels
        middle = max(16, min(128, representation_dim // 2))
        self.classifier = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, middle),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(middle, 1),
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        heads = self.encoder(sequences)
        if self.pooling == "attention":
            weights = torch.softmax(self.feature_attention(heads), dim=1)
            representation = torch.sum(heads * weights, dim=1)
        else:
            representation = heads.flatten(start_dim=1)
        if self.input_skip:
            latest = sequences[:, -1].flatten(start_dim=1)
            delta = (sequences[:, -1] - sequences[:, -2]).flatten(start_dim=1)
            representation = torch.cat([representation, latest, delta], dim=1)
        return self.classifier(representation).squeeze(-1)
