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
