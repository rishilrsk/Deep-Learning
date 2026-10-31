"""CNN encoder and additive-attention LSTM decoder for image captioning."""

from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention over the encoder's spatial image features."""

    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, attention_dim)
        self.hidden_projection = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        scores = self.score(
            torch.tanh(
                self.feature_projection(features)
                + self.hidden_projection(hidden).unsqueeze(1)
            )
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return context, weights


class AttentionLSTMCaptioner(nn.Module):
    """Decode cached ResNet spatial features into a word sequence."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 2048,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.initial_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.initial_cell = nn.Linear(hidden_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = AdditiveAttention(hidden_dim, hidden_dim, attention_dim)
        self.lstm = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 2, vocab_size)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.feature_projection(features))

    def forward(self, raw_features: torch.Tensor, captions: torch.Tensor):
        regions = self.project_features(raw_features)
        pooled = regions.mean(dim=1)
        hidden = torch.tanh(self.initial_hidden(pooled))
        cell = torch.tanh(self.initial_cell(pooled))
        logits = []

        for step in range(captions.size(1) - 1):
            context, _ = self.attention(regions, hidden)
            word_embedding = self.embedding(captions[:, step])
            hidden, cell = self.lstm(
                torch.cat((word_embedding, context), dim=1), (hidden, cell)
            )
            logits.append(self.output(self.dropout(torch.cat((hidden, context), dim=1))))

        return torch.stack(logits, dim=1)

    @torch.inference_mode()
    def generate(self, raw_features: torch.Tensor, start_id: int, end_id: int, max_len: int = 32):
        regions = self.project_features(raw_features)
        pooled = regions.mean(dim=1)
        hidden = torch.tanh(self.initial_hidden(pooled))
        cell = torch.tanh(self.initial_cell(pooled))
        token = torch.tensor([start_id], device=raw_features.device)
        generated = []

        for _ in range(max_len):
            context, _ = self.attention(regions, hidden)
            word_embedding = self.embedding(token)
            hidden, cell = self.lstm(
                torch.cat((word_embedding, context), dim=1), (hidden, cell)
            )
            token = self.output(torch.cat((hidden, context), dim=1)).argmax(dim=1)
            token_id = int(token.item())
            if token_id == end_id:
                break
            generated.append(token_id)
        return generated
