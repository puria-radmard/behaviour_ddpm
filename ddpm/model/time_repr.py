from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T



class TimeEmbeddingBlock(nn.Module):

    def __init__(self, total_time: int, time_embedding_dim: int, device = 'cuda'):
        super().__init__()
        frequencies = 2 * torch.pi * torch.arange(1, time_embedding_dim + 1).unsqueeze(0)
        times = torch.linspace(0.0, 1.0, total_time+1)[:-1].unsqueeze(1)
        self.sin_embs = (frequencies * times).sin().to(device)     # [T, emb dim]
        self.layers = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.T = total_time

    def to(self, *args, **kwargs):
        self.sin_embs = self.sin_embs.to(*args, **kwargs)
        return super(TimeEmbeddingBlock, self).to(*args, **kwargs)

    def forward(self, time: _T):
        """
        TODO: docstring
        """
        assert (time >= 0).all() and (time <= self.T).all()
        embs = self.sin_embs[time]  # [T selected, emb dim]
        return self.layers(embs)


