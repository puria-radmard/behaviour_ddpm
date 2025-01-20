from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T


class TimeEmbeddingBlock(nn.Module):

    def __init__(self, total_time: int, time_embedding_dim: int, device="cuda"):
        super().__init__()

        assert time_embedding_dim % 2 == 0, "time embedding must be divisible by 2."

        # frequencies = 2 * torch.pi * torch.arange(1, time_embedding_dim + 1).unsqueeze(0)
        # times = torch.linspace(0.0, 1.0, total_time+1)[:-1].unsqueeze(1)
        # self.time_embs = (frequencies * times).sin().to(device)     # [T, emb dim]

        # Editted from https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch#Diffusion-Model---The-Intuition
        factor = (
            2
            * torch.arange(
                start=0, end=time_embedding_dim // 2, dtype=torch.float32, device=device
            )
            / time_embedding_dim
        )
        factor = 10000 ** factor.unsqueeze(0)

        times = (
            torch.arange(1, total_time + 1).unsqueeze(1).to(device) / factor
        )  # [total_time, time_embedding_dim // 2]
        self.time_embs = torch.cat([torch.sin(times), torch.cos(times)], dim=1)

        self.layers = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.T = total_time

    def to(self, *args, **kwargs):
        self.time_embs = self.time_embs.to(*args, **kwargs)
        return super(TimeEmbeddingBlock, self).to(*args, **kwargs)

    def forward(self, time: _T):
        """
        TODO: docstring
        """
        assert (time >= 0).all() and (time <= self.T).all()
        embs = self.time_embs[time]  # [T selected, emb dim]
        return self.layers(embs)
