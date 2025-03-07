from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from typing import Tuple, List


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

        self.time_embedding_dim = time_embedding_dim

        self.device = device
        self.to(device)

    def to(self, *args, **kwargs):
        self.time_embs = self.time_embs.to(*args, **kwargs)
        return super(TimeEmbeddingBlock, self).to(*args, **kwargs)

    def forward(self, time: _T, *args):
        """
        Input: [...]
        Output: [..., time embedding dim]
        """
        assert (time >= 0).all() and (time < self.T).all()
        embs = self.time_embs[time]  # [T selected, emb dim]
        return self.layers(embs)



class SmoothEmbeddingBlockWithExtraEmbeddings(TimeEmbeddingBlock):
    """
    Positive indices call TimeEmbeddingBlock
    Negative indices index an embedding block

    Used for classical conditioning RL tasks, only inherits from TimeEmbeddingBlock so that we can use the smooth embeddings
    """

    def __init__(self, total_time: int, time_embedding_dim: int, num_extra_embeddings: int, device="cuda"):
        super().__init__(total_time, time_embedding_dim, device)
        
        self.num_extra_embeddings = num_extra_embeddings
        self.neg_embeddings = nn.Embedding(num_extra_embeddings, time_embedding_dim)

    def forward(self, state_idx: _T, num_diffusion_timesteps: int) -> _T:
        """
        Expecting state_idx of shape [batch, task timesteps]
        """
        state_idx = state_idx.unsqueeze(-1).repeat_interleave(num_diffusion_timesteps, -1)
        canvas = torch.zeros_like(state_idx).unsqueeze(-1).to(device=self.device)              # [B, T_task, T_diff, D] = [2048, 8, 20, 16]
        canvas = canvas.repeat_interleave(self.time_embedding_dim, -1)
        positive_time = state_idx[state_idx >= 0]
        smooth_res = super(SmoothEmbeddingBlockWithExtraEmbeddings, self).forward(positive_time)
        canvas = canvas.to(smooth_res.dtype)
        canvas[state_idx >= 0] = smooth_res
        canvas[state_idx < 0] = self.neg_embeddings(- state_idx[state_idx < 0] - 1)
        return canvas


class HierarchicalEmbeddingsBlock(nn.Module):
    def __init__(self, time_embedding_dim: int, num_embeddings: Tuple[int], device="cuda"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings = nn.ModuleList(
            [nn.Embedding(ne, time_embedding_dim) for ne in num_embeddings]
        )
        self.device = device
        self.to(device)
        self.network_input_size = time_embedding_dim * len(num_embeddings)

    def forward(self, embedding_idx: List[_T], num_diffusion_timesteps: int) -> _T:
        """
        expecting embedding_idx of the same shapes, [...]
        return shaped [..., num_diffusion_timesteps, time_embedding_dim * len(hierarchy)]
        """
        all_embeddings = []
        for embedding_farm, eidx in zip(self.embeddings, embedding_idx):
            all_embeddings.append(embedding_farm(eidx))
        all_embeddings = torch.concat(all_embeddings, -1).unsqueeze(-2).repeat_interleave(num_diffusion_timesteps, -2)
        return all_embeddings
