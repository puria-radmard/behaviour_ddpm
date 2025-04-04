import torch
from torch import nn
from torch import vmap
from torch import Tensor as _T

from typing import List, Mapping, Any, Optional


from ddpm.model.unet import UNet




class UNetResidualModel(nn.Module):

    def __init__(
        self,
        image_size: int,
        input_size: int,
        time_embedding_size: int,
        num_channels: int,
        base_channels: int = 64,
    ) -> None:

        self.input_size = input_size

        super().__init__()
        self.image_size = image_size
        self.time_embedding_size = time_embedding_size
        self.num_channels = num_channels
        self.base_channels = base_channels

        print("NOT PASSING INPUT TO UNET FOR NOW!!!")

        total_input_vector_size = time_embedding_size  #  + input_size
        self.unet = UNet(
            image_size=image_size,
            num_channels=num_channels,
            vector_dim=total_input_vector_size,
            base_channels=base_channels,
        )

    def forward(self, x: _T, t_embeddings_schedule: _T, input_vector: _T) -> _T:
        """
        x of shape [B, T, num_channels, image_size, image_size]
        t_embeddings_schedule of shape [T, t_emb_size]
        input_vector of shape [B, T, input_size]
        """
        raise Exception("Make start dims general again!")
        reshaped_t_schedule = t_embeddings_schedule.unsqueeze(0).expand(
            x.shape[0], 1, 1
        )
        total_input_vector = reshaped_t_schedule  # torch.concat([input_vector, reshaped_t_schedule], -1).float()
        return self.unet(x, total_input_vector)

