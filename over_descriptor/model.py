import sys
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

superpoint_path = os.path.join(project_root, "superpoint")

if project_root not in sys.path:
    sys.path.append(project_root)

if superpoint_path not in sys.path:
    sys.path.append(superpoint_path)

import torch
from torch import nn
import torch.nn.functional as F
from superpoint.superpoint import SuperPointFrontend


# Using the same architecture as the mixVPR
class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class ReprDescriptor(nn.Module):
    def __init__(
        self,
        in_h,
        in_w,
        in_channels,
        out_channels,
        out_rows,
        mix_depth,
        mlp_ratio,
        *args,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = (
            mlp_ratio  # ratio of the mid projection layer in the mixer block
        )

        hw = self.in_h * self.in_w
        self.mix = nn.Sequential(
            *[
                FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
                for _ in range(self.mix_depth)
            ]
        )
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)
        self.final_proj = nn.Linear(out_channels * out_rows, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.mix(x)
        x = x.permute(1, 0)
        x = self.channel_proj(x)
        x = x.permute(1, 0)
        x = self.row_proj(x)
        x = self.final_proj(x.flatten(0))
        x = F.normalize(x, p=2, dim=0)
        return x


class network(nn.Module):
    def __init__(
        self,
        weights_path,
        features_dim: int = 256,
        nms_dist: int = 4,
        conf_thresh: float = 0.015,
        nn_thresh: float = 0.7,
        cuda: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.local_descriptor_extractor = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=cuda,
        )

        for param in self.local_descriptor_extractor.net.parameters():
            param.requires_grad = False

        # self.over_descriptor = ReprDescriptor(
        #     in_h=32,
        #     in_w=32,
        #     in_channels=256,
        #     out_channels=256,
        #     out_rows=64,
        #     mix_depth=4,
        #     mlp_ratio=1,
        # )

        self.mlp = nn.Sequential(
            nn.Linear(256 * 60 * 80, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_descriptor_extractor.run(x)
        N = x.shape[0]
        # TODO: mask
        # x = self.over_descriptor(x)
        x = x.reshape(N, -1)
        # Convert x to tensor
        x = torch.Tensor(x).to("cuda")
        x = self.mlp(x)
        return x
