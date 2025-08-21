from typing import Optional, List
import torch
import torch.nn as nn
import torchvision.transforms.functional as VF


class GazeModel(nn.Module):
    """
    Gaze model of similar and improved architecture to that proposed at https://arxiv.org/pdf/2401.07603
    """

    def __init__(self, num_segs: int, small_image_dim: List[int], super_res: Optional[int] = None):
        super().__init__()

        hidden_dim = 768
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        self.small_image_dim = small_image_dim

        self.super_res = super_res
        self.gaze_heads = nn.ModuleList([GazeHead(hidden_dim, super_res) for _ in range(num_segs)])

        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("[GazeModel] number of parameters: %.2fM" % (n_parameters / 1e6,))

    def forward(self, image: torch.Tensor, seg_idx: torch.LongTensor, denormalize: bool = False) -> torch.Tensor:
        B, _, C, H, W = image.shape
        device = image.device

        image = VF.resize(image.flatten(0, 1), size=self.small_image_dim[1:])  # (B * 2, C, H, W)

        feature_map = self.backbone.get_intermediate_layers(image, reshape=True)[0]  # (B * 2, hidden_dim, gridH, gridW)

        gaze_grid = torch.empty(
            B * 2, self.super_res * feature_map.shape[2], self.super_res * feature_map.shape[3], dtype=torch.float, device=device
        )  # (B * 2, gridH, gridW)
        for i in range(len(self.gaze_heads)):
            if torch.sum(seg_idx == i) > 0:
                mask = (seg_idx == i).squeeze(1).repeat_interleave(2)  # (B * 2,)
                gaze_grid[mask] = self.gaze_heads[i](feature_map[mask])  # (B' * 2, gridH, gridW)

        cent_grid = self.grid_coord(gaze_grid).reshape(-1, 2, 2)  # (B, 2, 2)
        if self.training:
            cent_grid = torch.clip(cent_grid + 0.01 * (2 * torch.rand_like(cent_grid) - 1), 0, 1)

        if denormalize:
            return self.denormalize(cent_grid, H, W), gaze_grid  # (B, 2, 2), (B * 2, gridH, gridW)
        return cent_grid, gaze_grid  # (B, 2, 2), (B * 2, gridH, gridW)

    def grid_coord(self, gaze_grid: torch.Tensor) -> torch.Tensor:
        B2, H, W = gaze_grid.shape
        idx = torch.argmax(gaze_grid.flatten(1), dim=1)  # (B,)
        x = (idx % W + 0.5) / W  # [0, 1)
        y = (idx // W + 0.5) / H  # [0, 1)
        return torch.stack([x, y], dim=-1)  # (B, 2)

    def denormalize(self, gaze_norm: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
        # [0, 1] -> [0, W], [0, H]
        gaze = torch.zeros_like(gaze_norm)  # (B, 2, 2)
        gaze[:, :, 0] = gaze_norm[:, :, 0] * image_width
        gaze[:, :, 1] = gaze_norm[:, :, 1] * image_height

        return torch.round(gaze).to(torch.long)


class GazeHead(nn.Module):
    def __init__(self, hidden_dim: int, super_res: Optional[int] = None):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.ConvTranspose2d(32, 1, kernel_size=super_res or 1, stride=super_res or 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)  # (B, 1, H*N, W*N)

        return x.squeeze(1)  # (B, H*N, W*N)
