import types
import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import cv2

from gazebot.train import train, get_args
from gazebot.agent import create_gaze_agent
from gazebot.dataset import create_gaze_dataset
from gazebot.utils import array2image, tensor2opencv


@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Get config
    args = get_args(cfg)

    assert args.agent.name == "gaze"

    # Create agent
    agent = create_gaze_agent(args)

    agent.update = types.MethodType(update, agent)
    agent.validate_fn = types.MethodType(validate_fn, agent)

    # Load demonstrations
    train_dataset, test_dataset = create_gaze_dataset(args)

    train_data = DataLoader(train_dataset, args.train.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_data = DataLoader(test_dataset, args.train.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Train agent
    agent, best_step = train(agent, train_data, test_data, args)
    print(f"best_step = {best_step}")


def update(self, batch):
    image, gaze, seg_idx = batch

    # To cuda
    image = image.to(self.device)  # (B, 2, C, H, W)
    gaze = gaze.to(self.device)

    self.model.train()
    self.model.zero_grad()

    gaze_hat, gaze_probs = self.model(image, seg_idx)

    target_probs = create_one_hot_targets(gaze.flatten(0, 1), grid_size=gaze_probs.shape[1:])  # (B * 2, gridH*gridW)
    loss = F.cross_entropy(gaze_probs.flatten(1), target_probs)

    loss.backward()
    self.optimizer.step()

    train_info = dict(loss=loss.item())

    return train_info


def validate_fn(self, batch):
    image, gaze, seg_idx = batch

    # To cuda
    image = image.to(self.device)  # (B, 2, C, H, W)
    gaze = gaze.to(self.device)

    self.model.eval()

    B, _, C, H, W = image.shape

    with torch.no_grad():
        gaze_hat, gaze_probs = self.model(image, seg_idx)

    target_probs = create_one_hot_targets(gaze.flatten(0, 1), grid_size=gaze_probs.shape[1:])  # (B * 2, gridH*gridW)
    loss = F.cross_entropy(gaze_probs.flatten(1), target_probs)

    val_info = dict(loss=loss.item())

    # Visualize
    human_gaze = self.model.denormalize(gaze, H, W).detach().cpu().numpy()  # (B, 2, 2), [W, H]
    human_gaze = np.round(human_gaze).astype(np.int64)  # (B, 2, 2)
    predict_gaze = self.model.denormalize(gaze_hat, H, W).detach().cpu().numpy()  # (B, 2, 2), [W, H]
    predict_gaze = np.round(predict_gaze).astype(np.int64)  # (B, 2, 2)

    im = tensor2opencv(image[0, 0])  # (H, W, 3)
    im = cv2.circle(im, (human_gaze[0, 0, 0], human_gaze[0, 0, 1]), 20, (0, 0, 255), 5)
    im = cv2.circle(im, (predict_gaze[0, 0, 0], predict_gaze[0, 0, 1]), 20, (255, 255, 255), 5)
    array2image(im[:, :, [2, 1, 0]]).resize((320, 180)).save("val_gaze.png")

    print(f"val_loss = {loss.item()}")
    print(f"gaze_hat:\n{gaze_probs.detach().cpu().numpy()[: B // 8 + 1]}\ngaze:\n{target_probs.detach().cpu().numpy()[: B // 8 + 1]}")
    print(f"predict gaze:\n{predict_gaze[: B // 8 + 1]}\nhuman gaze:\n{human_gaze[: B // 8 + 1]}")

    return loss.item(), val_info


def create_one_hot_targets(positions: torch.Tensor, grid_size: Tuple[int, int]):
    """
    Given normalized target positions [x, y] within the image (values between 0 and 1),
    creates a one-hot vector of size 196 indicating which grid cell (out of 14x14 grids)
    the target falls into.

    Args:
        positions (torch.Tensor): Tensor of shape (B, 2) containing the normalized
                                  (x, y) positions of the target within the image.
        grid_size (tuple[int, int]): grid size (H, W)

    Returns:
        one_hot_targets (torch.Tensor): Tensor of shape (B, 196) containing one-hot
                                        vectors indicating the grid cell for each target.
    """
    import torch

    B, _ = positions.size()
    num_cells = grid_size[0] * grid_size[1]  # Total number of grid cells (196)

    # Extract x and y positions
    x = positions[:, 0]  # Shape: (B,)
    y = positions[:, 1]  # Shape: (B,)

    # Compute grid indices along x and y axes
    # Multiply by grid_size and convert to integer indices
    grid_x = (x * grid_size[1]).long()  # Shape: (B,)
    grid_y = (y * grid_size[0]).long()  # Shape: (B,)

    # Ensure indices are within valid range [0, grid_size - 1]
    grid_x = torch.clamp(grid_x, 0, grid_size[1] - 1)
    grid_y = torch.clamp(grid_y, 0, grid_size[0] - 1)

    # Compute linear indices for the one-hot encoding
    # Each grid cell is assigned a unique index from 0 to 195
    grid_indices = grid_y * grid_size[1] + grid_x  # Shape: (B,)

    # Initialize one-hot vectors
    one_hot_targets = torch.zeros(B, num_cells, device=positions.device)

    # Set the corresponding indices to 1
    one_hot_targets[torch.arange(B), grid_indices] = 1

    return one_hot_targets


if __name__ == "__main__":
    main()
