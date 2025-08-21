import types
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from gazebot.train import train, get_args
from gazebot.dataset import create_manipulation_dataset
from gazebot.agent import create_manipulation_agent


"""
Training script for the action prediction model to determine bottleneck steps (called as localbc)
    - The model predict action sequence from only gaze centered point cloud
"""


@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Get config
    args = get_args(cfg)

    # Create dataset
    train_dataset, test_dataset = create_manipulation_dataset(args)

    # Create agent
    args.agent.enc_layers = 1  # NOTE: Use small transformer for localbc
    args.agent.dec_layers = 1
    args.agent.dim_feedforward = 1024
    agent = create_manipulation_agent(args)

    # Train
    agent.update = types.MethodType(update, agent)
    agent.validate_fn = types.MethodType(validate_fn, agent)

    train_data = DataLoader(train_dataset, args.train.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_data = DataLoader(test_dataset, args.train.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    args.train.max_epoch = 40
    agent, best_step = train(agent, train_data, test_data, args)
    print(f"best_step = {best_step}")


def update(self, batch):
    obs, action_pose, gaze, is_pad, is_done, sub_task_idx, step, bottleneck_step = batch

    # To cuda
    for key in obs.keys():
        obs[key] = obs[key].to(self.device)
    action_pose = action_pose.to(self.device)
    gaze = gaze.to(self.device)
    is_pad = is_pad.to(self.device)

    self.model.train()
    self.model.zero_grad()

    action_hat, _ = self.local_bc(obs, gaze)

    target_is_pad = torch.empty(action_hat.shape[:2], dtype=torch.bool, device=self.device)
    target_action = torch.empty_like(action_hat)
    for b in range(len(action_pose)):
        target_is_pad[b] = is_pad[b, step[b] : step[b] + self.model.localbc_action_chunk_size]
        target_action[b] = action_pose[b, step[b] : step[b] + self.model.localbc_action_chunk_size]

    # Loss
    all_l1 = F.l1_loss(action_hat, target_action, reduction="none")  # (B, chunk_size, state_dim)
    loss_localbc_action = torch.sum(all_l1 * ~target_is_pad.unsqueeze(-1)) / torch.sum(
        torch.ones_like(all_l1) * ~target_is_pad.unsqueeze(-1)
    )  # (B, chunk_size, state_dim)

    loss = loss_localbc_action

    loss.backward()
    self.optimizer.step()
    self.model.eval()

    train_info = {"loss": loss.item(), "loss_localbc_action": loss_localbc_action.item()}

    return train_info


def validate_fn(self, batch):
    obs, action_pose, gaze, is_pad, is_done, sub_task_idx, step, bottleneck_step = batch

    # To cuda
    for key in obs.keys():
        obs[key] = obs[key].to(self.device)
    action_pose = action_pose.to(self.device)
    gaze = gaze.to(self.device)
    is_pad = is_pad.to(self.device)

    self.model.eval()

    with torch.inference_mode():
        action_hat, _ = self.local_bc(obs, gaze)

    target_is_pad = torch.empty(action_hat.shape[:2], dtype=torch.bool, device=self.device)
    target_action = torch.empty_like(action_hat)
    for b in range(len(action_pose)):
        target_is_pad[b] = is_pad[b, step[b] : step[b] + self.model.localbc_action_chunk_size]
        target_action[b] = action_pose[b, step[b] : step[b] + self.model.localbc_action_chunk_size]

    all_l1 = F.l1_loss(action_hat, target_action, reduction="none")  # (B, chunk_size, state_dim)
    loss_localbc_action = torch.sum(all_l1 * ~target_is_pad.unsqueeze(-1)) / torch.sum(
        torch.ones_like(all_l1) * ~target_is_pad.unsqueeze(-1)
    )  # (B, chunk_size, state_dim)

    loss = loss_localbc_action

    self.model.train()

    val_info = {"loss": loss.item(), "loss_localbc_action": loss_localbc_action.item()}

    print(f"agent action:\n{action_hat[0][0]}\naction data:\n{action_pose[0][0]}")

    return loss.item(), val_info


if __name__ == "__main__":
    main()
