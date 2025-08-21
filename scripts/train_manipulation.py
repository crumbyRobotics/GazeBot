import types
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from gazebot.train import train, get_args
from gazebot.dataset import create_manipulation_dataset
from gazebot.agent import create_manipulation_agent
from gazebot.utils import bezier_term, bezier


@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Get config
    args = get_args(cfg)

    assert args.agent.name == "manipulation"

    # Load demonstrations
    train_dataset, test_dataset = create_manipulation_dataset(args)

    # Create agent
    agent = create_manipulation_agent(args)

    # Train
    agent.update = types.MethodType(update, agent)
    agent.validate_fn = types.MethodType(validate_fn, agent)

    train_data = DataLoader(train_dataset, args.train.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_data = DataLoader(test_dataset, args.train.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)

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
    is_done = is_done.to(self.device)

    self.model.train()
    self.model.zero_grad()

    B, state_dim = obs["state"].shape
    num_arm = 2

    last_action = action_pose[torch.arange(B), torch.where(step > 0, step - 1, step)]  # (B, state_dim)
    _, localbc_action, is_done_hat, bottleneck_hat, reaching_info, _ = self(obs, gaze, last_action)

    target_is_pad = torch.empty(localbc_action.shape[:2], dtype=torch.bool, device=self.device)
    target_action_pose = torch.empty_like(localbc_action)  # (B, localbc_action_chunk_size)
    for b in range(B):
        target_is_pad[b] = is_pad[b, step[b] : step[b] + self.model.localbc_action_chunk_size]
        target_action_pose[b] = action_pose[b, step[b] : step[b] + self.model.localbc_action_chunk_size]

    lr_idx = torch.BoolTensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])

    # bezier vector loss
    loss_bezier_vector = torch.tensor(0, dtype=torch.float, device=self.device)
    for b in range(B):
        for lr in range(2):
            if bottleneck_step[b, lr] < step[b] + 2:
                continue

            target_trajectory = action_pose[
                b, torch.where(step[b] > 0, step[b] - 1, step[b]) : bottleneck_step[b, lr], lr_idx[lr]
            ].cpu()  # (Ntraj, state_dim // 2)

            start_point = target_trajectory[0]  # (state_dim // 2,)
            goal_point = target_trajectory[-1]  # (state_dim // 2,)
            bezier_point = (start_point + goal_point) / 2  # (state_dim // 2,)

            bezier_points = torch.stack([start_point, bezier_point, goal_point]).T  # (state_dim // 2, 3)

            # bezier fitting
            t = torch.linspace(0, 1, len(target_trajectory) * 5)
            f = bezier_term(3, t)  # (3, 5 * Ntraj)
            for _ in range(5):  # 一回で収束しないことが多いので何回かイテレーション
                t_opt = (
                    ((bezier(bezier_points, t).T[..., None] - target_trajectory.T[None, ...]) ** 2).sum(1).argmin(0)
                )  # t_opt = argmin(pi-P(t)), shape: (Ntraj,)
                # print("convergence:", np.sum((bezier(bezier_points, t[t_opt]).T - target_trajectory) ** 2))
                if torch.sum(f[1, t_opt] * f[1, t_opt]) > 1e-12:
                    bezier_points[:, 1] = (
                        torch.sum(target_trajectory * f[1, t_opt, None], axis=0)
                        - bezier_points[:, 0] * torch.sum(f[0, t_opt] * f[1, t_opt])
                        - bezier_points[:, 2] * torch.sum(f[1, t_opt] * f[2, t_opt])
                    ) / torch.sum(f[1, t_opt] * f[1, t_opt])
                else:
                    bezier_points[:, 1] = (start_point + goal_point) / 2
            # t_opt = ((bezier(bezier_points, t).T[..., None] - target_trajectory.T[None, ...]) ** 2).sum(1).argmin(0)  # t_opt = argmin(pi-P(t)), shape: (Ntraj,)

            bezier_vector = bezier_points[:, 1] - (start_point + goal_point) / 2  # (state_dim // 2,)
            loss_bezier_vector += F.l1_loss(reaching_info["bezier_vector"][b, lr_idx[lr]], bezier_vector.to(self.device)) / B / 2

    # bottleneck loss
    bottleneck = torch.cat(
        [
            action_pose[torch.arange(B), bottleneck_step[:, 0], : state_dim // num_arm],
            action_pose[torch.arange(B), bottleneck_step[:, 1], state_dim // num_arm :],
        ],
        dim=1,
    )  # (B, state_dim)
    loss_left_bottleneck = F.l1_loss(bottleneck_hat[:, : state_dim // num_arm], bottleneck[:, : state_dim // num_arm])
    loss_right_bottleneck = F.l1_loss(bottleneck_hat[:, state_dim // num_arm :], bottleneck[:, state_dim // num_arm :])

    # localbc Loss
    before_bottleneck = (
        torch.cat(
            [
                torch.where(
                    (step < bottleneck_step[:, 0]).unsqueeze(-1), torch.ones(state_dim // num_arm), torch.zeros(state_dim // num_arm)
                ),  # (B, state_dim // num_arm)
                torch.where(
                    (step < bottleneck_step[:, 1]).unsqueeze(-1), torch.ones(state_dim // num_arm), torch.zeros(state_dim // num_arm)
                ),  # (B, state_dim // num_arm)
            ],
            dim=-1,
        )
        .to(torch.bool)
        .to(self.device)
    )  # (B, state_dim)
    if torch.all(before_bottleneck):
        loss_localbc_action = torch.tensor(0, dtype=torch.float, device=self.device)
    else:
        all_l1 = F.l1_loss(localbc_action, target_action_pose, reduction="none")  # (B, localbc_action_chunk_size, state_dim)
        loss_localbc_action = torch.sum(all_l1 * ~target_is_pad.unsqueeze(-1) * ~before_bottleneck.unsqueeze(1)) / torch.sum(
            torch.ones_like(all_l1) * ~target_is_pad.unsqueeze(-1) * ~before_bottleneck.unsqueeze(1)
        )  # (B, localbc_action_chunk_size, state_dim)

    # gaze transition Loss
    loss_is_done = F.l1_loss(is_done_hat[torch.arange(B), sub_task_idx.flatten()], is_done[torch.arange(B), sub_task_idx.flatten()])  # (B,)

    loss = 0.5 * loss_bezier_vector + loss_left_bottleneck + loss_right_bottleneck + 2.0 * loss_localbc_action + loss_is_done

    loss.backward()
    self.optimizer.step()
    self.model.eval()

    train_info = {
        "loss": loss.item(),
        "loss_left_bottleneck": loss_left_bottleneck.item(),
        "loss_right_bottleneck": loss_right_bottleneck.item(),
        "loss_bezier_vector": loss_bezier_vector.item(),
        "loss_localbc_action": loss_localbc_action.item(),
        "loss_is_done": loss_is_done.item(),
    }

    return train_info


def validate_fn(self, batch):
    obs, action_pose, gaze, is_pad, is_done, sub_task_idx, step, bottleneck_step = batch

    # To cuda
    for key in obs.keys():
        obs[key] = obs[key].to(self.device)
    action_pose = action_pose.to(self.device)
    gaze = gaze.to(self.device)
    is_pad = is_pad.to(self.device)
    is_done = is_done.to(self.device)

    self.model.eval()
    self.model.zero_grad()

    B, state_dim = obs["state"].shape
    num_arm = 2

    last_action = action_pose[torch.arange(B), torch.where(step > 0, step - 1, step)]
    with torch.no_grad():
        reaching_action, localbc_action, is_done_hat, bottleneck_hat, reaching_info, _ = self(obs, gaze, last_action)

    target_is_pad = torch.empty(localbc_action.shape[:2], dtype=torch.bool, device=self.device)
    target_action_pose = torch.empty_like(localbc_action)  # (B, localbc_action_chunk_size)
    for b in range(B):
        target_is_pad[b] = is_pad[b, step[b] : step[b] + self.model.localbc_action_chunk_size]
        target_action_pose[b] = action_pose[b, step[b] : step[b] + self.model.localbc_action_chunk_size]

    lr_idx = torch.BoolTensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])

    # bezier vector loss
    loss_bezier_vector = torch.tensor(0, dtype=torch.float, device=self.device)
    for b in range(B):
        for lr in range(2):
            if bottleneck_step[b, lr] < step[b] + 2:
                continue

            target_trajectory = action_pose[
                b, torch.where(step[b] > 0, step[b] - 1, step[b]) : bottleneck_step[b, lr], lr_idx[lr]
            ].cpu()  # (Ntraj, state_dim // 2)

            start_point = target_trajectory[0]  # (state_dim // 2,)
            goal_point = target_trajectory[-1]  # (state_dim // 2,)
            bezier_point = (start_point + goal_point) / 2  # (state_dim // 2,)

            bezier_points = torch.stack([start_point, bezier_point, goal_point]).T  # (state_dim // 2, 3)

            # bezier fitting
            t = torch.linspace(0, 1, len(target_trajectory) * 5)
            f = bezier_term(3, t)  # (3, 5 * Ntraj)
            for _ in range(5):  # 一回で収束しないことが多いので何回かイテレーション
                t_opt = (
                    ((bezier(bezier_points, t).T[..., None] - target_trajectory.T[None, ...]) ** 2).sum(1).argmin(0)
                )  # t_opt = argmin(pi-P(t)), shape: (Ntraj,)
                # print("convergence:", np.sum((bezier(bezier_points, t[t_opt]).T - target_trajectory) ** 2))
                if torch.sum(f[1, t_opt] * f[1, t_opt]) > 1e-12:
                    bezier_points[:, 1] = (
                        torch.sum(target_trajectory * f[1, t_opt, None], axis=0)
                        - bezier_points[:, 0] * torch.sum(f[0, t_opt] * f[1, t_opt])
                        - bezier_points[:, 2] * torch.sum(f[1, t_opt] * f[2, t_opt])
                    ) / torch.sum(f[1, t_opt] * f[1, t_opt])
                else:
                    bezier_points[:, 1] = (start_point + goal_point) / 2
            # t_opt = ((bezier(bezier_points, t).T[..., None] - target_trajectory.T[None, ...]) ** 2).sum(1).argmin(0)  # t_opt = argmin(pi-P(t)), shape: (Ntraj,)

            bezier_vector = bezier_points[:, 1] - (start_point + goal_point) / 2  # (state_dim // 2,)
            loss_bezier_vector += F.l1_loss(reaching_info["bezier_vector"][b, lr_idx[lr]], bezier_vector.to(self.device)) / B / 2

    # bottleneck loss
    bottleneck = torch.cat(
        [
            action_pose[torch.arange(B), bottleneck_step[:, 0], : state_dim // num_arm],
            action_pose[torch.arange(B), bottleneck_step[:, 1], state_dim // num_arm :],
        ],
        dim=1,
    )  # (B, state_dim)
    loss_left_bottleneck = F.l1_loss(bottleneck_hat[:, : state_dim // num_arm], bottleneck[:, : state_dim // num_arm])
    loss_right_bottleneck = F.l1_loss(bottleneck_hat[:, state_dim // num_arm :], bottleneck[:, state_dim // num_arm :])

    # localbc Loss
    before_bottleneck = (
        torch.cat(
            [
                torch.where(
                    (step < bottleneck_step[:, 0]).unsqueeze(-1), torch.ones(state_dim // num_arm), torch.zeros(state_dim // num_arm)
                ),  # (B, state_dim // num_arm)
                torch.where(
                    (step < bottleneck_step[:, 1]).unsqueeze(-1), torch.ones(state_dim // num_arm), torch.zeros(state_dim // num_arm)
                ),  # (B, state_dim // num_arm)
            ],
            dim=-1,
        )
        .to(torch.bool)
        .to(self.device)
    )  # (B, state_dim)
    if torch.all(before_bottleneck):
        loss_localbc_action = torch.tensor(0, dtype=torch.float, device=self.device)
    else:
        all_l1 = F.l1_loss(localbc_action, target_action_pose, reduction="none")  # (B, localbc_action_chunk_size, state_dim)
        loss_localbc_action = torch.sum(all_l1 * ~target_is_pad.unsqueeze(-1) * ~before_bottleneck.unsqueeze(1)) / torch.sum(
            torch.ones_like(all_l1) * ~target_is_pad.unsqueeze(-1) * ~before_bottleneck.unsqueeze(1)
        )  # (B, localbc_action_chunk_size, state_dim)

    # GazeTransition Loss
    loss_is_done = F.l1_loss(is_done_hat[torch.arange(B), sub_task_idx.flatten()], is_done[torch.arange(B), sub_task_idx.flatten()])  # (B,)

    loss = 0.5 * loss_bezier_vector + loss_left_bottleneck + loss_right_bottleneck + 2.0 * loss_localbc_action + loss_is_done

    self.model.train()

    val_info = {
        "loss": loss.item(),
        "loss_left_bottleneck": loss_left_bottleneck.item(),
        "loss_right_bottleneck": loss_right_bottleneck.item(),
        "loss_bezier_vector": loss_bezier_vector.item(),
        "loss_localbc_action": loss_localbc_action.item(),
        "loss_is_done": loss_is_done.item(),
    }

    print("sub_task_idx:", sub_task_idx[0])
    print(
        "reaching pred:\n",
        reaching_action[0, : self.model.reaching_action_chunk_size // 5 + 1],
        "\ntarget:\n",
        target_action_pose[0, step[0] : step[0] + self.model.reaching_action_chunk_size // 5 + 10],
    )
    print("bottleneck_hat:\n", bottleneck_hat[0], "\nbottleneck:\n", bottleneck[0])
    print("bottleneck step:", bottleneck_step[0], "step:", step[0])
    print(f"localbc action:\n{localbc_action[0][:5]},\n action data:\n{target_action_pose[0][:5]}")
    print(f"is_done_hat:\n {is_done_hat[0]}, \nis_done:\n {is_done[0]}")

    return loss.item(), val_info


if __name__ == "__main__":
    main()
