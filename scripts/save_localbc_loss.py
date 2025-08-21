import os
import sys
import numpy as np
import torch
import h5py
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torchvision.transforms.functional as VF

from gazebot.train import get_args, load_agent
from gazebot.agent import create_manipulation_agent


@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Get config
    args = get_args(cfg)

    # Save localbc loss
    args.agent.enc_layers = 1  # NOTE: Use small transformer for localbc
    args.agent.dec_layers = 1
    args.agent.dim_feedforward = 1024
    agent = create_manipulation_agent(args)

    localbc_path = hydra.utils.to_absolute_path("path/to/localbc/model/dir")  # NOTE Specify trained localbc path
    localbc_agent = load_agent(agent, localbc_path, args, ignore_error=False)

    save_localbc_loss(localbc_agent, args)

    print("COMPLETE!")


def save_localbc_loss(agent, args):
    def denormalize(action):
        assert action.shape[-1] == len(agent.model.action_pose_mean) == len(agent.model.action_pose_std)
        return action * agent.model.action_pose_std + agent.model.action_pose_mean

    agent.eval()

    LOSS_WEIGHTS = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0]

    data_dir = [hydra.utils.to_absolute_path(os.path.expanduser(dir_path)) for dir_path in args.expert.train_path + args.expert.test_path]
    all_episodes = np.array([os.path.join(d, f) for d in data_dir for f in os.listdir(d) if "h5" in f])

    for eps_idx, episode_file in enumerate(all_episodes):
        print(f"\n\nLoad episode {eps_idx}: {episode_file}\n")

        with h5py.File(episode_file, "r+") as e:
            eps_steps = len(e["left_f_state"])
            if "localbc_loss" not in e:
                e.create_dataset("localbc_loss", data=np.inf * np.ones((eps_steps, 2)))

            if "change_steps" in e:
                change_steps = e["change_steps"][1:]
                print("change_steps:", change_steps)
            else:
                print("change_steps is not found. Skipping episode...")
                continue

            localbc_losses = []

            init_step = 0
            for change_step in change_steps:
                print("init_step, change_step:", init_step, change_step)

                # Action
                action_left_arm = np.array(e["left_f_state"][init_step + 1 : change_step, :6])  # (N, ARM_DOF)
                action_left_gripper = np.array(
                    e["left_f_hstate"][init_step : change_step - 1, [6]]
                )  # (N, GRIPPER_DOF), Use hstate as action for pseudo force control of the gripper
                action_right_arm = np.array(e["right_f_state"][init_step + 1 : change_step, :6])  # (N, ARM_DOF)
                action_right_gripper = np.array(
                    e["right_f_hstate"][init_step : change_step - 1, [6]]
                )  # (N, GRIPPER_DOF), Use hstate as action for pseudo force control of the gripper
                action_pose_seq = np.concatenate(
                    (action_left_arm, action_left_gripper, action_right_arm, action_right_gripper), axis=1
                )  # (N, state_dim)

                for step in range(init_step, change_step - 1):
                    ### Measure bottleneck score of sampled bottleneck step ###
                    # Obs
                    obs = {}
                    obs["state"] = np.concatenate([e["left_f_state"][step], e["right_f_state"][step]], axis=0)  # (state_dim,)
                    obs["image"] = np.transpose(np.array(e["left_img"][step]), (2, 0, 1)) / 255.0  # [0, 1], (C, H, W)
                    obs["depth"] = np.transpose(e["depth_img"][step], (2, 0, 1)).astype(np.float32)  # mm, [0, inf), (1, H, W)

                    if args.expert.bgr:
                        obs["image"] = obs["image"][[2, 1, 0]]  # BGR2RGB

                    obs["state"] = torch.as_tensor(obs["state"], dtype=torch.float, device=agent.device)
                    obs["state"] = (obs["state"] - agent.model.state_mean) / agent.model.state_std
                    obs["image"] = torch.as_tensor(obs["image"], dtype=torch.float, device=agent.device)
                    obs["image"] = VF.normalize(obs["image"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    if obs["image"].shape[1:] != torch.Size(args.env.image_dim[1:]):
                        obs["image"] = VF.resize(obs["image"], size=args.env.image_dim[1:])
                    obs["depth"] = torch.as_tensor(obs["depth"], dtype=torch.float, device=agent.device)
                    if obs["depth"].shape[1:] != torch.Size(args.env.image_dim[1:]):
                        obs["depth"] = VF.resize(obs["depth"], size=args.env.image_dim[1:])

                    _, H, W = obs["image"].shape

                    # Gaze
                    gaze = np.array(e["gaze"][step])  # (4,), [W, H]
                    gaze = np.clip(gaze, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])

                    gaze = torch.as_tensor(gaze, dtype=torch.long, device=agent.device)

                    # Batch
                    obs["state"] = obs["state"].unsqueeze(0)
                    obs["image"] = obs["image"].unsqueeze(0)
                    obs["depth"] = obs["depth"].unsqueeze(0)
                    gaze = gaze.unsqueeze(0)

                    step = step - init_step

                    # LocalBC
                    with torch.no_grad():
                        action_pose_hat, _ = agent.local_bc(obs, gaze)

                    action_pose = action_pose_seq[step : step + agent.model.localbc_action_chunk_size]  # (Ntraj, state_dim)
                    action_pose_hat = denormalize(action_pose_hat)[0].detach().cpu().numpy()[: len(action_pose)]  # (Ntraj, state_dim)

                    left_loss_action = np.average((action_pose_hat[0, :7] - action_pose[0, :7]) ** 2, weights=np.array(LOSS_WEIGHTS) ** 2)
                    right_loss_action = np.average((action_pose_hat[0, 7:] - action_pose[0, 7:]) ** 2, weights=np.array(LOSS_WEIGHTS) ** 2)

                    # Log history for visualize
                    localbc_losses.append([left_loss_action, right_loss_action])
                    sys.stdout.flush()
                localbc_losses.append([0, 0])  # for step = chage_step - 1

                init_step = change_step

            e["localbc_loss"][: len(localbc_losses)] = np.array(localbc_losses)  # (N, 2)
            print("localbc_loss:\n", np.array(e["localbc_loss"]))

            plt.plot(localbc_losses, label="localbc")
            plt.legend()
            plt.savefig("loss.png")
            plt.clf()
            plt.close()


if __name__ == "__main__":
    main()
