import time
from typing import List
from dataclasses import dataclass, field
from itertools import count
from collections import deque
import numpy as np
import torch
import cv2
import hydra
from omegaconf import DictConfig

from gazebot.train import get_args
from gazebot.test import make_env, make_agent
from gazebot.utils import PointCloudViewer, array2image, image2opencv, tensor2opencv

from tongsystem.device import KeyboardManager


@dataclass
class StateMachine:
    # manipulation progress
    sub_task_idx: int = 0
    progress: torch.Tensor = None  # (1, N)
    progress_count: int = 0

    # action
    is_reaching: List[bool] = field(default_factory=lambda: [True, True])  # start with ReachingBottleneck at first
    action_buffer: List[deque] = None
    last_action: torch.Tensor = None  # (1, state_dim)
    reaching_traj: torch.Tensor = None  # (reaching_action_chunk_size, state_dim)
    reaching_step: int = 0
    last_bottleneck: torch.Tensor = None  # (1, state_dim)
    last_gaze: torch.Tensor = None  # (1, 4)
    count_bottleneck: int = 0


@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    args = get_args(cfg, random_seed=False)
    device = torch.device(args.device)

    env, metrics = make_env(args)

    gaze_agent, manipulation_agent = make_agent(args, ignore_error=True)

    # Point cloud visualizer
    pcd_viewer = PointCloudViewer()

    keyboard = KeyboardManager()

    EPISODE_STEPS = args.env.eps_steps
    NUM_SUB_TASKS = args.expert.num_sub_task

    # パラメータ
    PROGRESS_THRESH = args.agent.progress_thresh
    GAZE_REPLACEMENT_THRESH = args.agent.gaze_replacement_thresh  # px
    CONVERGENCE_THRESH = args.agent.convergence_thresh

    abort_flag = False
    success_count = 0

    print("\n[Info] Loop start")
    print("------------------------\n")
    for epoch in count():
        obs = env.reset(random_init=True)

        # parameters of state machine
        sm = StateMachine(
            action_buffer=[deque(maxlen=args.agent.num_queries) for _ in range(2)],
            last_action=torch.as_tensor(
                ((obs["state"].detach().cpu().numpy() * metrics["state_std"] + metrics["state_mean"]) - metrics["action_pose_mean"])
                / metrics["action_pose_std"],
                dtype=torch.float,
                device=device,
            ),
        )

        print(f"[Info] ReachingBottleneck start! (sub-task: {sm.sub_task_idx})")
        for episode_step in range(EPISODE_STEPS):
            # Keyboard interface
            pressed_key = keyboard.get_key()
            if pressed_key == "a":
                abort_flag = True
                break
            elif pressed_key == "b":
                time.sleep(1.0)
                break

            # Gaze Transition
            if sm.progress_count > 5 or pressed_key == "c":
                # Change gaze target and start ReachingBottleneck
                if sm.sub_task_idx < NUM_SUB_TASKS - 1:
                    print("[Info] gaze transition")
                    sm.sub_task_idx += 1
                    start_reaching_bottleneck(sm)
                sm.progress_count = 0

            # Gaze Inference
            gaze = get_gaze(gaze_agent, obs, sm.sub_task_idx)  # (1, 4)

            # Gaze replacement (e.g. when object is moved by others) -> start ReachingBottleneck
            if sm.last_gaze is not None and torch.linalg.norm((gaze[0, :2] - sm.last_gaze[0, :2]).to(torch.float)) > GAZE_REPLACEMENT_THRESH:
                start_reaching_bottleneck(sm)

            # (optional) Update bottleneck in the first few steps of ReachingBottleneck for stability
            if sm.count_bottleneck != -1:
                print("[Info] bottleneck updated")
                sm.count_bottleneck += 1
                sm.last_bottleneck = None  # estimate new bottleneck in this step
                if sm.count_bottleneck == 5:
                    sm.count_bottleneck = -1  # stop updating

            # Actions & Progress Inference
            action, vis = action_process(manipulation_agent, obs, gaze, sm, metrics, CONVERGENCE_THRESH)

            # Step Env
            obs, _, _, _ = env.step(action, gaze[0].cpu().numpy())

            # Update StateMachine
            if sm.progress[0, sm.sub_task_idx] > PROGRESS_THRESH - 0.05:
                print(f"[Info] progress (sub_task_idx={sm.sub_task_idx}):", sm.progress[0, sm.sub_task_idx].item())
            if sm.progress[0, sm.sub_task_idx] > PROGRESS_THRESH:
                sm.progress_count += 1
            sm.last_action = torch.as_tensor(
                (action - metrics["action_pose_mean"]) / metrics["action_pose_std"], dtype=torch.float, device=device
            ).unsqueeze(0)  # (1, state_dim)
            sm.last_gaze = gaze

            # Visualize
            pcd_viewer.render(vis[0])

        # env.save_log()
        env.reset_log()
        print(f"[Info] episode: {epoch}, steps: {episode_step}")

        if abort_flag:
            break

    pcd_viewer.close()
    cv2.destroyAllWindows()
    env.close()
    print(f"[Info] success rate = {success_count / max(epoch, 1)}")
    print("[Info] Finished!")


def start_reaching_bottleneck(sm: StateMachine):
    print(f"[Info] ReachingBottleneck start! (sub-task: {sm.sub_task_idx})")
    sm.count_bottleneck = 0
    sm.is_reaching = [True, True]
    sm.action_buffer[0].clear()
    sm.action_buffer[1].clear()


def action_process(manipulation_agent, obs: dict, gaze: torch.Tensor, sm: StateMachine, metrics: dict, convergence_thresh: float) -> np.ndarray:
    with torch.no_grad():
        reaching_action, localbc_action, sm.progress, bottleneck, _, vis = manipulation_agent(
            obs, gaze, sm.last_action, sm.last_bottleneck, is_inference=True
        )
    if sm.last_bottleneck is None:
        sm.reaching_traj = reaching_action
        sm.reaching_step = 0
        sm.last_bottleneck = bottleneck

    lr_action = []
    for lr in range(2):
        l_idx = np.arange(7, dtype=np.int64)
        r_idx = np.arange(7, 14, dtype=np.int64)
        lr_idx = l_idx if lr == 0 else r_idx  # (state_dim // 2,)

        # convg_bottleneck: bottleneck is reached or not
        bottleneck = sm.last_bottleneck.detach().cpu().numpy()[0, lr_idx][:7]
        current = sm.last_action.detach().cpu().numpy()[0, lr_idx][:7]
        is_bottleneck = np.linalg.norm(bottleneck - current) < convergence_thresh  # True/False

        # Action Gating
        if sm.is_reaching[lr] and is_bottleneck:
            # switching process (reaching -> localbc)
            print(f"[Info] ({'left' if lr == 0 else 'right'}) LocalBC start! (sub-task: {sm.sub_task_idx})")
            sm.action_buffer[lr].clear()
            sm.is_reaching[lr] = False

        # Action
        if sm.is_reaching[lr]:
            action = sm.reaching_traj[0, sm.reaching_step, lr_idx].detach().cpu().numpy()  # (state_dim // 2,)
            action = action * np.array(metrics["action_pose_std"])[lr_idx] + np.array(metrics["action_pose_mean"])[lr_idx]
            sm.reaching_step = min(sm.reaching_step + 1, sm.reaching_traj.shape[1] - 1)
        else:
            action_chunk = reaching_action if sm.is_reaching[lr] else localbc_action
            action_chunk = action_chunk.detach().cpu().numpy()[0][:, lr_idx]  # (1, num_queries, state_dim) -> (num_queries, state_dim // 2)
            action_chunk = (
                action_chunk * np.array(metrics["action_pose_std"])[lr_idx] + np.array(metrics["action_pose_mean"])[lr_idx]
            )  # action in diff-state space

            sm.action_buffer[lr].append(action_chunk)
            action = temporal_agg(sm.action_buffer[lr])  # (state_dim // 2,)

        lr_action.append(action)
    lr_action = np.concatenate(lr_action)  # (state_dim,)

    return lr_action, vis


def get_gaze(gaze_agent, obs: dict, sub_task_idx_: int) -> torch.Tensor:
    image = torch.stack(torch.split(obs["image"], 3, dim=1), dim=1)  # (1, 2, C, H, W)

    sub_task_idx = np.array([sub_task_idx_]).reshape(1, -1)  # (1, 1)
    sub_task_idx = torch.as_tensor(sub_task_idx, dtype=torch.long)  # (1, 1)

    with torch.inference_mode():
        gaze = gaze_agent(image, sub_task_idx, denormalize=True)[0].reshape(1, -1)  # (1, 4)
        gaze = torch.round(gaze).to(torch.int)

    # Visualize gaze
    predict_gaze = np.round(gaze[0].detach().cpu().numpy()).astype(np.int64)
    left_im = tensor2opencv(obs["image"][0, :3])
    right_im = tensor2opencv(obs["image"][0, 3:])
    left_im = cv2.circle(left_im, (predict_gaze[0], predict_gaze[1]), 20, (255, 255, 255), 5)
    right_im = cv2.circle(right_im, (predict_gaze[2], predict_gaze[3]), 20, (255, 255, 255), 5)
    im = np.concatenate([left_im, right_im], axis=1)  # (H, 2 * W, C)
    im = image2opencv(array2image(im[:, :, [2, 1, 0]]).reduce(4))
    cv2.imshow("gaze", im)
    cv2.waitKey(1)

    return gaze


def temporal_agg(action_buffer: deque) -> np.ndarray:
    # NOTE if you use temporal_agg, action should not be state_diff but state (due to the accumulation of diff through time)
    action = 0
    total_weight = 0
    m = 0.01
    for i in range(len(action_buffer)):
        weight = np.exp(-m * (len(action_buffer) - i - 1))
        action += weight * action_buffer[-i - 1][i]
        total_weight += weight
    action = action / total_weight
    return action


if __name__ == "__main__":
    main()
