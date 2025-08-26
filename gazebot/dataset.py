import os
import sys
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as VF
import h5py
import hydra

from .utils import custom_array2string
from .train import state_action_metrics


def create_gaze_dataset(args):
    TRAIN_MAX_DEMO_NUM = int(args.expert.train_demos)
    TEST_MAX_DEMO_NUM = int(args.expert.test_demos)

    train_data_dir = [os.path.expanduser(train_dir) for train_dir in args.expert.train_path]
    test_data_dir = [os.path.expanduser(test_dir) for test_dir in args.expert.test_path]

    train_dataset = GazeDataset(train_data_dir, TRAIN_MAX_DEMO_NUM, args.expert.bgr)
    test_dataset = GazeDataset(test_data_dir, TEST_MAX_DEMO_NUM, args.expert.bgr)

    return train_dataset, test_dataset


def create_manipulation_dataset(args):
    TRAIN_MAX_DEMO_NUM = int(args.expert.train_demos)
    TEST_MAX_DEMO_NUM = int(args.expert.test_demos)
    train_data_dir = [os.path.expanduser(train_dir) for train_dir in args.expert.train_path]
    test_data_dir = [os.path.expanduser(test_dir) for test_dir in args.expert.test_path]

    try:
        state_mean, state_std, action_pose_mean, action_pose_std = state_action_metrics(args)
    except NotImplementedError as e:
        print("Current task_type has no corresponding metrics in state_action_metrics")
        print("Implement below:")
        measure_metrics(train_data_dir)
        measure_metrics(test_data_dir)
        raise e

    # Remove neck metrics
    state_mean = state_mean[:14]
    state_std = state_std[:14]
    action_pose_mean = action_pose_mean[:14]
    action_pose_std = action_pose_std[:14]

    train_dataset = ManipulationDataset(
        train_data_dir,
        TRAIN_MAX_DEMO_NUM,
        args.agent.name,
        args.device,
        args.expert.bgr,
        state_mean,
        state_std,
        action_pose_mean,
        action_pose_std,
        args.agent.num_queries,
        args.expert.num_sub_task,
        args.env.state_dim,
        args.env.image_dim,
        args.agent.bottleneck_thresh,
        args.agent.minimum_step_ratio,
    )
    test_dataset = ManipulationDataset(
        test_data_dir,
        TEST_MAX_DEMO_NUM,
        args.agent.name,
        args.device,
        args.expert.bgr,
        state_mean,
        state_std,
        action_pose_mean,
        action_pose_std,
        args.agent.num_queries,
        args.expert.num_sub_task,
        args.env.state_dim,
        args.env.image_dim,
        args.agent.bottleneck_thresh,
        args.agent.minimum_step_ratio,
    )

    return train_dataset, test_dataset


class GazeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_episode_num, is_bgr):
        self.data_dir = data_dir
        self.data_idxs = []

        self.max_episode_num = max_episode_num or 1e12

        self.is_bgr = is_bgr

        self.load_data()

    def __len__(self):
        return len(self.data_idxs)

    def load_data(self):
        if isinstance(self.data_dir, list):
            episode_files = sorted([os.path.join(d, f) for d in self.data_dir for f in os.listdir(d) if "h5" in f])
        else:
            episode_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if "h5" in f])
        assert len(episode_files) != 0

        valid_eps_count = 0
        for i, episode_file in enumerate(episode_files):
            if valid_eps_count == self.max_episode_num:
                break

            with h5py.File(episode_file, "r") as e:
                if "left_img" not in e or "gaze" not in e:
                    continue

                eps_steps = len(e["left_img"])
                if eps_steps < 2:
                    continue

                # change_steps: steps in which gaze transition is occurred
                if "change_steps" in e:
                    change_steps = e["change_steps"][1:]
                    print("change_steps:", change_steps)
                else:
                    print("[GazeDataset] change_steps is not found. Skipping episode...")
                    continue

                # Segment episode by change_steps
                init_step = 0
                for seg_idx, change_step in enumerate(change_steps):
                    for step in range(init_step, change_step):
                        data_idx = dict(file=episode_file, step=step, seg_idx=seg_idx)
                        self.data_idxs.append(data_idx)
                    init_step = change_step

                valid_eps_count += 1
                print(f"Load episode {i} ({episode_file}, {eps_steps} step)")
        print(f"Total data: {valid_eps_count} episodes, {len(self.data_idxs)} total steps")

    def __getitem__(self, index):
        file, step, seg_idx = self.data_idxs[index].values()

        with h5py.File(file, "r") as demo:
            # Image
            image = np.transpose(np.stack([demo["left_img"][step], demo["right_img"][step]]), (0, 3, 1, 2)) / 255.0  # (2, C, H, W)
            if self.is_bgr:
                image = image[:, [2, 1, 0]]  # BGR2RGB

            _, _, H, W = image.shape

            # Gaze
            gaze = np.array(demo["gaze"][step], dtype=np.float64).reshape(2, 2)  # gaze position in pixel coord
            gaze = np.clip(gaze, [0, 0], [W - 1, H - 1])

        # Normalize gaze ([0, 0] -> [0, 0], [W, H] -> [1, 1])
        gaze_norm = np.zeros_like(gaze)
        gaze_norm[:, 0] = gaze[:, 0] / W
        gaze_norm[:, 1] = gaze[:, 1] / H

        # To tensor
        image = torch.as_tensor(image, dtype=torch.float)  # (2, C, H, W)
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)
        image = color_jitter(image)
        image = VF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        gaze = torch.as_tensor(gaze_norm, dtype=torch.float)  # (2, 2)

        seg_idx = torch.tensor([seg_idx], dtype=torch.long)  # (1,)

        return image, gaze, seg_idx


class ManipulationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        max_episode_num,
        agent_name,
        device,
        is_bgr,
        state_mean,
        state_std,
        action_pose_mean,
        action_pose_std,
        action_chunk_size,
        num_sub_task,
        state_dim,
        image_dim,
        bottleneck_thresh,
        minimum_step_ratio,
    ):
        self.agent_name = agent_name

        self.state_dim = state_dim  # num_arm * (ARM_DOF + GRIPPER_DOF)  # left & right of arm + gripper
        self.image_size = image_dim[::-1][:2]  # (W, H)

        self.device = torch.device(device)

        self.data_dir = data_dir
        self.max_episode_num = max_episode_num or 1e12
        self.data_idxs = []

        measure_metrics(self.data_dir)

        self.is_bgr = is_bgr

        assert len(state_mean) == len(state_std) == len(action_pose_mean) == len(action_pose_std) == self.state_dim
        self.state_mean = np.array(state_mean)
        self.state_std = np.array(state_std)
        self.action_pose_mean = np.array(action_pose_mean)
        self.action_pose_std = np.array(action_pose_std)

        self.max_motion_step = 300
        self.action_chunk_size = action_chunk_size
        self.num_sub_task = num_sub_task
        self.bottleneck_thresh = bottleneck_thresh
        self.minimum_step_ratio = minimum_step_ratio

        self.load_data()

    def __len__(self):
        return len(self.data_idxs)

    def load_data(self):
        if isinstance(self.data_dir, list):
            episode_files = sorted([os.path.join(d, f) for d in self.data_dir for f in os.listdir(d) if "h5" in f])
        else:
            episode_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if "h5" in f])
        assert len(episode_files) != 0

        valid_eps_count = 0
        for eps_idx, episode_file in enumerate(episode_files):
            if valid_eps_count // self.num_sub_task > self.max_episode_num - 1:
                break

            print(f"\n\nLoad episode {eps_idx}: {episode_file}\n")

            with h5py.File(episode_file, "r") as e:
                if "right_state" not in e:
                    break
                eps_steps = len(e["right_state"])
                if eps_steps < 2:
                    break

                # change_steps: steps in which gaze transition is occurred
                if "change_steps" in e:
                    change_steps = e["change_steps"][1:]
                    # NOTE Remove gaze transition after the task is completed
                    if len(change_steps) == self.num_sub_task + 1:
                        change_steps = change_steps[:-1]
                    print("change_steps:", change_steps)
                else:
                    print("[ManipulationDataset] change_steps is not found. Skipping episode...")
                    continue

                # episodeをchange_stepで分割
                init_step = 0
                for sub_task_idx, change_step in enumerate(change_steps):
                    # Add data to dataset
                    for step in range(init_step + 1, change_step - 1):
                        data_idx = dict(file=episode_file, step=step, init_step=init_step, change_step=change_step, sub_task_idx=sub_task_idx)
                        self.data_idxs.append(data_idx)
                    init_step = change_step
                    valid_eps_count += 1
                    sys.stdout.flush()
        print(f"\nDataset Size: {len(self.data_idxs)} (Episode Num: {valid_eps_count // self.num_sub_task})\n")

    def __getitem__(self, index):
        file, step, init_step, change_step, sub_task_idx = self.data_idxs[index].values()

        assert self.max_motion_step > change_step - 1 - init_step

        with h5py.File(file, "r") as e:
            # Detect bottleneck
            bottleneck_step_lr = []
            for _lr in range(2):
                if "localbc_loss" in e:
                    localbc_loss = np.array(e["localbc_loss"][init_step:change_step, _lr])
                    # Filter
                    filtered_localbc_loss = np.array(
                        [np.mean(localbc_loss[max(0, i - 4) : min(i + 5, len(localbc_loss))]) for i in range(len(localbc_loss))]
                    )
                    # normalize scale
                    normalized_filtered_localbc_loss = filtered_localbc_loss * 1 / np.median(filtered_localbc_loss)

                    minimum_steps = int((change_step - init_step) * self.minimum_step_ratio)
                    bottleneck_step = init_step + 1
                    max_score = -np.inf
                    for _step in range(init_step + 1, change_step - 2):
                        score = np.sum(normalized_filtered_localbc_loss[: _step - init_step] > self.bottleneck_thresh) + np.sum(
                            normalized_filtered_localbc_loss[_step - init_step :] < self.bottleneck_thresh
                        )
                        if score > max_score:
                            max_score = score
                            bottleneck_step = _step
                    if change_step - bottleneck_step < minimum_steps:
                        bottleneck_step = init_step + 1
                else:
                    bottleneck_step = init_step + 1
                bottleneck_step_lr.append(bottleneck_step)

            # Obs
            obs = {}
            obs["state"] = np.concatenate([e["left_f_state"][step], e["right_f_state"][step]], axis=0)  # (state_dim,)
            obs["state"] = (obs["state"] - self.state_mean) / self.state_std
            obs["image"] = np.transpose(e["left_img"][step], (2, 0, 1)) / 255.0  # [0, 1], (3, H, W)
            obs["depth"] = np.transpose(e["depth_img"][step], (2, 0, 1)).astype(np.float32)  # mm, [0, inf), (1, H, W)

            if self.is_bgr:
                obs["image"] = np.ascontiguousarray(obs["image"][[2, 1, 0]])  # BGR2RGB

            # Action pose
            action_pose_left_arm = np.array(e["left_f_state"][init_step + 1 : change_step, :6])  # (N, ARM_DOF)
            action_pose_left_gripper = np.array(
                e["left_f_hstate"][init_step : change_step - 1, [6]]
            )  # (N, GRIPPER_DOF), Use hstate as action for pseudo force control of the gripper
            action_pose_right_arm = np.array(e["right_f_state"][init_step + 1 : change_step, :6])  # (N, ARM_DOF)
            action_pose_right_gripper = np.array(
                e["right_f_hstate"][init_step : change_step - 1, [6]]
            )  # (N, GRIPPER_DOF), Use hstate as action for pseudo force control of the gripper
            action_pose = np.concatenate(
                (action_pose_left_arm, action_pose_left_gripper, action_pose_right_arm, action_pose_right_gripper), axis=1
            )  # (N, state_dim)
            action_pose = (action_pose - self.action_pose_mean) / self.action_pose_std  # (N, state_dim)

            # Gaze
            gaze = np.array(e["gaze"][step])  # (4,), [W, H]
            gaze = np.clip(gaze, [0, 0, 0, 0], [self.image_size[0] - 1, self.image_size[1] - 1, self.image_size[0] - 1, self.image_size[1] - 1])

        # Gaze Transition
        is_done = np.concatenate(
            (
                np.ones(sub_task_idx),
                np.array([max(0, (step - init_step) / (change_step - 1 - init_step))]),
                np.zeros(self.num_sub_task - 1 - sub_task_idx),
            )
        )  # progress for the gaze transition (1: sub_task_idx += 1)

        # Info
        bottleneck_step = np.array(bottleneck_step_lr)  # (2,)
        sub_task_idx = np.array([sub_task_idx])

        # Convert to tensor (load on cpu initially)
        obs["state"] = torch.as_tensor(obs["state"], dtype=torch.float)
        obs["image"] = torch.as_tensor(obs["image"], dtype=torch.float)
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)
        obs["image"] = color_jitter(obs["image"])
        obs["image"] = VF.normalize(obs["image"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if obs["image"].shape[1:] != torch.Size(self.image_size[::-1]):
            obs["image"] = VF.resize(obs["image"], size=self.image_size[::-1])
        obs["depth"] = torch.as_tensor(obs["depth"], dtype=torch.float)
        if obs["depth"].shape[1:] != torch.Size(self.image_size[::-1]):
            obs["depth"] = VF.resize(obs["depth"], size=self.image_size[::-1])

        action_pose = torch.as_tensor(action_pose, dtype=torch.float)

        gaze = torch.as_tensor(gaze, dtype=torch.long)

        is_done = torch.as_tensor(is_done, dtype=torch.float)

        step = torch.as_tensor(step - init_step, dtype=torch.long)
        bottleneck_step = torch.as_tensor(bottleneck_step - init_step, dtype=torch.long)

        sub_task_idx = torch.as_tensor(sub_task_idx, dtype=torch.long)

        # Padding
        padding_size = self.max_motion_step - (change_step - 1 - init_step)
        is_pad = torch.cat((torch.zeros(change_step - 1 - init_step), torch.ones(padding_size + self.action_chunk_size))).to(
            torch.bool
        )  # (max_motion_sep + num_queries,)
        action_pose = torch.cat(
            (action_pose, torch.zeros(padding_size + self.action_chunk_size, self.state_dim, dtype=torch.float))
        )  # (max_motion_step + num_queries, state_dim)

        return obs, action_pose, gaze, is_pad, is_done, sub_task_idx, step, bottleneck_step


def measure_metrics(data_dir):
    if isinstance(data_dir, list):
        episode_files = sorted([os.path.join(d, f) for d in data_dir for f in os.listdir(d) if "h5" in f])
    else:
        episode_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if "h5" in f])
    assert len(episode_files) != 0

    state = []
    action_pose = []
    init_pose = []
    for episode_file in episode_files:
        with h5py.File(episode_file, "r") as e:
            if "right_state" not in e:
                break
            eps_steps = len(e["right_state"])
            if eps_steps < 2:
                break

            state.append(np.concatenate([e["left_f_state"], e["right_f_state"]], axis=1))  # (step, 14)
            action_pose.append(
                np.concatenate(
                    [
                        np.array(e["left_f_state"][1:, :6]),
                        np.array(e["left_f_hstate"][:-1, [6]]),
                        np.array(e["right_f_state"][1:, :6]),
                        np.array(e["right_f_hstate"][:-1, [6]]),
                    ],
                    axis=1,
                )
            )  # (step, 14)

            init_pose.append(np.concatenate([e["left_state"][0], e["right_state"][0]]))  # (14,)

    state = np.concatenate(state)  # (total_steps, 14)
    action_pose = np.concatenate(action_pose)  # (total_steps, 14)
    init_pose = np.array(init_pose)  # (num_episodes, 14)

    # metrics
    state_mean = state.mean(axis=0)
    action_pose_mean = action_pose.mean(axis=0)
    state_std = state.std(axis=0)
    action_pose_std = action_pose.std(axis=0)

    mean_init_pose = init_pose.mean(0)

    print("\n[measured metrics]")
    print("state_mean =")
    print(custom_array2string(state_mean, 7, formatter={"float_kind": lambda x: "{: .7f}".format(x)}))
    print("state_std =")
    print(custom_array2string(state_std, 7, formatter={"float_kind": lambda x: "{: .7f}".format(x)}))
    print("action_pose_mean =")
    print(custom_array2string(action_pose_mean, 7, formatter={"float_kind": lambda x: "{: .7f}".format(x)}))
    print("action_pose_std =")
    print(custom_array2string(action_pose_std, 7, formatter={"float_kind": lambda x: "{: .7f}".format(x)}))
    print("mean_init_pose =")
    print(custom_array2string(mean_init_pose, 7, formatter={"float_kind": lambda x: "{: .7f}".format(x)}))
    print("\n")

    return state_mean, state_std, action_pose_mean, action_pose_std, mean_init_pose
