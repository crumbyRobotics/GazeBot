import datetime
import os
import sys
import random
import time
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from .utils import AverageMeter, AverageMeterDict


def get_args(cfg: DictConfig, random_seed=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # in order to remove warning

    # remove py.warnings
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")

    cfg.hydra_base_dir = os.getcwd()

    # Serverの空いているGPUを指定するとよい
    cfg.device = cfg.cuda_device if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))

    if not random_seed:
        # Set seeds
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    if device.type == "cuda" and torch.cuda.is_available() and cfg.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(device)  # change default device (e.g. .cuda() or torch.cuda.IntTensor(x))

    return cfg


def state_action_metrics(args):
    if args.env.name == "tongsim":
        if args.expert.task_type == "PileBox":
            # fmt: off
            state_mean = [-0.6068457, 0.3021088,-0.3056861, 3.0657687,-0.0642200,-0.7288149, 0.4018294,
                          0.4587743, 0.2815206, 0.2600906, 3.0862043,-0.2168547, 0.7276848,-0.5235988]
            state_std = [0.0479745, 0.0433459, 0.1536488, 0.0003788, 0.0027617, 0.0006637, 0.1149221,
                         0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000]
            action_pose_mean = [-0.6078158, 0.3023462,-0.3084407, 3.0657675,-0.0642120,-0.7288125, 0.3473460,
                                0.4587743, 0.2815206, 0.2600906, 3.0862043,-0.2168547, 0.7276848,-0.5235988]
            action_pose_std = [0.0463780, 0.0433578, 0.1497415, 0.0003796, 0.0027676, 0.0006649, 0.1651571,
                               0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000]
            # fmt: on

        else:
            raise NotImplementedError

        state_std[:3] = np.clip(state_std[:3], 0.03, None)
        state_std[7:10] = np.clip(state_std[7:10], 0.03, None)
        state_std[3:7] = np.clip(state_std[3:7], 0.01, None)
        state_std[10:14] = np.clip(state_std[10:14], 0.01, None)
        state_std[6] = np.clip(state_std[6], 0.1, None)
        state_std[13] = np.clip(state_std[13], 0.1, None)

        action_pose_std[:3] = np.clip(action_pose_std[:3], 0.03, None)
        action_pose_std[7:10] = np.clip(action_pose_std[7:10], 0.03, None)
        action_pose_std[3:7] = np.clip(action_pose_std[3:7], 0.01, None)
        action_pose_std[10:14] = np.clip(action_pose_std[10:14], 0.01, None)
        action_pose_std[6] = np.clip(action_pose_std[6], 0.1, None)
        action_pose_std[13] = np.clip(action_pose_std[13], 0.1, None)

        if args.agent.name == "manipulation":
            # General metrics
            state_std = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.15, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.15]  # NOTE better performance?

    elif args.env.name == "tong":
        # NOTE gripper angle of real robot is not radian but degree (it is not problematic due to the normalization)
        if args.expert.task_type == "PenInCup":
            # fmt: off
            state_mean = [-0.6352632, 0.3386952,-0.1137690, 2.2742393, 0.1969251,-0.6800710, 12.0330992,
                          0.4862332, 0.1920269, 0.3245165, 2.8493283,-0.2797547,-0.2163827, 0.4157381]
            state_std = [0.0508097, 0.0572489, 0.1680862, 0.7778065, 0.5383984, 0.3051235, 8.4543200,
                         0.0134045, 0.0165333, 0.0259728, 0.1133326, 0.0414804, 0.0742406, 0.0845590]
            action_pose_mean = [-0.6353450, 0.3386310,-0.1149432, 2.2722950, 0.1957269,-0.6798583, 10.5215597,
                                0.4862323, 0.1920251, 0.3245153, 2.8493230,-0.2797512,-0.2163779, 1.3503994]
            action_pose_std = [0.0508135, 0.0573137, 0.1672879, 0.7786169, 0.5389619, 0.3056446, 10.0603971,
                               0.0134024, 0.0165314, 0.0259750, 0.1133290, 0.0414784, 0.0742383, 0.0248943]
            # fmt: on

        else:
            raise NotImplementedError

        state_std[:3] = np.clip(state_std[:3], 0.03, None)
        state_std[7:10] = np.clip(state_std[7:10], 0.03, None)
        state_std[3:7] = np.clip(state_std[3:7], 0.01, None)
        state_std[10:14] = np.clip(state_std[10:14], 0.01, None)
        state_std[6] = np.clip(state_std[6], 4, None)
        state_std[13] = np.clip(state_std[13], 4, None)

        action_pose_std[:3] = np.clip(action_pose_std[:3], 0.03, None)
        action_pose_std[7:10] = np.clip(action_pose_std[7:10], 0.03, None)
        action_pose_std[3:7] = np.clip(action_pose_std[3:7], 0.05, None)
        action_pose_std[10:14] = np.clip(action_pose_std[10:14], 0.05, None)
        action_pose_std[6] = np.clip(action_pose_std[6], 4, None)
        action_pose_std[13] = np.clip(action_pose_std[13], 4, None)

        if args.agent.name == "manipulation":
            # General metrics
            # fmt: off
            state_std = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 8.594367, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 8.594367]  # NOTE better performance? <- state_stdの0.1のミスのせいだけかも
            # fmt: on

    else:
        raise NotImplementedError

    return state_mean, state_std, action_pose_mean, action_pose_std


def train(agent, train_dataloader, val_dataloader, args):
    if args.wandb:
        wandb.init(
            project=args.project_name,
            name=str(args.agent.name) + datetime.datetime.fromtimestamp(time.time()).strftime("_%Y-%m-%d_%H-%M-%S"),
            config=dict(args),
        )

    MAX_STEP = int(args.train.learn_steps)
    MAX_EPOCH = int(args.train.max_epoch)

    # Train agent with expert demos
    learn_steps = 0
    best_step = 0
    min_val_score = np.inf
    timestamp = time.perf_counter()
    for epoch in range(MAX_EPOCH):
        print(f"~~~~~ epoch {epoch} ~~~~~")

        # Validate agent
        if epoch % args.val_interval == 0:
            print("[Validation]")
            val_score = AverageMeter()
            val_info = AverageMeterDict()
            for val_idx, val_batch in enumerate(val_dataloader):
                _val_score, _val_info = agent.validate_fn(val_batch)
                val_score.update(_val_score)
                val_info.update(_val_info)
                print(f"val info ({val_idx}):\n{_val_info}\n")
                if args.val_niter and val_idx == args.val_niter - 1:
                    break

            if val_score.avg < min_val_score:
                min_val_score = val_score.avg
                best_step = learn_steps
                save_agent(agent, learn_steps, args, "results", best_step=True)

            if args.wandb:
                log_dict = {}
                for k, v in val_info.items():
                    log_dict["val/" + k] = v.avg
                wandb.log(log_dict, step=learn_steps)

        # Train agent
        perf_time = AverageMeter()
        for index, batch in enumerate(train_dataloader):
            if learn_steps > MAX_STEP:
                break
            learn_steps += 1

            train_info = agent.update(batch)

            if args.wandb:
                log_dict = {}
                for k, v in train_info.items():
                    log_dict["train/" + k] = v
                wandb.log(log_dict, step=learn_steps)

            # Log
            if learn_steps % args.log_interval == 0:
                print("[Log]")
                print(f"learn_step: {learn_steps}")
                print(f"train info:\n{train_info}\n")
                if args.wandb:
                    wandb.log({"train/performance_time_per_batch": perf_time.avg}, step=learn_steps)

            save_agent(agent, learn_steps, args, "results")

            perf_time.update(time.perf_counter() - timestamp)
            timestamp = time.perf_counter()
            sys.stdout.flush()

        if args.wandb:
            wandb.log({"train/epoch": epoch}, step=learn_steps)

        if learn_steps > MAX_STEP:
            print("[Max learn steps exceeded]")
            break

    print(f"Finished! (best_step = {best_step})")
    if args.wandb:
        wandb.finish()

    return agent, best_step


def save_agent(agent, steps, args, output_dir="results", best_step=False):
    if steps % args.save_interval == 0:
        os.makedirs(output_dir, exist_ok=True)
        agent.save(f"{output_dir}/{args.agent.name}_{args.env.name}_{args.expert.task_type}")

    if hasattr(args.save_step_interval, "__iter__"):
        if steps in args.save_step_interval:
            os.makedirs(f"{output_dir}/{steps}step", exist_ok=True)
            agent.save(f"{output_dir}/{steps}step/{args.agent.name}_{args.env.name}_{args.expert.task_type}")
    else:
        if steps % args.save_step_interval == 0:
            os.makedirs(f"{output_dir}/{steps}step", exist_ok=True)
            agent.save(f"{output_dir}/{steps}step/{args.agent.name}_{args.env.name}_{args.expert.task_type}")

    if best_step:
        os.makedirs(f"{output_dir}/best_step", exist_ok=True)
        agent.save(f"{output_dir}/best_step/{args.agent.name}_{args.env.name}_{args.expert.task_type}")
        f = open(f"{output_dir}/best_step/best_step.txt", "w")
        f.write(f"{steps}")
        f.close()


def load_agent(agent, pretrain_path, args, ignore_error=True):
    try:
        print(f"=> loading pretrain '{pretrain_path}/{args.agent.name}_{args.env.name}_{args.expert.task_type}'")
        agent.load(f"{pretrain_path}/{args.agent.name}_{args.env.name}_{args.expert.task_type}")
    except Exception as e:
        if ignore_error:
            print("\033[93m" + f"[Info] Warning: cannnot load parameters from {pretrain_path}" + "\033[0m")  # colored print
            time.sleep(1)
        else:
            raise e
    return agent
