import torch
import numpy as np


def create_gaze_agent(args):
    from .models.gaze import GazeModel

    device = torch.device(args.device)

    gaze_model = GazeModel(num_segs=args.expert.num_sub_task, small_image_dim=args.agent.small_image_dim, super_res=args.agent.super_res).to(device)
    gaze_model_optimizer = torch.optim.AdamW(gaze_model.parameters(), lr=args.agent.lr, weight_decay=args.agent.weight_decay)

    agent = Agent(gaze_model, gaze_model_optimizer, device)

    return agent


def create_manipulation_agent(args):
    from .train import state_action_metrics
    from .models.manipulation import ManipulationModel

    device = torch.device(args.device)

    state_mean, state_std, action_pose_mean, action_pose_std = state_action_metrics(args)

    if args.env.name == "tong":
        camera2left_ee = np.array(
            [
                [-0.02177536, 0.03698473, -0.99907856, -0.22352613],
                [-0.03215057, -0.99882456, -0.03627459, 0.30907208],
                [-0.9992458, 0.03133106, 0.02293885, -0.18261354],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        camera2right_ee = np.array(
            [
                [0.02638825, -0.06179807, 0.99773977, 0.22079462],
                [-0.03824855, -0.9974189, -0.0607666, 0.32936743],
                [0.99891977, -0.03655858, -0.02868382, -0.22785084],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif args.env.name == "tongsim":
        camera2left_ee = np.array(
            [
                [0.000, 0.114, -0.9934, -0.2534],
                [0.000, -0.9934, -0.1144, 0.31953],
                [-1.000, 0.0000, 0.0000, -0.1715],
                [0.000, 0.0000, 0.0000, 1.00000],
            ]
        )
        camera2right_ee = np.array(
            [
                [0.000, -0.114, 0.9934, 0.2534],
                [0.000, -0.9934, -0.114, 0.3195],
                [1.000, 0.000, 0.000, -0.2285],
                [0.000, 0.000, 0.000, 1.0000],
            ]
        )
    else:
        raise NotImplementedError

    model = ManipulationModel(
        args.env.state_dim,
        args.env.image_dim,
        args.expert.num_sub_task,
        args.agent.crop_size_pcd,
        args.agent.num_queries,
        state_mean[:14],
        state_std[:14],
        action_pose_mean[:14],
        action_pose_std[:14],
        camera2left_ee,
        camera2right_ee,
        args.env.left_camera_intrinsic,
        args.agent.num_points,
        args.agent.num_pcd_group,
        args.agent.pcd_group_size,
        args.agent.pcd_mask_ratio,
        args.agent.bezier_vector_thresh,
        args.agent.hidden_dim,
        args.agent.dropout,
        args.agent.nheads,
        args.agent.dim_feedforward,
        args.agent.enc_layers,
        args.agent.dec_layers,
        args.agent.pre_norm,
        args.agent.max_pcd_token_length,
    ).to(device)

    def exclude(n, p):
        return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n

    def include(n, p):
        return not exclude(n, p)

    gain_or_bias_params = [p for n, p in model.named_parameters() if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in model.named_parameters() if include(n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.agent.weight_decay},
        ],
        lr=args.agent.lr,
    )

    agent = ManipulationAgent(model, optimizer, device)

    return agent


class Agent:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer

        self.device = device

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    def update(self, batch):
        raise NotImplementedError  # add update method (e.g. agent.update = types.MethodType(update, agent))

    def validate_fn(self, batch):
        raise NotImplementedError  # add update method (e.g. agent.update = types.MethodType(validate_fn, agent))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, self.device))


class ManipulationAgent(Agent):
    def __call__(self, obs, gaze, last_action, last_bottleneck=None, is_inference=False):
        return self.model(obs["image"], obs["depth"], obs["state"], gaze, last_action, last_bottleneck, is_inference=is_inference)

    def local_bc(self, obs, gaze, is_inference=False):
        return self.model(obs["image"], obs["depth"], obs["state"], gaze, None, None, localbc_only=True, is_inference=is_inference)
