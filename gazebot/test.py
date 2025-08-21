from omegaconf import open_dict
import hydra

from .agent import create_gaze_agent, create_manipulation_agent
from .train import state_action_metrics, load_agent
from .env_wrapper import TongSystemObsWrapper, TongSystemActionWrapper

from tongsystem import TongSystemFixedNeck, TongSystemUseOnlyLeftArm


def make_env(args):
    # NOTE various initial poses for the robot
    # fmt: off
    init_poses = [
        [-0.8516850, -1.5127810, 1.8618909, -2.3572762, -1.2322809, 1.5295182, 16.0550690, 0.8669366, -1.6407274, -1.7815359, -0.8877410, 1.1811637, -1.5764903, -21.7168102, 0.0, -0.9],
        # [-0.837758, -0.575959, 1.492257, -2.49425003, -1.06465, 1.1894419, 11.001, 0.8669366, -1.6407274, -1.7815359, -0.8877410, 1.1811637, -1.5764903, -21.7168102, 0.000, -0.900],
        # [-0.97337012, -0.59358648, 1.0400417, -2.022313, -0.4171337, 1.4081316, 1.001, 0.8669366, -1.6407274, -1.7815359, -0.8877410, 1.1811637, -1.5764903, -21.7168102, 0.000, -0.900],
        # [-0.7470009, -0.9711012, 2.284636, -3.07178, -1.15192, 1.62316, 20.0, 0.8669366, -1.6407274, -1.7815359, -0.8877410, 1.1811637, -1.5764903, -21.7168102, 0.000, -0.900],
        # [-0.8516850, -1.5127810, 1.8618909, -2.3572762, -1.2322809, 1.5295182, 16.0550690, 0.855211, -2.3911, -1.22173, -0.872665, 0.820305, -1.36136, -5, 0.000, -0.900],
        # [-0.8516850, -1.5127810, 1.8618909, -2.3572762, -1.2322809, 1.5295182, 16.0550690, 0.959931, -2.68781, -1.36136, -1.11701, 0.872665, -1.43117, -21, 0.000, -0.900],
        # [-0.8516850, -1.5127810, 1.8618909, -2.3572762, -1.2322809, 1.5295182, 16.0550690, 0.663225, -2.04204, -1.8326, -1.01229, 1.44862, -1.85005, -11, 0.000, -0.900],
    ]
    # fmt: on

    env = TongSystemFixedNeck(
        state_type="pos",
        action_type="state",
        image_width=args.env.image_dim[2],
        image_height=args.env.image_dim[1],
        init_poses=init_poses,
        log_dir=args.log_dir,
    )

    state_mean, state_std, action_pose_mean, action_pose_std = state_action_metrics(args)

    metrics = {
        "state_mean": state_mean[:14],
        "state_std": state_std[:14],
        "action_pose_mean": action_pose_mean[:14],
        "action_pose_std": action_pose_std[:14],
    }

    env = TongSystemObsWrapper(
        env,
        metrics["state_mean"],
        metrics["state_std"],
        args.device,
    )
    # env = TongSystemActionWrapper(env, control_arm="left") # NOTE single arm

    return env, metrics


def make_agent(args, ignore_error=False):
    assert args.agent.name == "manipulation"

    # TODO adhoc configure
    assert not hasattr(args.agent, "super_res") and not hasattr(args.agent, "small_image_dim")
    with open_dict(args.agent):
        args.agent.super_res = 4
        args.agent.small_image_dim = [3, 224, 224]

    gaze_agent = make_gaze_agent(args, ignore_error)

    manipulation_agent = make_manipulation_agent(args, ignore_error)

    return gaze_agent, manipulation_agent


def make_manipulation_agent(args, ignore_error=False):
    agent = create_manipulation_agent(args)

    model_path = hydra.utils.to_absolute_path(args.agent.manipulation_model_path)
    load_agent(agent, model_path, args, ignore_error)

    agent.eval()

    return agent


def make_gaze_agent(args, ignore_error=False):
    agent = create_gaze_agent(args)

    model_path = hydra.utils.to_absolute_path(args.agent.gaze_model_path)
    load_agent(agent, model_path, args, ignore_error)

    agent.eval()

    return agent
