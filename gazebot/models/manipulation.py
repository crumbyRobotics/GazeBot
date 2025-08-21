import numpy as np
import torch
from torch import nn
import open3d as o3d
from pytorch3d.structures import Pointclouds, packed_to_list, list_to_padded, list_to_packed
from pytorch3d.ops import sample_farthest_points, knn_points

from ..utils import mlp
from .point_transformer import PointTransformer, Bottleneck
from .transformer import TransformerWithClsToken


class ManipulationModel(nn.Module):
    def __init__(
        self,
        state_dim,
        image_dim,
        num_transition,
        crop_size,
        num_queries,
        state_mean,
        state_std,
        action_pose_mean,
        action_pose_std,
        camera2left_ee,
        camera2right_ee,
        left_camera_intrinsic,
        num_points,
        num_pcd_group,
        pcd_group_size,
        pcd_mask_ratio,
        bezier_vector_thresh,
        hidden_dim=512,
        dropout=0.1,
        nheads=8,
        dim_feedforward=3200,
        enc_layers=4,
        dec_layers=7,
        pre_norm=False,
        max_pcd_token_length=256,
    ):
        super().__init__()

        self.num_arm = num_arm = 2
        self.state_dim = state_dim
        self.num_transition = num_transition
        self.crop_size = crop_size
        self.localbc_action_chunk_size = num_queries
        self.reaching_action_chunk_size = 20  # steps for reaching bottleneck

        # self.se3_param = nn.Parameter(torch.Tensor(np.array(se3_param), dtype=torch.float), requires_grad=True)
        self.register_buffer("camera2left_ee", torch.Tensor(np.array(camera2left_ee)))  # left_camera -> left_arm
        self.register_buffer("camera2right_ee", torch.Tensor(np.array(camera2right_ee)))  # left_camera -> right_arm

        assert len(state_mean) == len(state_std) == len(action_pose_mean) == len(action_pose_std) == state_dim
        self.register_buffer("state_mean", torch.as_tensor(np.array(state_mean), dtype=torch.float))
        self.register_buffer("state_std", torch.as_tensor(np.array(state_std), dtype=torch.float))
        self.register_buffer("action_pose_mean", torch.as_tensor(np.array(action_pose_mean), dtype=torch.float))
        self.register_buffer("action_pose_std", torch.as_tensor(np.array(action_pose_std), dtype=torch.float))

        self.pinhole_camera_intrinsic = left_camera_intrinsic

        self.num_points = num_points
        self.num_pcd_group = num_pcd_group
        self.pcd_group_size = pcd_group_size
        self.pcd_mask_ratio = pcd_mask_ratio

        self.bezier_vector_thresh = (
            bezier_vector_thresh  # replace bezier_vector with zero when |bottleneck - init_state| is under this value (stabilize reaching trajectory)
        )

        # backbone
        self.backbone = PointTransformer(Bottleneck, [1, 2, 3, 5, 2], out_channels=hidden_dim)

        # transformer
        self.hidden_dim = hidden_dim
        self.transformer = TransformerWithClsToken(
            control_atten_mask=True,
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=False,
        )

        self.max_pcd_token_length = max_pcd_token_length

        self.query_embed = nn.Embedding(num_queries + 1, hidden_dim)  # [action head, is_done head
        self.additional_pos_embed = nn.Embedding(num_arm, hidden_dim)  # learned position embedding for state
        self.input_proj_left_state = nn.Linear(state_dim // num_arm, hidden_dim)  # left state
        self.input_proj_right_state = nn.Linear(state_dim // num_arm, hidden_dim)  # right state

        # transformer heads
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_done_head = nn.Linear(hidden_dim, num_transition)
        self.left_bottleneck_adjust_head = nn.Linear(hidden_dim, state_dim // num_arm)
        self.right_bottleneck_adjust_head = nn.Linear(hidden_dim, state_dim // num_arm)

        # bezier vector model
        self.trajectory_mlp_left = mlp(512 + state_dim // num_arm + 3, 256, state_dim // num_arm, 3)
        self.trajectory_mlp_right = mlp(512 + state_dim // num_arm + 3, 256, state_dim // num_arm, 3)

        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("[ManipulationModel] number of parameters: %.2fM" % (n_parameters / 1e6,))

    def forward(
        self,
        image,
        depth_,
        state,
        gaze,
        last_action,
        bottleneck=None,
        localbc_only=False,
        is_inference=False,
    ):
        B, C, H, W = image.shape
        device = image.device

        num_points = self.num_points // 2 if localbc_only else self.num_points  # half num_points used for localbc_only mode

        # Depth preprocess
        for batch_idx in range(B):
            # Interpolate depth map if necessary
            gaze_2d = gaze[batch_idx, :2]  # (2,)
            for i in range(1, 300):
                if depth_[batch_idx, 0, gaze_2d[1], gaze_2d[0]] > 0:
                    break
                neighbor = depth_[batch_idx, 0, max(gaze_2d[1] - i, 0) : gaze_2d[1] + i + 1, max(gaze_2d[0] - i, 0) : gaze_2d[0] + i + 1]
                depth_[batch_idx, 0, gaze_2d[1], gaze_2d[0]] = torch.median(neighbor[neighbor > 0])  # Fill

        # RGBD->pointclouds
        pcd = rgbd_to_pointclouds(depth_, image[:, :3], self.pinhole_camera_intrinsic, valid_depth_only=False)

        # 3D gaze position
        gaze_3d_tensor = pcd.points_padded()[torch.arange(B), gaze[:, 0] + gaze[:, 1] * W]  # (B, 3)
        if self.training:
            gaze_3d_tensor = gaze_3d_tensor + (torch.rand_like(gaze_3d_tensor) * 0.04 - 0.02)

        # Crop pointclouds around the gaze position
        crop_box = torch.stack(
            [
                gaze_3d_tensor - self.crop_size / 2,
                gaze_3d_tensor + self.crop_size / 2,
            ],
            dim=1,
        )  # (B, 2, 3)
        crop_mask = pcd.inside_box(crop_box)  # (N,)
        points_cropped = pcd.points_packed()[crop_mask]  # (N, 3)
        features_cropped = pcd.features_packed()[crop_mask]  # (N, 3)

        cloud_idx = pcd.packed_to_cloud_idx()  # (N,) 各点が属する元のバッチのインデックス
        cloud_idx_cropped = cloud_idx[crop_mask]  # (N',)

        split_size = [torch.sum(cloud_idx_cropped == b).item() for b in range(B)]  # (B,)
        offset = [0] + np.cumsum(split_size).tolist()  # (B+1,)

        # Random down sample
        K = num_points * 2
        down_sample_idx = []
        for b, P in enumerate(split_size):
            if P <= K:  # 点が少なければ全採用
                idx = torch.arange(P, device=device)
            else:  # ランダムに K 個選択
                idx = torch.randperm(P, device=device)[:K]
            idx = offset[b] + idx
            down_sample_idx.append(idx)
            split_size[b] = len(idx)
        offset = [0] + np.cumsum(split_size).tolist()  # (B+1,)
        down_sample_idx = torch.cat(down_sample_idx)
        points_cropped = points_cropped[down_sample_idx]
        features_cropped = features_cropped[down_sample_idx]

        # Translate pointclouds
        for b in range(B):
            points_cropped[offset[b] : offset[b + 1]] -= gaze_3d_tensor[b]
        # Scale pointclouds
        points_cropped *= 2.0 / self.crop_size

        # Crop randomization
        randomize_batch_idx = torch.LongTensor(split_size) > num_points + self.pcd_group_size  # (B,)
        if self.training and randomize_batch_idx.any():
            points_cropped_padded = list_to_padded(packed_to_list(points_cropped, split_size))

            center_pcd_cropped, _ = sample_farthest_points(
                points_cropped_padded[randomize_batch_idx],
                lengths=torch.LongTensor(split_size)[randomize_batch_idx].to(device),
                K=self.num_pcd_group,
                random_start_point=True,
            )  # (B', num_pcd_group, 3)

            center_distances = torch.linalg.norm(center_pcd_cropped, dim=-1)  # (B', num_pcd_group)
            mask_prob = torch.cumsum(center_distances / center_distances.sum(1, keepdim=True), dim=1)  # (B', num_pcd_group)
            mask_idx = torch.sum(mask_prob < torch.rand(len(mask_prob), 1, device=device), dim=-1).long()  # (B',)
            center_pcd_cropped = center_pcd_cropped[torch.arange(len(mask_idx)), mask_idx].unsqueeze(1)  # (B', 1, 3)
            remove_idx = knn_points(
                center_pcd_cropped.cpu(),
                points_cropped_padded[randomize_batch_idx].cpu(),
                lengths1=None,
                lengths2=torch.LongTensor(split_size)[randomize_batch_idx],
                K=self.pcd_group_size,
            ).idx.to(device)  # (B', 1, num_masked_points) # NOTE knn_points is faster in CPU than GPU

            remove_idx = torch.LongTensor(offset[:-1])[randomize_batch_idx].unsqueeze(-1).to(device) + remove_idx.squeeze(1)
            remove_idx = remove_idx.flatten()  # (B'*num_masked_points,)
            remove_mask = torch.ones(len(points_cropped), dtype=torch.bool, device=device)
            remove_mask[remove_idx] = False

            points_cropped = points_cropped[remove_mask]
            features_cropped = features_cropped[remove_mask]

            split_size = [remove_mask[offset[b] : offset[b + 1]].sum().item() for b in range(B)]
            offset = [0] + np.cumsum(split_size).tolist()  # (B+1,)

        # Random down sample 2
        K = num_points + np.random.randint(-1000, 1000) if self.training else num_points
        down_sample_idx = []
        for b, P in enumerate(split_size):
            if P <= K:  # 点が少なければ全採用
                idx = torch.arange(P, device=device)
            else:  # ランダムに K 個選択
                idx = torch.randperm(P, device=device)[:K]
            idx = offset[b] + idx
            down_sample_idx.append(idx)
            split_size[b] = len(idx)
        offset = [0] + np.cumsum(split_size).tolist()  # (B+1,)
        down_sample_idx = torch.cat(down_sample_idx)
        points_cropped = points_cropped[down_sample_idx]
        features_cropped = features_cropped[down_sample_idx]

        # Verify the number of points contained in pointcloud
        if min(split_size) < 256:
            points_list = list(packed_to_list(points_cropped, split_size))
            features_list = list(packed_to_list(features_cropped, split_size))
            for b in range(B):
                while len(points_list[b]) < 256:
                    points_list[b] = torch.cat([points_list[b], points_list[b]])  # Duplicate point cloud
                    features_list[b] = torch.cat([features_list[b], features_list[b]])  # Duplicate point cloud
                    split_size[b] = len(points_list[b])
                    print("[ManipulationModel] Duplicate cropped pcd size:", split_size[b])
                offset[b + 1] = offset[b] + split_size[b]
            points_cropped = list_to_packed(points_list)[0]
            features_cropped = list_to_packed(features_list)[0]

        pcd_cropped_list = []
        if is_inference:
            for b in range(B):
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(points_cropped[offset[b] : offset[b + 1]].cpu().numpy())
                pcd_o3d.colors = o3d.utility.Vector3dVector(
                    features_cropped[offset[b] : offset[b + 1]].cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                )
                pcd_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                pcd_cropped_list.append(pcd_o3d)

        points_dict = {}
        points_dict["coord"] = points_cropped  # (N, 3)
        points_dict["feat"] = features_cropped  # (N, 3)
        points_dict["offset"] = torch.as_tensor(np.array(offset[1:]), dtype=torch.float, device=device)  # (B,)

        # pointcloud embedding
        coord, feat, offset = self.backbone(points_dict)  # (B*N, 3), (B*N, hidden_dim), (B,)

        src, pos, pad_mask = self.padding_token(
            coord, feat, offset
        )  # (B, max_pcd_token_length, hidden_dim), (B, max_pcd_token_length, hidden_dim), (B, max_pcd_token_length)

        # no state embedding
        additional_input = additional_embed = None

        # encoding with transformer
        hs, pcd_feature = self.transformer(
            src,
            pad_mask,
            self.query_embed.weight,
            pos,
            additional_input,
            additional_embed,
        )  # (B, L, hidden_dim), (B, hidden_dim)

        if localbc_only:
            localbc_action = self.local_bc(hs, state)  # (B, localbc_action_chunk_size, state_dim)
            return localbc_action, pcd_cropped_list

        reaching_action, bottleneck, reaching_info = self.reaching_bottleneck(
            gaze_3d_tensor, pcd_feature, last_action, bottleneck, is_inference
        )  # pose, (B, reaching_action_chunk_size, state_dim)

        localbc_action = self.local_bc(hs, state)  # (B, localbc_action_chunk_size, state_dim)

        is_done = self.gaze_transition(hs)  # (B, num_transition)

        return (
            reaching_action,
            localbc_action,
            is_done,
            bottleneck,
            reaching_info,
            pcd_cropped_list,
        )

    def reaching_bottleneck(self, gaze_3d_tensor, pcd_feature, init_state, bottleneck, is_inference):
        B, _ = gaze_3d_tensor.shape
        device = gaze_3d_tensor.device

        # Left
        if bottleneck is None:
            left_bottleneck_adjust = self.left_bottleneck_adjust_head(pcd_feature)  # (B, state_dim // num_arm)
            left_bottleneck_in_gaze = (
                gaze_3d_tensor + left_bottleneck_adjust[:, :3] / 2.0 * self.crop_size
            )  # (B, 3), NOTE normalization scale of left_bottleneck_adjust: -crop_size/2 (m) to crop_size/2 (m) -> -1 to 1
            left_bottleneck_in_f_state = self.transform(left_bottleneck_in_gaze[:, :3], self.camera2left_ee)  # (B, 3)
            left_bottleneck = torch.cat(
                [
                    (left_bottleneck_in_f_state - self.action_pose_mean[:3]) / self.action_pose_std[:3],
                    left_bottleneck_adjust[:, 3:],
                ],  # NOTE normalization scale: 1 m == 1 rad
                dim=1,
            )  # (B, state_dim // num_arm)
        else:
            left_bottleneck = bottleneck[:, : self.state_dim // self.num_arm]  # (B, state_dim // num_arm)

        # reaching trajectory parameter (bezier control point)
        left_init_state = init_state[:, : self.state_dim // self.num_arm]
        left_bezier_vector = self.trajectory_mlp_left(
            torch.cat([pcd_feature, left_init_state, gaze_3d_tensor], dim=1)
        )  # (B, state_dim // num_arm), NOTE output scale: meter, usually smaller than norm of bottleneck - init_state

        # replace bezier_vector to zero when trajectory is short
        if is_inference:
            left_bezier_vector = torch.where(
                torch.norm(left_init_state[: self.state_dim] - left_bottleneck[: self.state_dim], dim=1, keepdim=True)
                > self.bezier_vector_thresh,  # NOTE exclude gripper
                left_bezier_vector,
                torch.zeros_like(left_bezier_vector),
            )

        # Right
        if bottleneck is None:
            right_bottleneck_adjust = self.right_bottleneck_adjust_head(pcd_feature)  # (B, state_dim // num_arm)
            right_bottleneck_in_gaze = (
                gaze_3d_tensor + right_bottleneck_adjust[:, :3] / 2.0 * self.crop_size
            )  # (B, 3), NOTE normalization scale of left_bottleneck_adjust: -crop_size/2 (m) to crop_size/2 (m) -> -1 to 1
            right_bottleneck_in_f_state = self.transform(right_bottleneck_in_gaze[:, :3], self.camera2right_ee)  # (B, 3)
            right_bottleneck = torch.cat(
                [
                    (right_bottleneck_in_f_state - self.action_pose_mean[self.state_dim // self.num_arm :][:3])
                    / self.action_pose_std[self.state_dim // self.num_arm :][:3],
                    right_bottleneck_adjust[:, 3:],
                ],  # NOTE normalization scale: 1 m == 1 rad
                dim=1,
            )  # (B, state_dim // num_arm)
        else:
            right_bottleneck = bottleneck[:, self.state_dim // self.num_arm :]  # (B, state_dim // num_arm)

        # reaching trajectory parameter (bezier control point)
        right_init_state = init_state[:, self.state_dim // self.num_arm :]
        right_bezier_vector = self.trajectory_mlp_right(
            torch.cat([pcd_feature, left_init_state, gaze_3d_tensor], dim=1)
        )  # (B, state_dim // num_arm), NOTE output scale: meter, usually smaller than norm of bottleneck - init_state

        # replace bezier_vector to zero when trajectory is short
        if is_inference:
            right_bezier_vector = torch.where(
                torch.norm(right_init_state[: self.state_dim] - right_bottleneck[: self.state_dim], dim=1, keepdim=True)
                > self.bezier_vector_thresh,  # NOTE exclude gripper
                right_bezier_vector,
                torch.zeros_like(right_bezier_vector),
            )

        # Combine
        bottleneck = torch.cat([left_bottleneck, right_bottleneck], dim=1)  # (B, state_dim)
        bezier_vector = torch.cat([left_bezier_vector, right_bezier_vector], dim=1)  # (B, state_dim)

        # bezier curve (extended to s < 0 and s > 1)
        def extended_bezier_traj(p0, p1, c, s):
            return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s**2 * p1
            # return torch.where(s < 0, (1 - 2 * s) * p0 + 2 * s * c, torch.where(s > 1, 2 * (1 - s) * c + (2 * s - 1) * p1, (1 - s)**2 * p0 + 2 * (1 - s) * s * c + s**2 * p1))

        bezier_control = (init_state + bottleneck.detach()) / 2 + bezier_vector  # (B, state_dim)

        # bezier_params = 0.1 * torch.ones(
        #     B, 2 * self.reaching_action_chunk_size, dtype=torch.float, device=device
        # )  # (B, 2 * num_queries), NOTE action norm is no longer important in inference
        bezier_params = torch.linspace(0, 1, self.reaching_action_chunk_size + 1, device=device)[1:].repeat(B, 2)  # (B, 2*num_queries)

        all_action = extended_bezier_traj(
            init_state.unsqueeze(1),
            bottleneck.detach().unsqueeze(1),
            bezier_control.unsqueeze(1),
            bezier_params.unsqueeze(-1),
        )  # (B, 2 * num_queries, state_dim)
        action = torch.cat(
            [
                all_action[
                    :,
                    : self.reaching_action_chunk_size,
                    : self.state_dim // self.num_arm,
                ],
                all_action[
                    :,
                    self.reaching_action_chunk_size :,
                    self.state_dim // self.num_arm :,
                ],
            ],
            dim=2,
        )  # (B, num_queries, state_dim)

        return (
            action,
            bottleneck,
            dict(bezier_vector=bezier_vector, bezier_params=bezier_params),
        )

    def local_bc(self, hs, state):
        action = self.action_head(hs[:, :-1])  # (B, num_queries, state_dim)

        state_norm = (
            (state * self.state_std + self.state_mean) - self.action_pose_mean
        ) / self.action_pose_std  # (B, state_dim) normalized state by action metrics
        action = torch.cat(
            [
                state_norm.unsqueeze(1)[:, :, :6] + action[:, :, :6],
                action[:, :, [6]],
                state_norm.unsqueeze(1)[:, :, 7:13] + action[:, :, 7:13],
                action[:, :, [13]],
            ],
            dim=2,
        )  # action in state space

        return action  # (B, num_queries, state_dim)

    def gaze_transition(self, hs):
        return self.is_done_head(hs[:, -1])  # (B, num_transition)

    def transform(self, x, T):
        """
        args
            x: (B, 3)
        return
            (B, 3)
        """
        B = x.shape[0]
        T = T.expand(B, 4, 4)  # (B, 4, 4)

        return T.matmul(torch.cat((x, torch.ones(B, 1, dtype=torch.float, device=x.device)), dim=1).unsqueeze(-1)).squeeze(-1)[:, :3]  # (B, 3)

    def padding_token(self, coord, feat, offset):
        pos_ = self.coord_embedding_sine(coord, scale=True)  # (B*N, hidden_dim)

        offset = [0] + offset.detach().cpu().tolist()
        src_pad = []
        pos_pad = []
        key_padding_mask = []
        for i in range(len(offset) - 1):
            src = feat[offset[i] : offset[i + 1]]
            pos = pos_[offset[i] : offset[i + 1]]
            mask = torch.zeros(self.max_pcd_token_length, dtype=torch.bool, device=feat.device)

            pad_length = self.max_pcd_token_length - len(src)
            if pad_length > 0:
                pad = torch.zeros((pad_length, feat.shape[1]), dtype=feat.dtype, device=feat.device)
                mask = torch.cat((torch.zeros(len(src)), torch.ones(pad_length))).to(torch.bool).to(feat.device)
                src = torch.cat((src, pad))
                pos = torch.cat((pos, pad))
            else:
                print("[ManipulationModel] Warning: size of pointcloud features exceed the max token length!")
                src = src[: self.max_pcd_token_length]
                pos = pos[: self.max_pcd_token_length]

            src_pad.append(src)
            pos_pad.append(pos)
            key_padding_mask.append(mask)

        src_pad = torch.stack(src_pad)  # (B, T, hidden_dim)
        pos_pad = torch.stack(pos_pad)  # (B, T, hidden_dim)
        key_padding_mask = torch.stack(key_padding_mask)  # (B, T)

        return src_pad, pos_pad, key_padding_mask

    def coord_embedding_sine(self, coord, temperature=10000, scale=False):
        num_pos_feats = self.hidden_dim // 3
        num_pad_feats = self.hidden_dim - num_pos_feats * 3

        x_embed = coord[:, 0:1]
        y_embed = coord[:, 1:2]
        z_embed = coord[:, 2:3]

        if scale:
            scale = 2 * torch.pi
            x_embed = coord[:, 0] * scale
            y_embed = coord[:, 1] * scale
            z_embed = coord[:, 2] * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=coord.device)  # (num_pos_feats,)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # (num_pos_feats,)

        pos_x = x_embed[..., None] / dim_t  # (Npoints, 1, num_pos_feats)
        pos_y = y_embed[..., None] / dim_t  # (Npoints, 1, num_pos_feats)
        pos_z = z_embed[..., None] / dim_t  # (Npoints, 1, num_pos_feats)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=2).flatten(1)  # (Npoints, num_pos_feats)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=2).flatten(1)  # (Npoints, num_pos_feats)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=2).flatten(1)  # (Npoints, num_pos_feats)
        pos = torch.cat((pos_x, pos_y, pos_z), dim=1)  # (Npoints, 3 * num_pos_feats)

        pos = torch.cat((pos, torch.zeros_like(pos)[:, :num_pad_feats]), dim=1)  # (Npoints, hidden_dim)
        return pos


def rgbd_to_pointclouds(
    depths: torch.Tensor,
    rgb_images: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    depth_scale: float = 1000.0,
    valid_depth_only=False,
) -> Pointclouds:
    """
    depths: (B, 1, H, W)
    rgb_images: (B, 3, H, W)
    intrinsics: (4,) -> fx, fy, cx, cy
    """
    B, _, H, W = depths.shape
    device = depths.device
    fx, fy, cx, cy = intrinsics

    # Make a mesh-grid of pixel centres (u,v).
    u = torch.arange(W, device=device, dtype=depths.dtype)
    v = torch.arange(H, device=device, dtype=depths.dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")  # (H, W)

    # Expand to (B, H, W).
    uu = uu.unsqueeze(0).expand(B, -1, -1)
    vv = vv.unsqueeze(0).expand(B, -1, -1)

    # Depth in metres.
    z = depths.squeeze(1) / depth_scale  # (B, H, W)

    # Back-project.
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z

    # Stack → (B, H, W, 3).
    xyz = torch.stack((x, y, z), dim=-1)

    # Flatten → (B, H*W, 3) and RGB → (B, H*W, 3).
    xyz_flat = xyz.reshape(B, -1, 3)
    rgb_flat = rgb_images.permute(0, 2, 3, 1).reshape(B, -1, 3)

    # Optionally drop invalid depths.
    if valid_depth_only:
        mask = (z > 0 ^ ~torch.isnan(z)).view(B, -1)  # (B, H*W)
    else:
        mask = torch.ones_like(z.reshape(B, -1), dtype=torch.bool)

    points_list = [xyz_b[m] for xyz_b, m in zip(xyz_flat, mask)]
    features_list = [rgb_b[m] for rgb_b, m in zip(rgb_flat, mask)]

    return Pointclouds(points_list, features=features_list)
