import io
import os
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as VF
import cv2
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt


def to_absolute_path(path):
    """
    Convert a relative path to an absolute path.
    The relative path in this project is relative to the root directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, path))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterDict(dict):
    def _update(self, key, val, n=1):
        if key not in self.keys():
            self[key] = AverageMeter()  # Create a new AverageMeter if not exists
        self[key].update(val, n)

    def update(self, val_dict: dict, n=1):
        for key in val_dict.keys():
            self._update(key, val_dict[key], n)

    def reset(self, key=None):
        if key is None:
            for meter in self.values():
                meter.reset()
        elif key in self.keys():
            self[key].reset()

    def get_averages(self) -> dict:
        return {key: meter.avg for key, meter in self.items()}


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def crop(image, gaze, crop_size, resize=(224, 224)):
    # 十分な大きさpadding(100pixelずつくらい)
    padding_size = 100
    image = VF.pad(image, padding_size, padding_mode="constant")

    B, H, W = image.shape[0], *image.shape[-2:]
    cropped_images = []
    for i in range(B):
        crop_height = crop_width = crop_size

        # Calculate the top-left corner of the crop based on the center for both left and right images
        start_yl = gaze[i, 1] - crop_height // 2 + padding_size
        start_xl = gaze[i, 0] - crop_width // 2 + padding_size
        start_yr = gaze[i, 3] - crop_height // 2 + padding_size
        start_xr = gaze[i, 2] - crop_width // 2 + padding_size

        # Ensure the start positions are within image boundaries
        start_yl = torch.clamp(start_yl, 0, H - crop_height).to(torch.long)
        start_xl = torch.clamp(start_xl, 0, W - crop_width).to(torch.long)
        start_yr = torch.clamp(start_yr, 0, H - crop_height).to(torch.long)
        start_xr = torch.clamp(start_xr, 0, W - crop_width).to(torch.long)

        # Create a meshgrid for cropping
        y = torch.arange(0, crop_height).unsqueeze(1).repeat(1, crop_width).to(image.device)
        x = torch.arange(0, crop_width).unsqueeze(0).repeat(crop_height, 1).to(image.device)

        # Add the start positions to the meshgrid for both left and right images
        yl = y + start_yl
        xl = x + start_xl
        yr = y + start_yr
        xr = x + start_xr

        # Gather the pixel values using the computed positions for left and right images
        if len(image.shape) == 4:
            # B, 6, H, W
            cropped_image = torch.cat((image[i, :3, yl, xl], image[i, 3:, yr, xr]), dim=0)  # (C, crop_height, crop_width)
        else:
            # B, 2, C, H, W
            cropped_image = torch.stack((image[i, 0, :, yl, xl], image[i, 1, :, yr, xr]), dim=0)  # (2, C, crop_height, crop_width)

        # Resize to (224, 224)
        cropped_image = VF.resize(cropped_image, resize).contiguous()
        cropped_images.append(cropped_image)

    return torch.stack(cropped_images)  # (B, C, 224, 224)


def bezier_term(n, t):
    """
    input:
        n: dim of bezier + 1
        t: param, (num_points,)
    """
    device = t.device

    t = t.unsqueeze(0)  # Shape (1, num_points)
    i = torch.arange(n, device=device).unsqueeze(1)  # Shape (n, 1)

    # Calculate binomial coefficients
    binomial_coeffs = torch.exp(
        torch.lgamma(torch.tensor(n, device=device)) - (torch.lgamma(i + 1) + torch.lgamma(torch.tensor(n - i, device=device)))
    )

    # Compute the Bernstein basis polynomials
    J = binomial_coeffs * t**i * (1 - t) ** (n - 1 - i)
    return J  # Shape (n, num_points)


def bezier(B, t):
    *_, n = B.shape  # (2, dim+1)
    return B @ bezier_term(n, t)


class PointCloudViewer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.window = False

        self.geoms = []

    def _add_geometry(self, *geoms):
        for geom in geoms:
            self.vis.add_geometry(geom)
        self.geoms = geoms

    def _clear_geometries(self):
        self.vis.clear_geometries()
        self.geoms = []

    def render(self, *geoms):
        if not self.window:
            self.vis.create_window(window_name="PointCloud", width=960, height=540, left=1000, top=50)
            self.window = True

        self._clear_geometries()
        self._add_geometry(*geoms)

        for geom in self.geoms:
            self.vis.update_geometry(geom)

        self.vis.poll_events()
        self.vis.update_renderer()

    def save(self, path):
        self.vis.capture_screen_image(path)

    def close(self):
        self.vis.destroy_window()


def pcd2image(pcd: o3d.geometry.PointCloud, elev: int = 70, azim: int = -80) -> Image.Image:
    # PointCloudをnumpy配列に変換
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # matplotlibでプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.gca().axis("off")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)

    # プロットをPILイメージに変換
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)

    return image


def array2image(array_: np.ndarray, normalize=True) -> Image.Image:
    array = array_.copy()
    if normalize:
        array = (array - array.min()) / (array.max() - array.min())
    return Image.fromarray((array * 255.0).astype(np.uint8))


def tensor2image(tensor: torch.Tensor) -> Image.Image:
    """
    tensor: (3, H, W), torch.Tensor
    output: (H, W, 3), PIL.Image.Image
    """
    return array2image(tensor.detach().cpu().numpy().transpose(1, 2, 0))


def tensor2opencv(tensor: torch.Tensor) -> np.ndarray:
    """
    tensor: (3, H, W), torch.Tensor
    output: (H, W, 3), np.ndarray
    """
    return cv2.cvtColor(np.array(tensor2image(tensor)), cv2.COLOR_RGB2BGR)


def image2opencv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def save_as_gif(image_list, path):
    assert isinstance(image_list[0], Image.Image)
    image_list[0].save(path, save_all=True, append_images=image_list[1:], loop=0)


def custom_array2string(arr, max_elements_per_line, separator=",", formatter={"float_kind": lambda x: "{: .3f}".format(x)}, **kwargs):
    array_string = np.array2string(arr, separator=separator, max_line_width=np.inf, formatter=formatter, **kwargs)
    elements = array_string.strip("[]").split(separator)

    lines = []
    for i in range(0, len(elements), max_elements_per_line):
        line = separator.join(elements[i : i + max_elements_per_line])
        lines.append(line)

    return "[{}]".format(f"{separator}\n ".join(lines))
