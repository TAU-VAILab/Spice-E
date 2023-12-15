import base64
import io
from typing import Union

import ipywidgets as widgets
import numpy as np
import torch
from PIL import Image

from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
from shap_e.models.transmitter.base import Transmitter, VectorDecoder
from shap_e.rendering.torch_mesh import TorchMesh
from shap_e.util.collections import AttrDict


def create_pan_cameras(size: int, device: torch.device, num_frames: int=20) -> DifferentiableCameraBatch:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=num_frames):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
            width=size,
            height=size,
            x_fov=0.7,
            y_fov=0.7,
        ),
    )


@torch.no_grad()
def decode_latent_images(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
    cameras: DifferentiableCameraBatch,
    rendering_mode: str = "stf",
    background_color: torch.Tensor = torch.tensor([255.0, 255.0, 255.0]),
):
    decoded = xm.renderer.render_views(
        AttrDict(cameras=cameras),
        params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode=rendering_mode, render_with_direction=False),
    )
    bg_color = background_color.to(decoded.channels.device)
    arr = bg_color * decoded.transmittance + (1 - decoded.transmittance) * decoded.channels
    arr = arr.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return [Image.fromarray(x) for x in arr]


@torch.no_grad()
def decode_latent_mesh(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
) -> TorchMesh:
    decoded = xm.renderer.render_views(
        #AttrDict(cameras=create_pan_cameras(2, latent.device)),  # lowest resolution possible
        AttrDict(cameras=create_pan_cameras(16, latent.device)),  # lowest resolution possible
        params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    return decoded.raw_meshes[0]


def gif_widget(images):
    writer = io.BytesIO()
    images[0].save(
        writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0
    )
    writer.seek(0)
    data = base64.b64encode(writer.read()).decode("ascii")
    return widgets.HTML(f'<img src="data:image/gif;base64,{data}" />')
