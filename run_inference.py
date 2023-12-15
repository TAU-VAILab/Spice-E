import argparse
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from shap_e.models.testing_utils import test_model
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh
from shap_e.util.data_util import load_or_create_multimodal_batch




parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, 
                    help='path to model', required=True)
parser.add_argument('-d', '--data_path', type=str, 
                    help='path to data', required=True)
parser.add_argument('-o', '--output_dir', type=str, 
                    help='path to output dir', required=True)
parser.add_argument('-p', '--prompt', type=str, 
                    help='text prompt', required=True)
parser.add_argument('--encode_guidance', action='store_true',
                    help='whether to encode the input data with shap-e encoder')
parser.add_argument('--guidance_scale', type=float, default=7.5, 
                    help='guidance scale')
parser.add_argument('--render_mode', type=str, default='stf',
                    help='the decoding mode to render')
parser.add_argument('--output_resolution', type=int, default=512, 
                    help='resolution of output images')
parser.add_argument('--save_mesh', action='store_true', default=False, 
                    help='whether to save mesh as output')
parser.add_argument('--render_guidance', action='store_true', default=False,
                    help='whether to render the guidance shape')
parser.add_argument('--input_guidance_object_path', type=str, 
                    help='path to input guidance object for decoding')
parser.add_argument('--mv_image_size', type=int, 
                    help='size of the images', default=256)
parser.add_argument('--verbose_blender', action='store_true', default=False,
                    help='if enabled, prints outputs from blender script')


def prompt2filename(prompt: str):
    filename = prompt.replace(" ", "_")
    filename = filename.replace("?", "")
    filename = filename.replace("!", "")
    filename = filename.replace(",", "")
    filename = filename.replace('\"', '')
    filename = filename.replace('\\', '')
    filename = filename.replace('/', '')
    return filename


def infer(args, device):
    model_path = args.model_path
    data_path = args.data_path
    output_dir = args.output_dir
    prompt = args.prompt
    encode_guidance = args.encode_guidance
    guidance_scale = args.guidance_scale
    render_mode = args.render_mode
    assert render_mode in ["stf", "nerf"]
    output_resolution = args.output_resolution
    save_mesh = args.save_mesh
    render_guidance = args.render_guidance
    input_guidance_object_path = args.input_guidance_object_path
    mv_image_size = args.mv_image_size
    verbose_blender = args.verbose_blender
    cameras = create_pan_cameras(output_resolution, device)

    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.wrapped.backbone.make_ctrl_layers()
    model.wrapped.set_up_controlnet_cond()
    model.load_state_dict(torch.load(model_path))
    diffusion = diffusion_from_config(load_config('diffusion'))

    # create output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if encode_guidance:
        print("Creating data for encoding from input 3D guidance shape")
        batch = load_or_create_multimodal_batch(
            device,
            model_path=input_guidance_object_path,
            mv_light_mode="basic",
            mv_image_size=mv_image_size,
            cache_dir=os.path.join(data_path, "cached_guidance"),
            verbose=verbose_blender,) # this will show Blender output during renders
        print("Encoding")
        guidance_shape = xm.encoder.encode_to_bottleneck(batch)      

    else:
        guidance_shape = torch.load(data_path)

    with torch.no_grad():        
        filename = prompt2filename(prompt)
        prompt = " ".join(filename.split("_"))
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_path, exist_ok=True)

        # Rendering Model Output
        print(f"rendering samples for prompt: {prompt}")
        test_model(model=model,
                diffusion=diffusion, 
                xm=xm,
                output_folder=output_path,
                cond=guidance_shape[0].to(device).detach(),
                epoch=0, 
                prompt=prompt,
                device=device,
                guidance_scale=guidance_scale,
                render_mode=render_mode,
                size=output_resolution,
                save_mesh=save_mesh)

        if render_guidance:
            # Rendering Guidance 
            print(f"rendering condition latent for prompt: {prompt}")
            images = decode_latent_images(xm, guidance_shape, cameras, rendering_mode=render_mode)
            cond_path = os.path.join(output_dir, "condition")
            os.makedirs(cond_path, exist_ok=True)
            torch.save(guidance_shape,os.path.join(cond_path, "condition.pt"))
            videowriter = cv2.VideoWriter(os.path.join(cond_path, 'condition.mp4'),
                                            cv2.VideoWriter_fourcc(*'mp4v'), 10, (size, size))
            for i, image in enumerate(images):
                image.save(os.path.join(cond_path, f'{(i):05}.png'))
                image = np.array(image)
                image = image[:,:,::-1]
                videowriter.write(image)
            videowriter.release()
            if save_mesh:
                t = decode_latent_mesh(xm, guidance_shape).tri_mesh()
                with open(os.path.join(cond_path, "condition.obj"), 'w') as f:
                    t.write_obj(f)
                with open(os.path.join(cond_path, "condition.ply"), 'wb') as f:
                    t.write_ply(f)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    infer(args, device)
