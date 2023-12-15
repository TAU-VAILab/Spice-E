import torch
import argparse
import os

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_examples_to_gen_for_text', type=int, default=1)
parser.add_argument('-p', '--prompt', type=str, default='a spider mech')
parser.add_argument('-o', '--output_folder', type=str, default='/storage/etaisella/repos/shape_proj/outputs/')


def generate_3d_from_text(args):
    batch_size = args.num_examples_to_gen_for_text
    prompt = args.prompt

    # setup output path
    output_folder = args.output_folder
    output_path = os.path.join(output_folder, prompt.replace(' ', '_'))
    os.makedirs(output_path, exist_ok=True)

    # setup model
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    guidance_scale = 15.0

    # generate latents
    print(f'generating {batch_size} latents for prompt "{prompt}"')
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # visualize examples
    print('visualizing examples')
    render_mode = 'nerf' # you can change this to 'stf'
    size = 64 # this is the size of the renders; higher values take longer to render.
    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        for j, image in enumerate(images):
            image.save(os.path.join(output_path, f'{i:05}_{j:05}.png'))
    
    # Export meshes
    print('Exporting meshes')
    for i, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()
        with open(os.path.join(output_path, f'example_mesh_{i}.ply'), 'wb') as f:
            t.write_ply(f)
        with open(os.path.join(output_path, f'example_mesh_{i}.obj'), 'w') as f:
            t.write_obj(f)

if __name__ == '__main__':
    args = parser.parse_args()
    generate_3d_from_text(args)