import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh

def test_model(model, 
               diffusion, 
               xm, 
               output_folder, 
               epoch, 
               prompt,
               device,
               cond=None,
               display=False,
               guidance_scale=15.0,
               render_mode='nerf',
               size=160,
               return_latents=False,
               seen_sample=False,
               save_mesh=False, 
               ):
    # make output folder if it doesn't exist
    render_path = os.path.join(output_folder, 'output')
    os.makedirs(render_path, exist_ok=True)

    # generate latents
    batch_size = 1
    model_kwargs = dict(texts=[prompt] * batch_size)
    if cond is not None:
        model_kwargs['cond'] = torch.unsqueeze(cond, 0).repeat(batch_size, 1, 1)
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=model_kwargs,
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    if return_latents:
        return latents

    # visualize examples
    cameras = create_pan_cameras(size, device)
    filename = prompt.replace(' ', '_')
    for _, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        video_path = os.path.join(render_path, 'output.mp4')
        torch.save(latent, os.path.join(render_path, "output.pt"))
        if seen_sample:
            video_path = video_path.replace('.mp4', '_seen.mp4')
        videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (size, size))
        for i, image in enumerate(images):
            filename = filename.replace('\"', '')
            filename = filename.replace('\\', '')
            filename = filename.replace('/', '')
            image_path = os.path.join(render_path, f'{(i):05}.png')
            if seen_sample:
                image_path = image_path.replace('.png', '_seen.png')
            image.save(os.path.join(render_path, f'{(i):05}.png'))
            # convert image to numpy
            image = np.array(image)
            image = image[:,:,::-1]
            videowriter.write(image)
        videowriter.release()
        if save_mesh:
            t = decode_latent_mesh(xm, latent).tri_mesh()
            with open(os.path.join(render_path, "output.obj"), 'w') as f:
                t.write_obj(f)
            with open(os.path.join(render_path, "output.ply"), 'wb') as f:
                t.write_ply(f)