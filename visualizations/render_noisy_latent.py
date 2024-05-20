import argparse
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shap_e.models.download import load_model
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh
from blender_rendering import good_looking_render

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, 
                    help='path to data', required=True)
parser.add_argument('-o', '--output_dir', type=str, 
                    help='path to output dir', required=True)
parser.add_argument('--noise_scale', type=float, default=0.4, 
                    help='guidance scale')
parser.add_argument('--render_mode', type=str, default='stf',
                    help='the decoding mode to render')
parser.add_argument('--output_resolution', type=int, default=64, 
                    help='resolution of output images')
parser.add_argument('--noise_chunks', type=int, default=1, 
                    help='number of chunks to split the noise into')
parser.add_argument('--save_mesh', action='store_true', default=True, 
                    help='whether to save mesh as output')
parser.add_argument('--blender_only', action='store_true', default=False, 
                    help='whether to only render output in blender')

def infer(args, device):
    data_path = args.data_path
    output_dir = args.output_dir
    noise_scale = args.noise_scale
    render_mode = args.render_mode
    assert render_mode in ["stf", "nerf"]
    output_resolution = args.output_resolution
    save_mesh = args.save_mesh
    cameras = create_pan_cameras(output_resolution, device)
    xm = load_model('transmitter', device=device)

    # create output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)   

    # load latent and add noise to it
    input_latent = torch.load(data_path)
    print(input_latent.shape)
    noise = torch.randn_like(input_latent)*1.0
    noisy_latent = input_latent * (1 - noise_scale) + noise_scale * noise
    input_latent = input_latent.view(1024, 1024)
    noisy_latent = noisy_latent.view(1024, 1024)
    chunk_size = 1024 // args.noise_chunks

    for i in range(args.noise_chunks + 1):
        # add noise to chunk
        noisy_latent_ch = input_latent.clone()

        if i > 0 :
            noisy_latent_ch[(i - 1) * chunk_size: i * chunk_size, :] = noisy_latent[(i - 1) * chunk_size: i * chunk_size, :]

        # change the shape back to 1048576
        noisy_latent_ch = noisy_latent_ch.view(1048576)

        with torch.no_grad():
            # Rendering Latent
            if not args.blender_only:
                images = decode_latent_images(xm, noisy_latent_ch, cameras, rendering_mode=render_mode)
                torch.save(noisy_latent_ch, os.path.join(output_dir, f"noisy_ch{i}_of_{args.noise_chunks}.pt"))
                videowriter = cv2.VideoWriter(os.path.join(output_dir, f"noisy_ch{i}_of_{args.noise_chunks}.mp4"),
                                                cv2.VideoWriter_fourcc(*'mp4v'), 10, (output_resolution, output_resolution))
                for j, image in enumerate(images):
                    image.save(os.path.join(output_dir, f'noisy_ch{i}_of_{args.noise_chunks}_{(j):05}.png'))
                    image = np.array(image)
                    image = image[:,:,::-1]
                    videowriter.write(image)
                videowriter.release()

            if save_mesh:
                t = decode_latent_mesh(xm, noisy_latent_ch).tri_mesh()
                with open(os.path.join(output_dir, "noisy.obj"), 'w') as f:
                    t.write_obj(f)
                mesh_path = os.path.join(output_dir, "noisy.ply")
                with open(mesh_path, 'wb') as f:
                    t.write_ply(f)

                blender_img_path = os.path.join(output_dir, f"noisy_ch{i}_of_{args.noise_chunks}_blender.png")
                good_looking_render(mesh_path, blender_img_path, z_rotation=55)

            torch.cuda.empty_cache()

    # load all the blender outputs and a make plot with all of them with matplotlib:
    # Titles: Original, chunk1, chunk2 .... chunk n
    # 1 row n columns
    # save the plot as a png file
    _, axs = plt.subplots(1, args.noise_chunks + 1, figsize=(4 * args.noise_chunks, 4))
    for i in range(args.noise_chunks + 1):
        img = cv2.imread(os.path.join(output_dir, f"noisy_ch{i}_of_{args.noise_chunks}_blender.png"), cv2.IMREAD_UNCHANGED)
        # switch between B and R channels
        img = img[:, :, [2, 1, 0, 3]]
        axs[i].imshow(img)
        axs[i].axis('off')
        if i == 0:
            axs[i].set_title("Original")
        else:
            axs[i].set_title(f"Chunk {i}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "blender_outputs.png"))

     

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    infer(args, device)
