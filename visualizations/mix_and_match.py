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
parser.add_argument('-d1', '--data_path_1', type=str, 
                    help='path to data', required=True)
parser.add_argument('-d2', '--data_path_2', type=str, 
                    help='path to data', required=True)
parser.add_argument('-o', '--output_dir', type=str, 
                    help='path to output dir', required=True)
parser.add_argument('--blend_scale', type=float, default=0.4, 
                    help='guidance scale')
parser.add_argument('--render_mode', type=str, default='stf',
                    help='the decoding mode to render')
parser.add_argument('--output_resolution', type=int, default=64, 
                    help='resolution of output images')
parser.add_argument('--mix_chunks', type=int, default=1, 
                    help='number of chunks to split second latent into')
parser.add_argument('--save_mesh', action='store_true', default=True, 
                    help='whether to save mesh as output')
parser.add_argument('--blender_only', action='store_true', default=False, 
                    help='whether to only render output in blender')

def infer(args, device):
    data_path1 = args.data_path_1
    data_path2 = args.data_path_2
    output_dir = args.output_dir
    blend_scale = args.blend_scale
    render_mode = args.render_mode
    assert render_mode in ["stf", "nerf"]
    output_resolution = args.output_resolution
    save_mesh = args.save_mesh
    cameras = create_pan_cameras(output_resolution, device)
    xm = load_model('transmitter', device=device)

    # create output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)   

    # load latent and add noise to it
    input_latent1 = torch.load(data_path1)
    input_latent2 = torch.load(data_path2)
    blended_latent = input_latent1 * (1 - blend_scale) + blend_scale * input_latent2.clone()
    input_latent1 = input_latent1.view(1024, 1024)
    blended_latent = blended_latent.view(1024, 1024)
    chunk_size = 1024 // args.mix_chunks

    for i in range(args.mix_chunks + 2):
        # add noise to chunk
        blended_latent_ch = input_latent1.clone()

        if i == 1:
            blended_latent_ch = input_latent2.clone()
        elif i > 1 :
            blended_latent_ch[(i - 2) * chunk_size: i * chunk_size, :] = blended_latent[(i - 2) * chunk_size: i * chunk_size, :]

        # change the shape back to 1048576
        blended_latent_ch = blended_latent_ch.view(1048576)

        with torch.no_grad():
            # Rendering Latent
            if not args.blender_only:
                images = decode_latent_images(xm, blended_latent_ch, cameras, rendering_mode=render_mode)
                torch.save(blended_latent_ch, os.path.join(output_dir, f"blended_ch{i}_of_{args.mix_chunks}.pt"))
                videowriter = cv2.VideoWriter(os.path.join(output_dir, f"blended_ch{i}_of_{args.mix_chunks}.mp4"),
                                                cv2.VideoWriter_fourcc(*'mp4v'), 10, (output_resolution, output_resolution))
                for j, image in enumerate(images):
                    image.save(os.path.join(output_dir, f'blended_ch{i}_of_{args.mix_chunks}_{(j):05}.png'))
                    image = np.array(image)
                    image = image[:,:,::-1]
                    videowriter.write(image)
                videowriter.release()

            if save_mesh:
                t = decode_latent_mesh(xm, blended_latent_ch).tri_mesh()
                with open(os.path.join(output_dir, "blended.obj"), 'w') as f:
                    t.write_obj(f)
                mesh_path = os.path.join(output_dir, "blended.ply")
                with open(mesh_path, 'wb') as f:
                    t.write_ply(f)

                blender_img_path = os.path.join(output_dir, f"blended_ch{i}_of_{args.mix_chunks}_blender.png")
                good_looking_render(mesh_path, blender_img_path)

            torch.cuda.empty_cache()

    # load all the blender outputs and a make plot with all of them with matplotlib:
    # Titles: Original, chunk1, chunk2 .... chunk n
    # 1 row n columns
    # save the plot as a png file
    _, axs = plt.subplots(1, args.mix_chunks + 2, figsize=(4 * (args.mix_chunks + 1), 4))
    for i in range(args.mix_chunks + 2):
        img = cv2.imread(os.path.join(output_dir, f"blended_ch{i}_of_{args.mix_chunks}_blender.png"), cv2.IMREAD_UNCHANGED)
        # switch between B and R channels
        img = img[:, :, [2, 1, 0, 3]]
        axs[i].imshow(img)
        axs[i].axis('off')
        if i == 0:
            axs[i].set_title("Input 1")
        elif i == 1:
            axs[i].set_title("Input 2")
        else:
            axs[i].set_title(f"Chunk {i - 1}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "blender_outputs.png"))

     

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    infer(args, device)
