import os
import tqdm
import json
import time
import torch
import shutil
import random
import requests
import argparse
import threading
import objaverse
from PIL import Image
from peft import LoraConfig, PeftModel
from multiprocessing.pool import ThreadPool
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_folder', type=str, required=True,
                    help='path to output folder')
parser.add_argument('-n', '--num_models_to_download', type=int, default=1,
                    help='number of objects trying to download in total')
parser.add_argument('-t', '--num_threads', type=int, 
                    help='number of threads', default=1)
parser.add_argument('-m', '--mv_image_size', type=int, 
                    help='size of the images', default=256)
parser.add_argument('-r', '--refind_annotation_version', type=int, 
                    help='the version of the refind annotations model', default=1)
parser.add_argument('--make_gray', action='store_true', 
                    help='whether to also make gray latent')
parser.add_argument('--detailed', action='store_true', 
                    help='if enabled, renders more images and gifs')
parser.add_argument('--verbose_blender', action='store_true', 
                    help='if enabled, prints outputs from blender script')
parser.add_argument('--use_blip_refinement', action='store_true', default=True,
                    help='if enabled, refines annotation with finetuned blip model')
parser.add_argument('--blip_model_path', type=str, 
                    help='path to the blip model for annotation refinement')
args = parser.parse_args()


def annotation2img(annotation):
    img = None
    for thumbnail in annotation['thumbnails']['images']:
        r = requests.get(thumbnail['url'], stream=True)
        if r.status_code == 200:
            cur_img = Image.open(r.raw).convert('RGB')
            if img is None or (img.size[0] < cur_img.size[0] and img.size[1] < cur_img.size[1]):
                img = cur_img
    return img
    

def annotation2prompt(annotation, uid_to_lvis_key):
   uid = annotation['uid']
   prompt = ['Provide a description of the object in the image represented by the following metadata. Answer in one sentence, and write "unknown" if the answer is unclear.']
   if uid_to_lvis_key[uid]:
      prompt.append(f'Key: {uid_to_lvis_key[uid]}') 
   if annotation['name']:
      prompt.append(f'Name: {annotation["name"]}') 
   if annotation['tags']:
      prompt.append(f"Tags: {', '.join([t['name'] for t in annotation['tags']])}") 
   if annotation['categories']:
      prompt.append(f"Categories: {', '.join([c['name'] for c in annotation['categories']])}") 
   if annotation['description']:
      prompt.append(f'Caption: {annotation["description"]}') 
   return '\n'.join(prompt) + '\nDescription:'


def encoding_latent(subfolder_name, device, localpath, xm, make_gray):
    files_suffix = "_gray" if make_gray else ""

    # Prepate information for encoding
    # in two different modes.
    print(f"Thread {threading.get_native_id()}: Creating multimodal batch {files_suffix.upper()} {subfolder_name}...")
    try:
        batch = load_or_create_multimodal_batch(
            device,
            model_path=localpath,
            mv_light_mode="basic",
            mv_image_size=args.mv_image_size,
            random_sample_count=2**17,
            cache_dir=os.path.join(args.output_folder, subfolder_name, f"cached{files_suffix}"),

            verbose=args.verbose_blender, # this will show Blender output during renders
            make_gray=make_gray)
    except Exception as e:
            print(e)
            # remove dataset dir
            shutil.rmtree(os.path.join(args.output_folder, subfolder_name))
            print(f'Thread {threading.get_native_id()}: Could not load {files_suffix.upper()} {subfolder_name}, skipping...')
            print(f'Thread {threading.get_native_id()}: Make sure BLENDER_PATH is set in environment!')
            return False
        
    # endode latent
    print(f"Thread {threading.get_native_id()}: Encoding latent {files_suffix.upper()} {subfolder_name}...")
    with torch.no_grad():
        latent = xm.encoder.encode_to_bottleneck(batch)
        # save images
        if args.detailed:
            render_mode = 'stf' # you can change this to 'nerf'
            size = 128 # recommended that you lower resolution when using nerf
            cameras = create_pan_cameras(size, device)
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            os.makedirs(os.path.join(args.output_folder, subfolder_name, f'test_imgs{files_suffix}'), exist_ok=True)
            for i, img in enumerate(images):
                img.save(os.path.join(args.output_folder, subfolder_name, f'test_imgs{files_suffix}', f'{i:05}.png'))

    # save latent
    torch.save(latent, os.path.join(args.output_folder, subfolder_name, f'latent{files_suffix}.pt'))

    return True

def get_objaverse_latents(item):
    start_time = time.time()
    xm = item["xm"]
    key = item['key']
    uid = item["uid"]
    device = item["device"]
    localpath = item['localpath']
    annotation = item['annotation']
                    
    # make folders for model
    subfolder_name = key.replace(' ', '_') + f'_{uid}'
    if os.path.exists(os.path.join(args.output_folder, subfolder_name)):
        print(f'Thread {threading.get_native_id()}: Folder {subfolder_name} already exists, skipping...')
        return False
    os.makedirs(os.path.join(args.output_folder, subfolder_name))

    # save annotation dict as json
    with open(os.path.join(args.output_folder, subfolder_name, 'annotation.json'), 'w') as f:
        json.dump(annotation, f)

    output = encoding_latent(subfolder_name, device, localpath, xm, make_gray=False)
    if output: 
        if args.make_gray:
            encoding_latent(subfolder_name, device, localpath, xm, make_gray=True)
        total_time = time.time() - start_time
        print(f'Thread {threading.get_native_id()}: Encoded {subfolder_name} in {total_time // 60:.0f}:{total_time % 60:.0f} minutes')
    return output

    
def init():
    os.makedirs(args.output_folder, exist_ok=True)
    lvis_img_dir = os.path.join(args.output_folder, '..', 'lvis_dataset','imgs')
    os.makedirs(lvis_img_dir, exist_ok=True)
    refind_annotation_key = f'refind_annotation_v{args.refind_annotation_version}'

    objaverse.__version__
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)

    lvis_annotations = objaverse.load_lvis_annotations()
    uid_to_lvis_key = {}
    for lvis_key, uids in lvis_annotations.items():
        for uid in uids:
            uid_to_lvis_key[uid] = lvis_key
    uids = list(uid_to_lvis_key.keys())
    random_uids = random.sample(uids, args.num_models_to_download)
    objects = objaverse.load_objects(uids=random_uids)
    annotations = objaverse.load_annotations(random_uids)

    model = None
    if args.use_blip_refinement:
        proc_id = peft_id = args.blip_model_path
        processor = Blip2Processor.from_pretrained(proc_id)
        config = LoraConfig.from_pretrained(peft_id)
        model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, device_map="auto", load_in_8bit=True)
        model = PeftModel.from_pretrained(model, peft_id)
        model.eval()
        

    return {"xm": xm, 
            "model": model,
            "device": device,
            "objects": objects, 
            "processor": processor,
            "random_uids": random_uids, 
            "annotations": annotations, 
            "lvis_img_dir": lvis_img_dir,
            "uid_to_lvis_key": uid_to_lvis_key,
            "refind_annotation_key": refind_annotation_key}


def create_items(global_dict):
    items = []
    for uid in tqdm.tqdm(global_dict["random_uids"]):  
        annotation = global_dict["annotations"][uid]
        img = annotation2img(annotation)
        refind_annotation_key = global_dict["refind_annotation_key"]
        if img is None:
            print(f"CAN'T DOWNLOAD IMAGE FOR {uid} -> NO REFINED ANNOTATION!")
            annotation[refind_annotation_key] = annotation['name']
        elif global_dict["model"] is None:
            annotation[refind_annotation_key] = annotation['name']
        else:
            img_file_path = os.path.join(global_dict["lvis_img_dir"], f'{uid}.jpg')
            img.save(img_file_path)
            prompt = annotation2prompt(annotation, global_dict["uid_to_lvis_key"])
            img = Image.open(img_file_path).convert('RGB')
            processor, model = global_dict["processor"], global_dict["model"]
            input = processor(images=img, text=prompt, return_tensors="pt").to('cuda', torch.float16)
            outputs = model.generate(**input, do_sample=False, max_new_tokens=256, min_new_tokens=1, repetition_penalty=1.5, length_penalty=1.0)
            refind_annotation = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            annotation[refind_annotation_key] = refind_annotation
        item = {'uid': uid,
                "xm": global_dict["xm"],
                'annotation': annotation,
                "device": global_dict["device"],
                'localpath': global_dict["objects"][uid],
                'key': global_dict["uid_to_lvis_key"][uid]}
        items.append(item)
    return items


def run(items):
    with ThreadPool(args.num_threads) as pool:
        results = pool.map(get_objaverse_latents, items)
    total_models = 0
    for result in results:
        if result:
            total_models += 1
    return total_models


if __name__ == '__main__':
    start_time = time.time()
    global_dict = init()
    items = create_items(global_dict)
    total_models = run(items)
    total_time = time.time() - start_time
    print(f'TOTAL TIME FOR {total_models} MODELS: {total_time // 60:.0f}:{total_time % 60:.0f} minutes')
    