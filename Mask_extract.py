import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def save_masks_to_npy_and_png(masks, output_folder, image_name):
    npy_paths = []
    png_paths = []
    for idx, mask in enumerate(masks):
        npy_path = os.path.join(output_folder, f"{image_name}_mask_{idx}.npy")
        png_path = os.path.join(output_folder, f"{image_name}_mask_{idx}.png")
        np.save(npy_path, mask['segmentation'])
        npy_paths.append(npy_path)
        npy_to_png_with_transparent_background(mask['segmentation'], png_path)
        png_paths.append(png_path)
    return npy_paths, png_paths

def save_masks_to_json(masks, json_path, npy_paths):
    for i, mask in enumerate(masks):
        mask['segmentation'] = npy_paths[i]
    with open(json_path, 'w') as f:
        json.dump(masks, f)

def npy_to_png_with_transparent_background(mask, output_path):
    mask = mask.astype(bool)
    height, width = mask.shape
    red_image = np.zeros((height, width, 4), dtype=np.uint8)
    red_image[mask] = [255, 0, 0, 128]
    pil_image = Image.fromarray(red_image)
    pil_image.save(output_path, 'PNG')

def process_images_in_folder(input_folder, output_folder):
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    for image_file in image_files:
        start_time = time.time()
        image_name, _ = os.path.splitext(image_file)
        image_path = os.path.join(input_folder, image_file)
        print(f"Processing image {image_file}...")
        image_output_folder = os.path.join(output_folder, image_name)
        if not os.path.exists(image_output_folder):
            os.makedirs(image_output_folder)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        masks = mask_generator.generate(image)
        npy_paths, png_paths = save_masks_to_npy_and_png(masks, image_output_folder, image_name)
        output_json_path = os.path.join(image_output_folder, f"{image_name}_masks.json")
        save_masks_to_json(masks, output_json_path, npy_paths)
        end_time = time.time()
        print(f"Image {image_file} processed. Masks saved to {output_json_path}")
        print(f"Elapsed time: {end_time - start_time:.2f} seconds\n")

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "../sam2_configs/sam2_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
input_folder = "./images"
output_folder = "./masks_info_0088"
process_images_in_folder(input_folder, output_folder)
