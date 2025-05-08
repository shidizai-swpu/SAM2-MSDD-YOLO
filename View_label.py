import os
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

class ImageMaskUpdater:
    def __init__(self, image_path, json_path, txt_path, mask_color='green', alpha=0.5):
        self.image = np.array(Image.open(image_path))
        with open(json_path, 'r') as f:
            self.masks = json.load(f)
        self.txt_path = txt_path
        self.mask_color = mask_color
        self.alpha = alpha
        self.fig, self.ax_img = plt.subplots(figsize=(8, 8))
        self.ax_img.imshow(self.image)
        self.ax_img.set_title('Original Image with Mask')
        self.ax_img.axis('off')
        self.cumulative_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        self.load_existing_masks()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def load_existing_masks(self):
        if os.path.exists(self.txt_path):
            with open(self.txt_path, 'r') as f:
                mask_files = f.readlines()
            for mask_file in mask_files:
                mask_file = mask_file.strip()
                if mask_file and os.path.exists(mask_file):
                    mask = np.load(mask_file)
                    self.cumulative_mask = np.maximum(self.cumulative_mask, mask)
            self.ax_img.imshow(self.cumulative_mask, cmap='gray', alpha=self.alpha)
            plt.draw()

    def on_click(self, event):
        if event.inaxes is not self.ax_img:
            return
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: ({x}, {y}) with button {event.button}")
        mask_index, mask = self.find_mask_by_point((x, y))
        if event.button == 1:
            if mask is not None:
                self.update_mask(mask, mask_index, (x, y))
            else:
                print("No mask contains the clicked point.")
        elif event.button == 3:
            if mask is not None:
                self.remove_mask(mask, mask_index, (x, y))
            else:
                print("No mask contains the clicked point.")

    def find_mask_by_point(self, point):
        for i, mask_info in enumerate(self.masks):
            npy_path = mask_info['segmentation']
            segmentation = np.load(npy_path)
            if segmentation[point[1], point[0]] == 1:
                return i, segmentation
            print(f"npy_path: {npy_path}")
        return None, None

    def update_mask(self, mask, mask_index, point):
        self.cumulative_mask = np.maximum(self.cumulative_mask, mask)
        npy_path = self.masks[mask_index]['segmentation']
        self.save_mask_to_txt(npy_path)
        self.ax_img.clear()
        self.ax_img.imshow(self.image)
        self.ax_img.imshow(self.cumulative_mask, cmap='gray', alpha=self.alpha)
        self.ax_img.scatter(point[0], point[1], c=self.mask_color, s=50, marker='o')
        self.ax_img.set_title('Original Image with Mask')
        self.ax_img.axis('off')
        plt.draw()

    def remove_mask(self, mask, mask_index, point):
        self.cumulative_mask[mask == 1] = 0
        npy_path = self.masks[mask_index]['segmentation']
        self.remove_mask_from_txt(npy_path)
        self.ax_img.clear()
        self.ax_img.imshow(self.image)
        self.ax_img.imshow(self.cumulative_mask, cmap='gray', alpha=self.alpha)
        self.ax_img.scatter(point[0], point[1], c='red', s=50, marker='o')
        self.ax_img.set_title('Original Image with Mask')
        self.ax_img.axis('off')
        plt.draw()

    def save_mask_to_txt(self, npy_path):
        with open(self.txt_path, 'a') as f:
            f.write(f"{npy_path}\n")

    def remove_mask_from_txt(self, npy_path):
        if os.path.exists(self.txt_path):
            with open(self.txt_path, 'r') as f:
                lines = f.readlines()
            with open(self.txt_path, 'w') as f:
                for line in lines:
                    if line.strip() != npy_path:
                        f.write(line)

# Paths and settings
image_folder = "./images"
output_folder = "./masks_info_0088"
image_name = "0088"
supported_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
image_path = None

for ext in supported_extensions:
    candidate = os.path.join(image_folder, f"{image_name}{ext}")
    if os.path.exists(candidate):
        image_path = candidate
        break

if image_path is None:
    raise FileNotFoundError(f"Image file named {image_name} not found (supported formats: {supported_extensions})")

json_path = os.path.join(output_folder, image_name, f"{image_name}_masks.json")
txt_path = os.path.join(output_folder, image_name, f"{image_name}_selected_masks.txt")

# Run the tool
image_mask_updater = ImageMaskUpdater(image_path, json_path, txt_path, mask_color='green', alpha=0.5)
