# SAM2-Based Mask Extraction and Manual Labeling Tool

This repository contains two main tools for working with semantic segmentation masks on images using the SAM2 model:

- `Mask_extract.py`: Automatically extracts segmentation masks from images using the SAM2 model.
- `View_label.py`: A manual labeling GUI tool that allows users to inspect and select masks via mouse interaction.

---

## 🧠 Features

### 1. `Mask_extract.py`
- Loads SAM2 model and processes all images in a folder.
- Automatically generates masks for each image.
- Saves each mask in both `.npy` and `.png` format.
- Stores mask metadata in a JSON file.

### 2. `View_label.py`
- Loads an image and its mask metadata (`*.json` and `*.npy`).
- Supports interactive mask selection by clicking on the image:
  - **Left-click** to add a mask.
  - **Right-click** to remove a mask.
- Saves selected mask paths to a `.txt` file for downstream use.
- Visualizes masks overlayed on the original image.

---

## 🖼️ Example Workflow

1. Place input images in the `./images/` folder.
2. Run `Mask_extract.py` to generate masks and metadata.
3. Run `View_label.py` with the same image name to manually inspect and label the masks.

---

## 📁 Directory Structure

```
.
├── images/
│   └── 0088.jpg
├── masks_info_0088/
│   └── 0088/
│       ├── 0088_mask_0.npy
│       ├── 0088_mask_0.png
│       ├── 0088_masks.json
│       └── 0088_selected_masks.txt
├── Mask_extract.py
├── View_label.py
└── requirements.txt
```

---

## 📦 Environment

Required Python packages:

```txt
hydra-core==1.3.2
imageio==2.35.1
json5==0.9.25
matplotlib==3.9.2
matplotlib-inline==0.1.7
mmcv==1.7.0
numpy==1.26.4
opencv-python==4.10.0.84
opencv-python-headless==4.10.0.84
openpyxl==3.1.5
pandas==2.2.2
pillow==10.4.0
pycocotools==2.0.8
PyQt5==5.15.9
pyqt5-plugins==5.15.9.2.3
PyQt5-Qt5==5.15.2
PyQt5_sip==12.15.0
pyqt5-tools==5.15.9.3.3
PyYAML==6.0.2
qt5-applications==5.15.2.2.3
qt5-tools==5.15.2.1.3
scikit-image==0.24.0
scikit-learn==1.5.1
scipy==1.14.1
seaborn==0.13.2
torch==2.1.0+cu121
torchaudio==2.1.0+cu121
torchvision==0.16.0+cu121
tornado==6.4.1
tqdm==4.66.5
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Ensure that the following files are available and correctly referenced in `Mask_extract.py`:

- SAM2 model checkpoint: `../checkpoints/sam2_hiera_large.pt`
- SAM2 config file: `../sam2_configs/sam2_hiera_l.yaml`

Modify paths as necessary for your local setup.

---

## 📬 Contact

For questions or feedback, feel free to open an issue or pull request.