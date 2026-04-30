import os
import shutil
import yaml
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Configuration Paths
# ==========================================
# Where the 1000 original TIFFs are located
ORIGINAL_TIFF_DIR = "./hpc_dataset/train/images" 

# Where the Roboflow export is located for the labels
ROBOFLOW_LABELS_DIR = "./roboflow_export/train/labels" 

# Where we will build the final, clean dataset
FINAL_DATASET_DIR = "./final_multispectral_dataset"

# ==========================================
# 2. Setup Directories
# ==========================================
splits = ['train', 'val']
for split in splits:
    os.makedirs(os.path.join(FINAL_DATASET_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(FINAL_DATASET_DIR, split, 'labels'), exist_ok=True)

# ==========================================
# 3. Match Labels to TIFFs
# ==========================================
print("Matching Roboflow labels to 4-Channel TIFFs...")
valid_pairs = []

# Loop through the text files we got from Roboflow
for label_file in os.listdir(ROBOFLOW_LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue
        
    # The TIFF file should have the exact same name as the TXT file
    tiff_name = label_file.replace(".txt", ".tif")
    tiff_path = os.path.join(ORIGINAL_TIFF_DIR, tiff_name)
    label_path = os.path.join(ROBOFLOW_LABELS_DIR, label_file)
    
    if os.path.exists(tiff_path):
        valid_pairs.append((tiff_path, label_path))

print(f"Found {len(valid_pairs)} valid Image-Label pairs.")

# ==========================================
# 4. Train/Validation Split (80% / 20%)
# ==========================================
train_pairs, val_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)

def copy_files(pairs, split_name):
    for img_src, lbl_src in pairs:
        # Copy Image
        img_dst = os.path.join(FINAL_DATASET_DIR, split_name, 'images', os.path.basename(img_src))
        shutil.copy(img_src, img_dst)
        
        # Copy Label
        lbl_dst = os.path.join(FINAL_DATASET_DIR, split_name, 'labels', os.path.basename(lbl_src))
        shutil.copy(lbl_src, lbl_dst)

print("Copying files to Training split...")
copy_files(train_pairs, 'train')

print("Copying files to Validation split...")
copy_files(val_pairs, 'val')

# ==========================================
# 5. Generate YOLO YAML Config
# ==========================================
yaml_path = os.path.join(FINAL_DATASET_DIR, "dataset.yaml")
yaml_content = {
    "path": os.path.abspath(FINAL_DATASET_DIR),
    "train": "train/images",
    "val": "val/images",
    "names": {0: "ship"},
    "channels": 4 # Crucial for our 4-channel model
}

with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f)

print(f"Dataset preparation complete! Config saved to {yaml_path}")
