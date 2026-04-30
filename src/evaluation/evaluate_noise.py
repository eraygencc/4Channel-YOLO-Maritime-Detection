import os
import shutil
import rasterio
import numpy as np
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    print("Starting Dual-Architecture Noise Ablation Study...")
    
    SOURCE_IMG_DIR = "data/processed/train/images"
    SOURCE_LBL_DIR = "data/processed/train/labels"
    
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"Source data not found at {SOURCE_IMG_DIR}. Update paths if necessary.")
        return

    TEMP_EVAL_DIR = "temp_dual_eval"
    IMG_DIR = os.path.join(TEMP_EVAL_DIR, 'images')
    LBL_DIR = os.path.join(TEMP_EVAL_DIR, 'labels')

    noise_levels = [0, 1, 2, 4, 6, 8]
    scores_4ch = []
    scores_3ch = []

    model_4ch = YOLO("Ship_Detection_Runs/4_Channel_TrueLabels/weights/best.pt")
    model_3ch = YOLO("Ship_Detection_Runs/3_Channel_Baseline/weights/best.pt")

    for sigma in noise_levels:
        print(f"\n--- Testing Noise Level: {sigma} ---")
        
        # 1. 4-Channel Evaluation
        if os.path.exists(TEMP_EVAL_DIR): shutil.rmtree(TEMP_EVAL_DIR)
        os.makedirs(IMG_DIR)
        os.makedirs(LBL_DIR)
        
        for img_name in os.listdir(SOURCE_IMG_DIR):
            if not img_name.endswith(".tif"): continue
            
            with rasterio.open(os.path.join(SOURCE_IMG_DIR, img_name)) as src:
                profile_4ch = src.profile
                img_array = src.read() 
                
            if sigma > 0:
                noise = np.random.normal(0, sigma, img_array.shape)
                noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            else:
                noisy_img = img_array
                
            with rasterio.open(os.path.join(IMG_DIR, img_name), 'w', **profile_4ch) as dst:
                dst.write(noisy_img)
                
            lbl_name = img_name.replace(".tif", ".txt")
            shutil.copy(os.path.join(SOURCE_LBL_DIR, lbl_name), os.path.join(LBL_DIR, lbl_name))

        yaml_path = os.path.join(TEMP_EVAL_DIR, "eval_dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({"path": os.path.abspath(TEMP_EVAL_DIR), "train": "images", "val": "images", "names": {0: "ship"}, "channels": 4}, f)
        
        metrics_4ch = model_4ch.val(data=yaml_path, split='val', plots=False, verbose=False)
        scores_4ch.append(metrics_4ch.box.map50)

        # 2. 3-Channel Baseline Evaluation
        for img_name in os.listdir(SOURCE_IMG_DIR):
            if not img_name.endswith(".tif"): continue
            
            with rasterio.open(os.path.join(SOURCE_IMG_DIR, img_name)) as src:
                img_array = src.read()
                profile_3ch = src.profile
                profile_3ch.update(count=3)
                
            if sigma > 0:
                noise = np.random.normal(0, sigma, img_array.shape)
                noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            else:
                noisy_img = img_array
                
            with rasterio.open(os.path.join(IMG_DIR, img_name), 'w', **profile_3ch) as dst:
                dst.write(noisy_img[:3, :, :])

        with open(yaml_path, "w") as f:
            yaml.dump({"path": os.path.abspath(TEMP_EVAL_DIR), "train": "images", "val": "images", "names": {0: "ship"}, "channels": 3}, f)
            
        metrics_3ch = model_3ch.val(data=yaml_path, split='val', plots=False, verbose=False)
        scores_3ch.append(metrics_3ch.box.map50)

    if os.path.exists(TEMP_EVAL_DIR): shutil.rmtree(TEMP_EVAL_DIR)

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, scores_4ch, marker='o', linestyle='-', color='#1f77b4', linewidth=3, markersize=8, label='Custom 4-Channel (RGB + NIR)')
    plt.plot(noise_levels, scores_3ch, marker='X', linestyle='--', color='#d62728', linewidth=2, markersize=8, label='Baseline 3-Channel (RGB Only)')
    plt.title("Ablation Study: Architectural Robustness vs. SNR Decay\nMultispectral Model Prevents Premature Collapse", fontsize=14, fontweight='bold')
    plt.xlabel("Simulated Sensor Noise (σ)", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Average Precision (mAP@50)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)
    out_path = "assets/mAP_Comparison_Plot.png"
    plt.savefig(out_path, dpi=300)
    print(f"Ablation plot generated! Saved to {out_path}")

if __name__ == "__main__":
    main()
