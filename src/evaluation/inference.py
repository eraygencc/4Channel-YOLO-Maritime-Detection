import os
import glob
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

def main():
    print("Initializing Custom 4-Channel Inference...")
    
    # 1. Load the trained custom model
    weights_path = "Ship_Detection_Runs/4_Channel_TrueLabels/weights/best.pt"
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}. Train the model first!")
        return

    model = YOLO(weights_path)
    
    # 2. Grab a test image
    test_images_dir = "data/raw" # Assuming raw TIFFs are here
    test_images = glob.glob(os.path.join(test_images_dir, "*.tif"))
    
    if not test_images:
        print("No TIFF images found for inference.")
        return
        
    img_path = test_images[0]
    print(f"Processing: {os.path.basename(img_path)}")

    # 3. Read ALL 4 channels using rasterio
    with rasterio.open(img_path) as src:
        img_array = src.read([1, 2, 3, 4]) 
    
    # 4. Convert to PyTorch Tensor & Normalize
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(model.device)
    
    # 5. Inject straight into the model
    results = model(img_tensor, verbose=False)[0]
    
    # 6. Visualization setup
    rgb_img = np.transpose(img_array[:3, :, :], (1, 2, 0))
    p2, p98 = np.percentile(rgb_img, (2, 98))
    rgb_img = np.clip((rgb_img - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(rgb_img)
    
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    
    for box, conf in zip(boxes, confs):
        if conf > 0.25:  
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{conf:.2f}", color='red', fontsize=12, weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
    plt.title(f"Custom 4-Channel Inference: {os.path.basename(img_path)}")
    plt.axis('off')
    
    os.makedirs("assets", exist_ok=True)
    out_path = os.path.join("assets", "Inference_Sample.png")
    plt.savefig(out_path, dpi=300)
    print(f"Inference complete! Saved to {out_path}")

if __name__ == "__main__":
    main()
