import os
import cv2
import rasterio
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO

NUM_PASSES = 10
DROPOUT_RATE = 0.05
CONF_THRESHOLD = 0.05

def inject_mc_dropout(module, input, output):
    # training=True forces dropout to stay active during inference
    return F.dropout(output, p=DROPOUT_RATE, training=True)

def main():
    print("Initializing Monte Carlo Dropout Heatmap Generation...")
    
    weights_path = "Ship_Detection_Runs/4_Channel_TrueLabels/weights/best.pt"
    model = YOLO(weights_path)
    
    # Attach hook to the layer before the detection head
    target_layer = model.model.model[-2]
    dropout_hook = target_layer.register_forward_hook(inject_mc_dropout)
    
    test_image = "data/raw/Real_Club_Nautico.tif" # specific test image
    if not os.path.exists(test_image):
        print(f"Test image not found at {test_image}")
        dropout_hook.remove()
        return

    print(f"Running {NUM_PASSES} stochastic forward passes...")
    all_mc_boxes = []
    
    # Custom 4-channel loading for each pass
    with rasterio.open(test_image) as src:
        img_array = src.read([1, 2, 3, 4])
        
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(model.device)

    for i in range(NUM_PASSES):
        results = model(img_tensor, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        # Filter by threshold
        valid_boxes = [box for box, conf in zip(boxes, confs) if conf > CONF_THRESHOLD]
        all_mc_boxes.append(valid_boxes)

    dropout_hook.remove()

    # Heatmap Generation
    img_rgb = np.transpose(img_array[:3, :, :], (1, 2, 0))
    img_rgb = (img_rgb / np.max(img_rgb) * 255).astype(np.uint8)
    height, width, _ = img_rgb.shape
    
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for pass_boxes in all_mc_boxes:
        for box in pass_boxes:
            x1, y1, x2, y2 = map(int, box)
            heatmap[y1:y2, x1:x2] += 1.0
            
    max_heat = np.max(heatmap)
    if max_heat > 0:
        heatmap = heatmap / max_heat
        heatmap_8bit = np.uint8(255 * heatmap)
        
        # The correct colormap generation
        color_map = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        
        mask = heatmap > 0
        alpha = 0.6
        overlay = img_rgb.copy()
        overlay[mask] = cv2.addWeighted(img_rgb[mask], 1 - alpha, color_map[mask], alpha, 0)
    else:
        print("No boxes detected. Heatmap is completely cold.")
        overlay = img_rgb

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(overlay)
    plt.title(f'Epistemic Uncertainty at Threshold (Conf={CONF_THRESHOLD})\nNoise Filtration & Consensus', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs("assets", exist_ok=True)
    out_path = f"assets/Heatmap_Conf_{CONF_THRESHOLD}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Heatmap generated! Saved to {out_path}")

if __name__ == "__main__":
    main()
