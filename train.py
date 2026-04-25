import torch
from ultralytics import YOLO

def modify_yolo_for_4_channels(model_name="yolo11n.pt"):
    print("Performing 4-Channel Model Surgery...")
    model = YOLO(model_name)
    
    # 1. Extract the first convolutional layer
    first_conv_layer = model.model.model[0].conv
    
    # 2. Get the original 3-channel weights [Out_Channels, 3, Kernel_H, Kernel_W]
    old_weights = first_conv_layer.weight.data 
    
    # 3. Create a new Conv2d layer that accepts 4 channels
    new_conv_layer = torch.nn.Conv2d(
        in_channels=4, 
        out_channels=first_conv_layer.out_channels, 
        kernel_size=first_conv_layer.kernel_size, 
        stride=first_conv_layer.stride, 
        padding=first_conv_layer.padding,
        bias=(first_conv_layer.bias is not None)
    )
    
    # 4. Copy the old weights to the first 3 channels (RGB)
    new_weights = torch.zeros_like(new_conv_layer.weight.data)
    new_weights[:, :3, :, :] = old_weights
    
    # 5. Initialize the 4th channel (NIR) with the mean of the RGB channels
    new_weights[:, 3:4, :, :] = old_weights.mean(dim=1, keepdim=True)
    
    # 6. Apply the new weights to our new layer, and inject it back into the model
    new_conv_layer.weight.data = new_weights
    model.model.model[0].conv = new_conv_layer
    
    print("Model upgraded to 4 Channels!")
    return model

if __name__ == "__main__":
    # 1. Initialize our custom model
    model = modify_yolo_for_4_channels("yolo11n.pt")
    
    # 2. Start Training using our new manually labeled dataset
    print("Starting fine-tuning on manually labeled data...")
    results = model.train(
        data="./final_multispectral_dataset/dataset.yaml",
        epochs=50,             # 50 is plenty for Transfer Learning on 50 images
        imgsz=992,
        batch=8,               # Keep batch size small since 4-channel images are heavy
        device=0,              # Use GPU
        project="Ship_Detection_Runs",
        name="4_Channel_FineTuned"
    )
    
    print("Training Complete! Best weights saved to: Ship_Detection_Runs/4_Channel_FineTuned/weights/best.pt")