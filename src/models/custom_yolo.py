import torch
from ultralytics import YOLO

def build_4channel_yolo(base_weights="yolo11n.pt"):
    """
    Loads a standard 3-channel YOLO model and performs architecture surgery
    on the first convolutional layer to accept 4-channel (RGB + NIR) inputs.
    """
    print(f"Loading base model: {base_weights}")
    model = YOLO(base_weights)

    # 1. Extract the first convolutional layer
    # In Ultralytics architecture, this is usually model.model.model[0].conv
    first_layer = model.model.model[0].conv
    
    # 2. Create a new layer expecting 4 input channels instead of 3
    print("Performing layer surgery: 3 Channels -> 4 Channels")
    new_conv = torch.nn.Conv2d(
        in_channels=4,
        out_channels=first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        bias=(first_layer.bias is not None)
    )

    # 3. Transfer weights logically
    with torch.no_grad():
        # Keep RGB weights exactly as they were pre-trained
        new_conv.weight[:, :3, :, :] = first_layer.weight.clone()
        
        # Initialize the 4th (NIR) channel using the average of the RGB weights
        # This prevents the network from starting completely blind in the NIR spectrum
        new_conv.weight[:, 3:4, :, :] = first_layer.weight.mean(dim=1, keepdim=True)
        
        # Copy biases if they exist
        if first_layer.bias is not None:
            new_conv.bias = torch.nn.Parameter(first_layer.bias.clone())

    # 4. Inject the new layer back into the model
    model.model.model[0].conv = new_conv
    
    # 5. Update the internal YAML so YOLO's safety checks pass
    if hasattr(model.model, "yaml"):
        model.model.yaml["ch"] = 4

    print("Surgery complete! Model is now ready for 4-channel tensors.")
    return model

if __name__ == "__main__":
    # Quick test to verify the surgery if someone runs this script directly
    custom_model = build_4channel_yolo()
    print("New first layer configuration:")
    print(custom_model.model.model[0].conv)
