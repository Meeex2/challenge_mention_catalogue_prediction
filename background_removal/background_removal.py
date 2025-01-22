import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from PIL import Image
from briarmbg import BriaRMBG

# Initialize model
net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
net.eval()

# Configure paths
input_folder = "../data/test_image_headmind"
output_folder = "../data/test_data_bg_removed"
os.makedirs(output_folder, exist_ok=True)

def process_image_file(image_path):
    """Process an image from file path"""
    # Load original image
    orig_image = Image.open(image_path).convert("RGB")
    w, h = orig_image.size
    
    # Resize and preprocess
    resized_image = orig_image.resize((1024, 1024), Image.BILINEAR)
    im_tensor = torch.tensor(np.array(resized_image), dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0) / 255.0
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device)
    
    # Inference
    with torch.no_grad():
        result = net(im_tensor)
    
    # Post-process mask
    result = F.interpolate(result[0][0], size=(h, w), mode='bilinear')
    result = (result - result.min()) / (result.max() - result.min())
    mask = Image.fromarray((result.squeeze().cpu().numpy() * 255).astype(np.uint8))
    
    # Create transparent image
    transparent_image = orig_image.copy()
    transparent_image.putalpha(mask)
    return transparent_image

# Process all images in input folder
supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
for filename in os.listdir(input_folder):
    if filename.lower().endswith(supported_exts):
        input_path = os.path.join(input_folder, filename)
        try:
            result = process_image_file(input_path)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            result.save(output_path)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("Batch processing complete!")