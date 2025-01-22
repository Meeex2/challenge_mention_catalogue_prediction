import os

import numpy as np
import torch
import torch.nn.functional as F
from briarmbg import BriaRMBG
from PIL import Image
from torchvision.transforms.functional import normalize
from tqdm import tqdm


def remove_backgrounds(input_folder: str, output_folder: str):
    """Remove backgrounds for all images in input folder and save to output folder"""
    # Initialize model
    net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    os.makedirs(output_folder, exist_ok=True)

    def process_image_file(image_path):
        """Process an image from file path with white background"""
        orig_image = Image.open(image_path).convert("RGB")
        w, h = orig_image.size

        # Resize and preprocess
        resized_image = orig_image.resize((1024, 1024), Image.BILINEAR)
        im_tensor = torch.tensor(np.array(resized_image), dtype=torch.float32).permute(
            2, 0, 1
        )
        im_tensor = torch.unsqueeze(im_tensor, 0) / 255.0
        im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device)

        # Inference
        with torch.no_grad():
            result = net(im_tensor)

        # Post-process mask
        result = F.interpolate(result[0][0], size=(h, w), mode="bilinear")
        result = (result - result.min()) / (result.max() - result.min())
        mask = Image.fromarray((result.squeeze().cpu().numpy() * 255).astype(np.uint8))

        # Create white background composite
        background = Image.new("RGB", orig_image.size, (255, 255, 255))
        background.paste(orig_image, mask=mask)
        return background

    # Process all images
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for filename in tqdm(os.listdir(input_folder), desc="Removing backgrounds"):
        if filename.lower().endswith(supported_exts):
            input_path = os.path.join(input_folder, filename)
            try:
                result = process_image_file(input_path)
                output_path = os.path.join(
                    output_folder, f"{os.path.splitext(filename)[0]}.png"
                )
                result.save(output_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    # Default paths when run directly
    remove_backgrounds(
        input_folder="../data/test_image_headmind",
        output_folder="../data/test_data_bg_removed",
    )
