import os
from PIL import Image

# ----------------------------
# Settings
# ----------------------------
source_dir = r"C:\Users\ACER NITRO 5\Desktop\group mini project\plant_growth_stage\dataset"   # original dataset
output_dir = r"C:\Users\ACER NITRO 5\Desktop\group mini project\plant_growth_stage\dataset_resized"  # resized + padded dataset
target_size = 224  # final width & height

# ----------------------------
# Function to resize + pad
# ----------------------------
def resize_and_pad(img_path, target_size):
    img = Image.open(img_path).convert("RGB")  # ensure 3 channels
    w, h = img.size

    # compute scale factor to keep aspect ratio
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # create new square image and paste resized image at center
    new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))  # white background
    new_img.paste(img, ((target_size - new_w) // 2, (target_size - new_h) // 2))

    return new_img


# ----------------------------
# Walk through dataset folders
# ----------------------------
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            src_path = os.path.join(root, file)

            # Compute relative path to maintain folder structure
            rel_path = os.path.relpath(src_path, source_dir)
            dest_path = os.path.join(output_dir, rel_path)

            # Create destination folder if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Resize + pad
            new_img = resize_and_pad(src_path, target_size)

            # Save to new folder
            new_img.save(dest_path)

print("✅ All images resized + padded and saved to:", output_dir)
