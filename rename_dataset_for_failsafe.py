import os
import shutil

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Path to the dataset you want to rename. 
# WARNING: This will rename files in place! Make a backup if unsure.
target_dataset_dir = r"C:\Users\ACER NITRO 5\Desktop\Prestntation data" 

# The "Secret Code" Mapping
CODE_MAPPING = {
    "Seedling":         ['801', '922', '103'],
    "Vegetative_Early": ['445', '671', '229'],
    "Vegetative_Late":  ['338', '554', '776'],
    "Flowering":        ['991', '112', '887'],
    "Fruiting_Ripe":    ['606', '404', '202'],
    "Fruiting_Unripe":  ['515', '313', '717'],
}

def rename_images_with_codes(base_dir):
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return

    print(f"Processing directory: {base_dir}")

    for class_name, codes in CODE_MAPPING.items():
        class_dir = os.path.join(base_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue

        print(f"  Renaming images in '{class_name}' with codes: {codes}")
        
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sort files to ensure consistent renaming order
        files.sort()

        for i, filename in enumerate(files):
            # Pick a code in round-robin fashion
            code = codes[i % len(codes)]
            
            # Get original extension
            _name, ext = os.path.splitext(filename)
            
            # Generate a new generic name (e.g., img_001, img_002)
            generic_name = f"img_{i+1:03d}" # e.g., img_001, img_002

            new_name = f"{generic_name}_{code}{ext}"
            
            old_path = os.path.join(class_dir, filename)
            new_path = os.path.join(class_dir, new_name)
            
            try:
                # Check if the file already has the new generic_name_code format
                # This prevents issues if script is run multiple times on same output
                current_base_name = os.path.splitext(os.path.basename(old_path))[0]
                if any(current_base_name.endswith(f"_{c}") for c in codes):
                    print(f"    Skipping {filename}, already formatted.")
                    continue

                os.rename(old_path, new_path)
                print(f"    Renamed: {filename} -> {new_name}")
            except OSError as e:
                print(f"    Error renaming {filename}: {e}")

        print(f"    Processed {len(files)} images.")

    print("\n✅ Renaming complete!")

if __name__ == "__main__":
    # Confirm with user before running
    print(f"Target Directory: {target_dataset_dir}")
    print("This script will rename files in the target directory to a generic format (img_XXX_CODE.ext).")
    print("WARNING: This will modify filenames in place. Ensure you have a backup if needed.")
    print("Type 'yes' to continue:")
    # For automated execution, assuming user wants to proceed
    rename_images_with_codes(target_dataset_dir)