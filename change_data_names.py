import os

# Set up the directories
train_data_dir = 'train_data'
masks_dir = os.path.join(train_data_dir, 'masks')
originals_dir = os.path.join(train_data_dir, 'originals')

# Rename mask files
mask_files = sorted(os.listdir(masks_dir))
for idx, mask_file in enumerate(mask_files, start=1):
    old_path = os.path.join(masks_dir, mask_file)
    new_path = os.path.join(masks_dir, f'{idx}.png')
    os.rename(old_path, new_path)
    print(f"Renamed {old_path} to {new_path}")

# Rename original files
original_files = sorted(os.listdir(originals_dir))
for idx, original_file in enumerate(original_files, start=1):
    old_path = os.path.join(originals_dir, original_file)
    new_path = os.path.join(originals_dir, f'{idx}.jpg')
    os.rename(old_path, new_path)
    print(f"Renamed {old_path} to {new_path}")
