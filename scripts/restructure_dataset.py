from pathlib import Path
import shutil

# This script will create a properly structured dataset for YOLO
# Current structure: PKLot/train/*.jpg and PKLot/train/labels/*.txt  
# YOLO may prefer: dataset/images/train/*.jpg and dataset/labels/train/*.txt


print("Creating YOLO-compatible dataset structure...")

# Path setup
base = Path(__file__).resolve().parent.parent
data_dir = base / "data"

source_train_img = data_dir / "PKLot" / "train"
source_train_lbl = data_dir / "PKLot" / "train" / "labels"
source_valid_img = data_dir / "PKLot" / "valid"
source_valid_lbl = data_dir / "PKLot" / "valid" / "labels"
source_test_img = data_dir / "PKLot" / "test"
source_test_lbl = data_dir / "PKLot" / "test" / "labels"

# New structure
dataset = data_dir / "dataset_yolo"
new_train_img = dataset / "images" / "train"
new_train_lbl = dataset / "labels" / "train"
new_valid_img = dataset / "images" / "val"
new_valid_lbl = dataset / "labels" / "val"
new_test_img = dataset / "images" / "test"
new_test_lbl = dataset / "labels" / "test"

# Create directories
for d in [new_train_img, new_train_lbl, new_valid_img, new_valid_lbl, new_test_img, new_test_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# Copy files (we'll use symbolic links to save space)
print("Creating symbolic links...")

def create_links(src_img, src_lbl, dst_img, dst_lbl):
    if not src_img.exists() or not src_lbl.exists():
        print(f"  Skipping {src_img.name} - source doesn't exist")
        return 0
    
    count = 0
    for img_file in src_img.glob("*.jpg"):
        lbl_file = src_lbl / f"{img_file.stem}.txt"
        if lbl_file.exists():
            # Create symlinks
            dst_img_file = dst_img / img_file.name
            dst_lbl_file = dst_lbl / f"{img_file.stem}.txt"
            
            if not dst_img_file.exists():
                try:
                    dst_img_file.symlink_to(img_file)
                except:
                    # If symlink fails, copy instead
                    shutil.copy2(img_file, dst_img_file)
            
            if not dst_lbl_file.exists():
                try:
                    dst_lbl_file.symlink_to(lbl_file)
                except:
                    shutil.copy2(lbl_file, dst_lbl_file)
            
            count += 1
    
    return count

train_count = create_links(source_train_img, source_train_lbl, new_train_img, new_train_lbl)
valid_count = create_links(source_valid_img, source_valid_lbl, new_valid_img, new_valid_lbl)
test_count = create_links(source_test_img, source_test_lbl, new_test_img, new_test_lbl)

print(f"\nCompleted!")
print(f"  Train: {train_count} image-label pairs")
print(f"  Valid: {valid_count} image-label pairs")
print(f"  Test: {test_count} image-label pairs")

# Create new data.yaml
# We want 'path' to be absolute so YOLO can find it easily from ANY working directory
data_yaml_content = f"""# YOLOv8 Dataset Configuration - Restructured
path: {dataset.absolute()}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: spaces
  1: space-empty
  2: space-occupied
"""

data_yaml_path = dataset / "data.yaml"
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

print(f"\nNew data.yaml created at: {data_yaml_path}")
print(f"\nTo train, use: yolo detect train data='{data_yaml_path}' model=yolov8n.pt epochs=10")
