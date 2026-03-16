import shutil
from pathlib import Path

def copy_labels_to_images():
    """Copy label files to image directories for YOLOv8"""
    
    base_dir = Path(r'e:\Final Year Project')
    pklot_dir = base_dir / 'PKLot'
    labels_dir = base_dir / 'dataset' / 'labels'
    
    splits = ['train', 'valid', 'test']
    
    print("=" * 60)
    print("Copying Labels to Image Directories")
    print("=" * 60)
    
    for split in splits:
        print(f"\nProcessing {split}...")
        
        src_labels = labels_dir / split
        dst_labels = pklot_dir / split / 'labels'
        
        # Create labels directory in PKLot
        dst_labels.mkdir(exist_ok=True)
        
        # Copy all label files
        label_files = list(src_labels.glob('*.txt'))
        print(f"Copying {len(label_files)} label files...")
        
        for label_file in label_files:
            dst_file = dst_labels / label_file.name
            shutil.copy2(label_file, dst_file)
        
        print(f"✓ Labels copied to {dst_labels}")
    
    print("\n" + "=" * 60)
    print("✓ All labels copied successfully!")
    print("=" * 60)

if __name__ == '__main__':
    copy_labels_to_images()
