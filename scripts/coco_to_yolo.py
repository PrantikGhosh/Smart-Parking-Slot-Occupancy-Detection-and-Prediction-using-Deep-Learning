import json
import os
from pathlib import Path
from tqdm import tqdm


def convert_coco_to_yolo(coco_json_path, images_dir, output_labels_dir):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON annotation file
        images_dir: Directory containing the images
        output_labels_dir: Directory to save YOLO format labels
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Load COCO annotations
    print(f"Loading annotations from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to image info mapping
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Create image_id to annotations mapping
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Process each image
    print(f"Converting {len(images_dict)} images...")
    converted_count = 0
    
    for image_id, image_info in tqdm(images_dict.items()):
        # Get image dimensions
        img_width = image_info['width']
        img_height = image_info['height']
        file_name = image_info['file_name']
        
        # Create label file path (same name as image but .txt extension)
        label_file_name = Path(file_name).stem + '.txt'
        label_file_path = os.path.join(output_labels_dir, label_file_name)
        
        # Get annotations for this image
        annotations = annotations_by_image.get(image_id, [])
        
        # Write YOLO format annotations
        with open(label_file_path, 'w') as f:
            for ann in annotations:
                # COCO format: [x, y, width, height] (top-left corner)
                # YOLO format: [class_id, x_center, y_center, width, height] (normalized 0-1)
                
                category_id = ann['category_id']
                bbox = ann['bbox']
                
                # Convert COCO bbox to YOLO format
                x_coco, y_coco, w_coco, h_coco = bbox
                
                # Calculate center coordinates
                x_center = (x_coco + w_coco / 2) / img_width
                y_center = (y_coco + h_coco / 2) / img_height
                
                # Normalize width and height
                width_norm = w_coco / img_width
                height_norm = h_coco / img_height
                
                # Ensure values are within [0, 1] range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width_norm = max(0, min(1, width_norm))
                height_norm = max(0, min(1, height_norm))
                
                # Write in YOLO format: class_id x_center y_center width height
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        converted_count += 1
    
    print(f"✓ Converted {converted_count} images")
    print(f"✓ Labels saved to {output_labels_dir}")


def main():
    """Main function to convert all splits (train, valid, test)"""
    
    # Base directories
    base_dir = Path(r'e:\Final Year Project')
    pklot_dir = base_dir / 'PKLot'
    dataset_dir = base_dir / 'dataset'
    
    # Splits to process
    splits = ['train', 'valid', 'test']
    
    print("=" * 60)
    print("COCO to YOLO Format Conversion")
    print("=" * 60)
    
    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Processing {split.upper()} split")
        print('=' * 60)
        
        # Paths for this split
        coco_json = pklot_dir / split / '_annotations.coco.json'
        images_source = pklot_dir / split
        
        # Output directory for labels (in standard YOLO structure: PKLot/{split}/labels/)
        output_labels_dir = pklot_dir / split / 'labels'
        
        # Create labels directory
        os.makedirs(output_labels_dir, exist_ok=True)
        
        # Convert annotations
        convert_coco_to_yolo(
            coco_json_path=str(coco_json),
            images_dir=str(images_source),
            output_labels_dir=str(output_labels_dir)
        )
        
        print(f"\n✓ Labels saved alongside images in: {output_labels_dir}")
    
    print("\n" + "=" * 60)
    print("✓ Conversion completed successfully!")
    print("=" * 60)
    print(f"\nLabels created in standard YOLO format:")
    print(f"  - Train: {pklot_dir / 'train' / 'labels'}")
    print(f"  - Valid: {pklot_dir / 'valid' / 'labels'}")
    print(f"  - Test: {pklot_dir / 'test' / 'labels'}")
    print("\nNext steps:")
    print("1. Review the converted labels")
    print("2. Verify data.yaml points to correct paths")
    print("3. Train YOLOv8 model")


if __name__ == '__main__':
    main()
