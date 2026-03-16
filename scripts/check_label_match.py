import os
from pathlib import Path

# Get all images and labels
train_path = Path(r"e:\Final Year Project\PKLot\train")
labels_path = train_path / "labels"

images = sorted([f.stem for f in train_path.glob("*.jpg")])
labels = sorted([f.stem for f in labels_path.glob("*.txt")])

print(f"Total images: {len(images)}")
print(f"Total labels: {len(labels)}")

# Check for exact matches
matched = 0
unmatched_images = []
unmatched_labels = []

image_set = set(images)
label_set = set(labels)

matched = len(image_set & label_set)
unmatched_images = list(image_set - label_set)
unmatched_labels = list(label_set - image_set)

print(f"\nMatched: {matched}")
print(f"Unmatched images (no label): {len(unmatched_images)}")
print(f"Unmatched labels (no image): {len(unmatched_labels)}")

if unmatched_images:
    print(f"\nFirst 5 unmatched images:")
    for img in unmatched_images[:5]:
        print(f"  {img}")

if unmatched_labels:
    print(f"\nFirst 5 unmatched labels:")
    for lbl in unmatched_labels[:5]:
        print(f"  {lbl}")

# Check first few for exact character comparison
if images and labels:
    print(f"\nFirst image basename: '{images[0]}'")
    print(f"First label basename: '{labels[0]}'")
    print(f"Are they equal? {images[0] == labels[0]}")
    
    if images[0] != labels[0]:
        print(f"\nCharacter-by-character comparison:")
        for i, (c1, c2) in enumerate(zip(images[0], labels[0])):
            if c1 != c2:
                print(f"  Position {i}: image='{c1}' (ord={ord(c1)}), label='{c2}' (ord={ord(c2)})")
