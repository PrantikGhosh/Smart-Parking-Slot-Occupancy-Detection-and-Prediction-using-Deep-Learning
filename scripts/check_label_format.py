from pathlib import Path

labels_path = Path(r"e:\Final Year Project\PKLot\train\labels")

# Check a few label files
label_files = list(labels_path.glob("*.txt"))[:5]

for lbl_file in label_files:
    print(f"\nChecking: {lbl_file.name}")
    
    # Read as bytes to check for issues
    with open(lbl_file, 'rb') as f:
        content_bytes = f.read()
    
    # Read as text
    with open(lbl_file, 'r') as f:
        content_text = f.read()
    
    lines = content_text.strip().split('\n')
    
    print(f"  File size: {len(content_bytes)} bytes")
    print(f"  Lines: {len(lines)}")
    print(f"  First line: {lines[0] if lines else 'EMPTY'}")
    
    # Check for line ending type
    if b'\r\n' in content_bytes:
        print(f"  Line endings: Windows (CRLF)")
    elif b'\n' in content_bytes:
        print(f"  Line endings: Unix (LF)")
    else:
        print(f"  Line endings: None/Unknown")
    
    # Validate YOLO format
    if lines and lines[0]:
        parts = lines[0].split()
        if len(parts) == 5:
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                if 0 <= cls <= 2 and all(0 <= v <= 1 for v in [x, y, w, h]):
                    print(f"  Format: ✓ Valid YOLO format")
                else:
                    print(f"  Format: ✗ Values out of range")
            except:
                print(f"  Format: ✗ Cannot parse numbers")
        else:
            print(f"  Format: ✗ Wrong number of values ({len(parts)} instead of 5)")
