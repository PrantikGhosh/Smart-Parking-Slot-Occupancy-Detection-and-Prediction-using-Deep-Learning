from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch

def benchmark_models():
    # Setup paths
    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / 'models'
    data_yaml = root_dir / 'data' / 'dataset_yolo' / 'data.yaml'
    
    if not models_dir.exists():
        print(f"Error: Models directory not found at {models_dir}")
        return

    # Find all .pt files
    # Prioritize specific models for quick assessment
    priority_models = ['abhivesh_model.pt', 'prantik_model.pt', 'yolov8n.pt']
    
    all_models = list(models_dir.glob("*.pt"))
    filtered_models = [m for m in all_models if m.name in priority_models]
    
    # If priority models not found, fall back to all (or just top 3)
    model_files = filtered_models if filtered_models else all_models[:3]
    
    if not model_files:
        print("No models found to benchmark.")
        return

    print(f"Found {len(model_files)} models to benchmark (Quick Mode: imgsz=320).")
    print(f"Using dataset: {data_yaml}")
    print("=" * 100)
    print(f"{'Model Name':<30} | {'mAP@50':<10} | {'mAP@50-95':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 100)
    
    results = []

    for model_path in model_files:
        model_name = model_path.name
        try:
            # Load model
            model = YOLO(str(model_path))
            
            # Validate on test set
            # Using imgsz=320 for speed
            metrics = model.val(data=str(data_yaml), split='test', verbose=False, plots=False, imgsz=320)
            
            # Extract metrics
            map50 = metrics.box.map50
            map5095 = metrics.box.map
            precision = metrics.box.mp
            recall = metrics.box.mr
            
            print(f"{model_name:<30} | {map50:.4f}     | {map5095:.4f}      | {precision:.4f}    | {recall:.4f}")
            
            results.append({
                'Model': model_name,
                'mAP@50': map50,
                'mAP@50-95': map5095,
                'Precision': precision,
                'Recall': recall
            })
            
        except Exception as e:
            print(f"{model_name:<30} | ERROR: {str(e)[:50]}")

    print("=" * 100)
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by='mAP@50', ascending=False)
        output_path = root_dir / 'benchmark_results.csv'
        df.to_csv(output_path, index=False)
        print(f"\nBenchmark completed. Results saved to {output_path}")
        print("\nTop 3 Models by mAP@50:")
        print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    benchmark_models()
