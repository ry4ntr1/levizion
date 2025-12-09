import argparse, os, sys, json, pathlib, time, shutil
from google.cloud import storage
from ultralytics import YOLO
import wandb

def parse_gcs(uri):
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

def download_gcs_dir(gcs_uri, dest_dir):
    bucket_name, prefix = parse_gcs(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix)
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for b in blobs:
        rel = b.name[len(prefix):].lstrip("/")
        if rel.endswith("/") or not rel:
            continue
        local_path = os.path.join(dest_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        b.download_to_filename(local_path)
        n += 1
    return n

def upload_dir_to_gcs(local_dir, gcs_uri):
    bucket_name, prefix = parse_gcs(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for f in files:
            local_path = os.path.join(root, f)
            rel = os.path.relpath(local_path, local_dir)
            blob = bucket.blob(os.path.join(prefix, rel).replace("\\", "/"))
            blob.upload_from_filename(local_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_uri", required=True, help="gs://.../datasets/<name>")
    ap.add_argument("--out_uri",  required=True, help="gs://.../models/<run_name>")
    ap.add_argument("--base", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    # Use local temp directory instead of /workspace
    work = pathlib.Path.cwd() / "temp_training"
    data_dir = work / "dataset"
    out_dir  = work / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from {args.data_uri} ...", flush=True)
    n = download_gcs_dir(args.data_uri, str(data_dir))
    print(f"Downloaded {n} files to {data_dir}", flush=True)

    data_yaml = str(data_dir / "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"ERROR: {data_yaml} not found", file=sys.stderr); sys.exit(2)

    print(f"Starting training: base={args.base}, imgsz={args.imgsz}, epochs={args.epochs}", flush=True)
    
    # Initialize Weights & Biases
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    wandb_initialized = False
    if wandb_api_key:
        try:
            print("Initializing Weights & Biases monitoring...", flush=True)
            wandb.login(key=wandb_api_key)
            wandb.init(
                project="levizion-basketball-detector",
                name=f"local_train_{args.base}_{args.epochs}ep_{args.imgsz}px",
                config={
                    "model": args.base,
                    "epochs": args.epochs, 
                    "imgsz": args.imgsz,
                    "dataset": args.data_uri,
                    "environment": "local_development"
                }
            )
            wandb_initialized = True
            print("Wandb initialized successfully!", flush=True)
        except Exception as e:
            print(f"Failed to initialize wandb: {e}. Continuing without monitoring.", flush=True)
    else:
        print("WANDB_API_KEY not found, training without wandb monitoring", flush=True)
    
    model = YOLO(args.base)
    
    # Basketball-optimized training parameters
    results = model.train(
        data=data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        device='mps' if hasattr(model, 'model') else 0,  # Use MPS on Apple Silicon, GPU otherwise
        project=str(out_dir),
        name="detector",
        # Optimization settings for basketball detection
        patience=max(5, args.epochs // 3),     # Early stopping scaled to epoch count
        save=True,
        save_period=max(5, args.epochs // 3),  # Save checkpoints more frequently for short runs
        batch=-1,                              # Auto batch size
        cache=False,                           # Disable caching for local runs
        workers=2,                             # Limited workers for local machine
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=min(3, args.epochs // 5),
        cos_lr=True,
        augment=True,
        # Basketball-specific augmentations
        hsv_h=0.015,         # Minimal hue shift
        hsv_s=0.7,           # Moderate saturation
        hsv_v=0.4,           # Moderate brightness
        degrees=0.0,         # No rotation (preserve court orientation)
        translate=0.1,       # Slight translation
        scale=0.5,           # Scale variation
        mosaic=1.0,          # Mosaic augmentation
        mixup=0.2,           # MixUp for better generalization
        copy_paste=0.1,      # Copy-paste for small objects (ball)
        flipud=0.0,          # Don't flip court upside down
        fliplr=0.5,          # Horizontal flip okay
        # Loss weights optimized for basketball
        cls=1.0,             # Classification loss
        box=7.5,             # Higher box loss for precise localization
        dfl=1.5,             # Distribution focal loss
    )

    # Locate best weights and export ONNX
    run_dir = pathlib.Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print("ERROR: best.pt not found", file=sys.stderr); sys.exit(3)

    print("Exporting ONNX ...", flush=True)
    YOLO(str(best_pt)).export(format="onnx", opset=12, imgsz=args.imgsz)

    # Collect artifacts
    export_onnx = next(run_dir.glob("weights/*.onnx"), None)
    summary = {
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "best_onnx": str(export_onnx) if export_onnx else None,
        "metrics": getattr(results, "results_dict", {}),
        "config": {
            "model": args.base,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "dataset": args.data_uri
        }
    }
    
    # Log final metrics to WandB
    if wandb_initialized:
        metrics_dict = getattr(results, 'results_dict', {})
        if metrics_dict:
            wandb.log({
                'final_mAP50': metrics_dict.get('metrics/mAP50(B)', 0),
                'final_mAP50-95': metrics_dict.get('metrics/mAP50-95(B)', 0),
                'final_precision': metrics_dict.get('metrics/precision(B)', 0),
                'final_recall': metrics_dict.get('metrics/recall(B)', 0),
            })
        # Save model artifacts
        if export_onnx:
            wandb.save(str(export_onnx))
        wandb.save(str(best_pt))
    
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Upload to GCS
    print(f"Uploading artifacts to {args.out_uri} ...", flush=True)
    upload_dir_to_gcs(str(run_dir), args.out_uri)
    
    # Finish wandb run if it was initialized
    if wandb_initialized:
        try:
            wandb.finish()
            print("Wandb run completed successfully!", flush=True)
        except Exception as e:
            print(f"Warning: Failed to finish wandb run: {e}", flush=True)
    
    # Clean up local temp directory
    try:
        shutil.rmtree(work)
        print("Cleaned up temporary files.", flush=True)
    except Exception as e:
        print(f"Warning: Failed to clean up temp files: {e}", flush=True)
    
    print("Done.", flush=True)

if __name__ == "__main__":
    main()