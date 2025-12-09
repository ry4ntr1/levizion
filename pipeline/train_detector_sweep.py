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

def train_with_wandb():
    """Train with WandB sweep integration for hyperparameter optimization"""
    
    # Initialize WandB run (gets sweep config automatically)
    wandb.init()
    config = wandb.config
    
    # Get base arguments (data paths etc.)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_uri", required=True, help="gs://.../datasets/<name>")
    ap.add_argument("--out_uri", required=True, help="gs://.../models/<run_name>")
    args = ap.parse_args()
    
    work = pathlib.Path("/workspace")
    data_dir = work / "dataset"
    out_dir = work / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset from {args.data_uri} ...", flush=True)
    n = download_gcs_dir(args.data_uri, str(data_dir))
    print(f"Downloaded {n} files to {data_dir}", flush=True)
    
    data_yaml = str(data_dir / "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"ERROR: {data_yaml} not found", file=sys.stderr)
        sys.exit(2)
    
    # Create unique run name with sweep parameters
    run_name = f"sweep_{config.base.replace('.pt', '')}_{config.imgsz}px_{config.epochs}ep_{wandb.run.id}"
    
    print(f"Starting training with sweep config: {dict(config)}", flush=True)
    
    # Basketball-specific class weights (adjust based on your dataset)
    basketball_class_weights = {
        'ball': 2.0,        # Ball is small and critical - higher weight
        'player': 1.0,      # Players are primary objects
        'hoop': 1.5,        # Hoop is important but easier to detect
        'ref': 0.8,         # Referee less critical
        'shot_clock': 1.2,  # Game elements moderately important
        'team_name': 0.7,   # Text elements lower priority
        'team_points': 0.7,
        'time_remaining': 0.7,
        'period': 0.6
    }
    
    # Enhanced training parameters combining sweep config with basketball optimizations
    model = YOLO(config.base)
    
    # Basketball-specific training optimizations
    train_params = {
        'data': data_yaml,
        'imgsz': config.imgsz,
        'epochs': config.epochs,
        'device': 0,
        'project': str(out_dir),
        'name': "detector",
        'batch': -1,  # Auto batch size for optimal GPU usage
        'cache': False,  # Avoid memory issues in cloud
        'workers': 0,   # Avoid multiprocessing issues
        'save': True,
        'save_period': max(10, config.epochs // 5),  # Save more frequently for longer runs
        'patience': max(10, config.epochs // 3),     # Early stopping scaled to epoch count
        
        # Optimizer parameters from sweep
        'optimizer': config.get('optimizer', 'AdamW'),
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'weight_decay': config.get('weight_decay', 0.0005),
        'dropout': config.get('dropout', 0.0),
        
        # Learning rate schedule
        'warmup_epochs': min(3, config.epochs // 10),
        'cos_lr': True,
        
        # Augmentation parameters from sweep
        'augment': True,
        'hsv_h': config.get('hsv_h', 0.015),
        'hsv_s': config.get('hsv_s', 0.7),
        'hsv_v': config.get('hsv_v', 0.4),
        'degrees': config.get('degrees', 0.0),
        'translate': config.get('translate', 0.1),
        'scale': config.get('scale', 0.5),
        'mosaic': config.get('mosaic', 1.0),
        'mixup': config.get('mixup', 0.2),
        'copy_paste': config.get('copy_paste', 0.1),
        
        # Basketball-specific augmentations
        'flipud': 0.0,      # Don't flip basketball court upside down
        'fliplr': 0.5,      # Horizontal flip okay for basketball
        'perspective': 0.0,  # Avoid perspective distortion for court geometry
        'shear': 2.0,       # Minimal shear to preserve court lines
        
        # Loss function weights for basketball classes
        'cls': 1.0,         # Classification loss weight
        'box': 7.5,         # Bounding box loss weight (higher for precise localization)
        'dfl': 1.5,         # Distribution focal loss weight
        
        # Multi-scale training for basketball (court varies in distance)
        'rect': True,       # Rectangular training for efficiency
        'single_cls': False, # Multi-class detection
    }
    
    # Log training parameters to WandB
    wandb.config.update(train_params, allow_val_change=True)
    
    print(f"Training {config.base} at {config.imgsz}px for {config.epochs} epochs", flush=True)
    
    try:
        results = model.train(**train_params)
        
        # Log final metrics to WandB
        metrics_dict = getattr(results, 'results_dict', {})
        if metrics_dict:
            wandb.log({
                'final_mAP50': metrics_dict.get('metrics/mAP50(B)', 0),
                'final_mAP50-95': metrics_dict.get('metrics/mAP50-95(B)', 0),
                'final_precision': metrics_dict.get('metrics/precision(B)', 0),
                'final_recall': metrics_dict.get('metrics/recall(B)', 0),
                'train_box_loss': metrics_dict.get('train/box_loss', 0),
                'train_cls_loss': metrics_dict.get('train/cls_loss', 0),
                'train_dfl_loss': metrics_dict.get('train/dfl_loss', 0),
                'val_box_loss': metrics_dict.get('val/box_loss', 0),
                'val_cls_loss': metrics_dict.get('val/cls_loss', 0),
                'val_dfl_loss': metrics_dict.get('val/dfl_loss', 0),
            })
        
        # Export model
        run_dir = pathlib.Path(results.save_dir)
        best_pt = run_dir / "weights" / "best.pt"
        
        if best_pt.exists():
            print("Exporting ONNX ...", flush=True)
            YOLO(str(best_pt)).export(format="onnx", opset=12, imgsz=config.imgsz)
            
            # Log model artifacts
            export_onnx = next(run_dir.glob("weights/*.onnx"), None)
            if export_onnx:
                wandb.save(str(export_onnx))
            wandb.save(str(best_pt))
            
            # Create summary
            summary = {
                "run_name": run_name,
                "sweep_id": wandb.run.sweep_id,
                "run_id": wandb.run.id,
                "config": dict(config),
                "run_dir": str(run_dir),
                "best_pt": str(best_pt),
                "best_onnx": str(export_onnx) if export_onnx else None,
                "metrics": metrics_dict
            }
            
            with open(run_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            # Upload to GCS
            sweep_out_uri = f"{args.out_uri}/{run_name}"
            print(f"Uploading artifacts to {sweep_out_uri} ...", flush=True)
            upload_dir_to_gcs(str(run_dir), sweep_out_uri)
            
            print("Training completed successfully!", flush=True)
        else:
            print("ERROR: best.pt not found", file=sys.stderr)
            wandb.run.finish(exit_code=1)
            sys.exit(3)
            
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        wandb.log({"error": str(e)})
        wandb.run.finish(exit_code=1)
        raise
    
    wandb.finish()

def main():
    """Standard training without sweep (backward compatibility)"""
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_uri", required=True, help="gs://.../datasets/<name>")
    ap.add_argument("--out_uri",  required=True, help="gs://.../models/<run_name>")
    ap.add_argument("--base", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sweep", action="store_true", help="Run with WandB sweep")
    args = ap.parse_args()
    
    if args.sweep:
        # This will be called by wandb agent
        train_with_wandb()
        return
    
    # Original training logic for non-sweep runs
    work = pathlib.Path("/workspace")
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
    
    # Initialize Weights & Biases for standard runs
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    wandb_initialized = False
    if wandb_api_key:
        try:
            print("Initializing Weights & Biases monitoring...", flush=True)
            wandb.login(key=wandb_api_key)
            wandb.init(
                project="levizion-basketball-detector",
                name=f"train_{args.base}_{args.epochs}ep_{args.imgsz}px",
                config={
                    "model": args.base,
                    "epochs": args.epochs, 
                    "imgsz": args.imgsz,
                    "dataset": args.data_uri,
                }
            )
            wandb_initialized = True
            print("Wandb initialized successfully!", flush=True)
        except Exception as e:
            print(f"Failed to initialize wandb: {e}. Continuing without monitoring.", flush=True)
    else:
        print("WANDB_API_KEY not found, training without wandb monitoring", flush=True)
    
    model = YOLO(args.base)
    results = model.train(
        data=data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        device=0,
        project=str(out_dir),
        name="detector",
        patience=10,
        save=True,
        save_period=10,
        batch=-1,
        cache=False,
        workers=0,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1
    )

    # Export and upload logic (same as original)
    run_dir = pathlib.Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print("ERROR: best.pt not found", file=sys.stderr); sys.exit(3)

    print("Exporting ONNX ...", flush=True)
    YOLO(str(best_pt)).export(format="onnx", opset=12, imgsz=args.imgsz)

    export_onnx = next(run_dir.glob("weights/*.onnx"), None)
    summary = {
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "best_onnx": str(export_onnx) if export_onnx else None,
        "metrics": getattr(results, "results_dict", {})
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Uploading artifacts to {args.out_uri} ...", flush=True)
    upload_dir_to_gcs(str(run_dir), args.out_uri)
    
    if wandb_initialized:
        try:
            wandb.finish()
            print("Wandb run completed successfully!", flush=True)
        except Exception as e:
            print(f"Warning: Failed to finish wandb run: {e}", flush=True)
    
    print("Done.", flush=True)

if __name__ == "__main__":
    main()