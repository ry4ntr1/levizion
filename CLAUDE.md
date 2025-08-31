# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Levizion is a basketball object detection system using YOLOv8 for training custom models on Google Cloud Platform. The system detects 9 classes: Ball, Hoop, Period, Player, Ref, Shot Clock, Team Name, Team Points, Time Remaining.

## Architecture

### Core Components

1. **Training Pipeline** (`pipeline/train_detector.py`)
   - Downloads datasets from GCS
   - Trains YOLOv8 models using Ultralytics
   - Exports trained models (PT and ONNX formats)
   - Uploads results back to GCS
   - Expects datasets with `data.yaml` specifying train/val/test paths

2. **Inference Service** (`services/infer/main.py`)
   - FastAPI application for GPU-accelerated inference
   - Deployed on Cloud Run with L4 GPU support
   - Provides health checks and GPU status endpoints

3. **API/UI Service** (`services/api/main.py`)
   - FastAPI application for CPU-based API and UI
   - Deployed on Cloud Run without GPU

### Deployment Infrastructure

All deployments use GitHub Actions workflows with Workload Identity Federation:
- **train-batch.yml**: Cloud Batch GPU training (L4/T4 GPUs)
- **deploy-infer-gpu.yml**: Deploy inference service to Cloud Run with GPU
- **deploy-api-ui.yml**: Deploy API/UI service to Cloud Run

Docker images are stored in Artifact Registry: `us-central1-docker.pkg.dev/levizion-ai/levizion/`

## Common Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install ultralytics google-cloud-storage pyyaml opencv-python-headless

# Test training script
python pipeline/train_detector.py --help

# Run API locally
uvicorn services.api.main:app --reload --port 8000

# Run inference service locally
uvicorn services.infer.main:app --reload --port 8001
```

### Training Operations
```bash
# Trigger training via GitHub Actions (recommended)
gh workflow run "Train Detector (Cloud Batch GPU)" \
  --field dataset_gcs_uri="gs://levizion-ai-oob-data/datasets/<dataset>" \
  --field out_gcs_uri="gs://levizion-ai-oob-data/models/<run_name>" \
  --field epochs="50" \
  --field imgsz="640" \
  --field base_model="yolov8s.pt"

# Monitor training job
gcloud batch jobs list --location=us-central1
gcloud batch jobs describe <job-id> --location=us-central1

# Check training logs
gcloud logging read 'resource.labels.job_id="<job-id>"' --limit=50 --freshness=1h
```

### Dataset Management
```bash
# Dataset structure required:
# dataset/
#   ├── data.yaml (paths: train/images, val/images, test/images)
#   ├── train/
#   │   ├── images/
#   │   └── labels/
#   ├── valid/
#   │   ├── images/
#   │   └── labels/
#   └── test/
#       ├── images/
#       └── labels/

# Upload dataset to GCS
gsutil -m cp -r <local_dataset> gs://levizion-ai-oob-data/datasets/

# Check dataset structure
gsutil ls -la gs://levizion-ai-oob-data/datasets/<dataset>/
```

### Deployment Commands
```bash
# Deploy services (auto-triggered on push to main)
# Manual deployment:
gh workflow run "Deploy Inference (GPU L4)"
gh workflow run "Deploy API-UI (CPU)"

# Check service status
gcloud run services list --region=us-central1
gcloud run services describe <service-name> --region=us-central1
```

## Key Configuration Details

### GCP Resources
- **Project ID**: levizion-ai
- **Region**: us-central1
- **Artifact Registry**: us-central1-docker.pkg.dev/levizion-ai/levizion
- **Storage Buckets**: levizion-ai-oob-data (datasets and models)
- **Service Accounts**:
  - github-deployer@levizion-ai.iam.gserviceaccount.com (deployments)
  - batch-runner@levizion-ai.iam.gserviceaccount.com (training)

### Training Configuration
- **Default GPU**: nvidia-l4 on g2-standard-4 machine
- **Max duration**: 3600s (1 hour) per job
- **Retry count**: 2 retries on failure
- **Docker base**: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

### Known Issues and Solutions

1. **Library Dependencies**: Dockerfile.train requires system libraries for OpenCV:
   ```dockerfile
   apt-get install -y libglib2.0-0 libgtk-3-0 libgl1-mesa-glx
   ```

2. **Dataset Paths**: data.yaml must use relative paths from dataset root:
   ```yaml
   train: train/images  # NOT ../train/images
   val: valid/images    # NOT ../valid/images
   ```

3. **Monitoring Training**: Cloud Batch logs may be delayed. Check:
   - GitHub Actions logs for submission status
   - Cloud Console for visual monitoring
   - Job state transitions: QUEUED → SCHEDULED → RUNNING → SUCCEEDED/FAILED

## Testing and Debugging

For debugging training issues, use minimal test dataset:
```bash
# Create tiny dataset (32 images) for quick testing
gsutil cp -r gs://levizion-ai-oob-data/datasets/bball_obb_v1/valid \
  gs://levizion-ai-oob-data/datasets/test_tiny/

# Run with minimal parameters
--epochs=2 --imgsz=320 --base_model=yolov8n.pt
```

Always check job status before assuming failure:
```bash
gcloud batch jobs describe <job-id> --location=us-central1 --format="value(status.state,status.runDuration)"
```