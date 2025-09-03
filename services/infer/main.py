from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, json, io, cv2, numpy as np
import torch
from ultralytics import YOLO
from google.cloud import storage
import tempfile
from pathlib import Path
from typing import Optional

app = FastAPI(title="Levizion Basketball Detection (GPU)")

# Mount static files for web interface
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model variable
model = None
class_names = ["Ball", "Hoop", "Period", "Player", "Ref", "Shot Clock", "Team Name", "Team Points", "Time Remaining"]

def download_model_from_gcs():
    """Download the trained model from GCS if not already present."""
    global model
    if model is not None:
        return model
    
    local_model_path = "/tmp/best.pt"
    
    if not os.path.exists(local_model_path):
        try:
            print("Downloading trained model from GCS...")
            client = storage.Client()
            bucket = client.bucket("levizion-ai-oob-data") 
            blob = bucket.blob("models/final_bulletproof_20250901_131338/weights/best.pt")
            blob.download_to_filename(local_model_path)
            print(f"Model downloaded to {local_model_path}")
        except Exception as e:
            print(f"Failed to download model: {e}")
            return None
    
    try:
        print("Loading YOLO model...")
        model = YOLO(local_model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def gpu_status():
    info = {"cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_env": os.environ.get("NVIDIA_VISIBLE_DEVICES", "")}
    try:
        cuda_ok = torch.cuda.is_available()
        devices = []
        if cuda_ok:
            count = torch.cuda.device_count()
            for i in range(count):
                devices.append(torch.cuda.get_device_name(i))
        info.update({"torch": True, "cuda_available": cuda_ok, "devices": devices})
    except Exception as e:
        info.update({"torch": False, "error": str(e)})
    return info

def draw_predictions(image, results, confidence_threshold=0.5):
    """Draw beautiful bounding boxes and labels on image."""
    # Color palette for different classes (BGR format for OpenCV)
    colors = [
        (0, 255, 0),     # Ball - Green
        (255, 0, 0),     # Hoop - Blue  
        (0, 0, 255),     # Period - Red
        (255, 255, 0),   # Player - Cyan
        (255, 0, 255),   # Ref - Magenta
        (0, 255, 255),   # Shot Clock - Yellow
        (128, 0, 128),   # Team Name - Purple
        (255, 165, 0),   # Team Points - Orange
        (0, 128, 255)    # Time Remaining - Light Blue
    ]
    
    annotated_img = image.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            # Get box coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            if confidence < confidence_threshold:
                continue
                
            # Get class name and color
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            color = colors[class_id % len(colors)]
            
            # Draw bounding box with thick border
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_img,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # White text
                2
            )
    
    return annotated_img

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the web interface."""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Levizion Basketball Detection API</h1>
                <p>Web interface not available. Use the API endpoints:</p>
                <ul>
                    <li>POST /predict/image - Upload an image for detection</li>
                    <li>POST /predict/video - Upload a video for detection</li>
                    <li>GET /health - Check service health</li>
                    <li>GET /gpu - Check GPU status</li>
                </ul>
            </body>
        </html>
        """)

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": model is not None}

@app.get("/gpu")
def gpu():
    return gpu_status()

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.5,
    return_image: Optional[bool] = True
):
    """Predict objects in an uploaded image."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Load model if not already loaded
    current_model = download_model_from_gcs()
    if current_model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        # Read and decode image
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run prediction
        results = current_model(image, conf=confidence)
        
        # Extract prediction data
        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = box.conf[0].cpu().numpy().item()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    
                    predictions.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        response_data = {
            "predictions": predictions,
            "count": len(predictions),
            "image_size": {"width": image.shape[1], "height": image.shape[0]}
        }
        
        if return_image:
            # Draw predictions on image
            annotated_image = draw_predictions(image, results, confidence)
            
            # Encode image to bytes
            _, buffer = cv2.imencode('.jpg', annotated_image)
            image_bytes = buffer.tobytes()
            
            return StreamingResponse(
                io.BytesIO(image_bytes),
                media_type="image/jpeg",
                headers={"X-Predictions": json.dumps(response_data)}
            )
        else:
            return JSONResponse(response_data)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.5,
    max_frames: Optional[int] = 300  # Limit processing for demo
):
    """Predict objects in an uploaded video and return annotated video."""
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Load model if not already loaded
    current_model = download_model_from_gcs()
    if current_model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(await file.read())
            input_path = temp_input.name
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer with web-compatible codec
        output_path = tempfile.mktemp(suffix=".mp4")
        
        # Try H.264 codec first (most compatible), fallback to others if needed
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264 (best browser support)
            cv2.VideoWriter_fourcc(*'H264'),  # Alternative H.264
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 Part 2
            cv2.VideoWriter_fourcc(*'XVID'),  # Xvid MPEG-4
        ]
        
        out = None
        for fourcc in fourcc_options:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                break
                
        if not out or not out.isOpened():
            raise HTTPException(status_code=500, detail="Could not create video writer")
        
        frame_count = 0
        total_predictions = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            
            # Run prediction on frame
            results = current_model(frame, conf=confidence)
            
            # Count predictions
            for result in results:
                if result.boxes is not None:
                    total_predictions += len(result.boxes)
            
            # Draw predictions
            annotated_frame = draw_predictions(frame, results, confidence)
            
            # Write frame to output video
            out.write(annotated_frame)
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Clean up input file
        os.unlink(input_path)
        
        # Read output video into memory and return as response
        try:
            with open(output_path, "rb") as f:
                video_content = f.read()
        finally:
            # Clean up output file
            if os.path.exists(output_path):
                os.unlink(output_path)
        
        headers = {
            "X-Frames-Processed": str(frame_count),
            "X-Total-Predictions": str(total_predictions),
            "Content-Length": str(len(video_content)),
            "Accept-Ranges": "bytes"
        }
        
        from fastapi.responses import Response
        return Response(
            content=video_content,
            media_type="video/mp4",
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video prediction failed: {str(e)}")

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    print("Initializing basketball detection model...")
    download_model_from_gcs()
