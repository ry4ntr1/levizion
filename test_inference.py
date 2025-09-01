#!/usr/bin/env python3
"""
Local testing script for Levizion Basketball Detection
Download the trained model and run inference on local images/videos
"""
import os
import cv2
import numpy as np
from google.cloud import storage
from ultralytics import YOLO
import argparse
from pathlib import Path

class BasketballDetector:
    def __init__(self, model_path=None):
        """Initialize the basketball detector."""
        self.class_names = ["Ball", "Hoop", "Period", "Player", "Ref", "Shot Clock", "Team Name", "Team Points", "Time Remaining"]
        self.colors = [
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
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Downloading trained model from GCS...")
            self.model = self.download_and_load_model()
    
    def download_and_load_model(self):
        """Download the trained model from GCS."""
        local_model_path = "best.pt"
        
        if not os.path.exists(local_model_path):
            try:
                client = storage.Client()
                bucket = client.bucket("levizion-ai-oob-data") 
                blob = bucket.blob("models/final_bulletproof_20250901_131338/weights/best.pt")
                blob.download_to_filename(local_model_path)
                print(f"Model downloaded to {local_model_path}")
            except Exception as e:
                print(f"Failed to download model: {e}")
                print("Please ensure you have GCS credentials configured")
                return None
        
        return YOLO(local_model_path)
    
    def draw_predictions(self, image, results, confidence_threshold=0.5):
        """Draw beautiful bounding boxes and labels on image."""
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
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                color = self.colors[class_id % len(self.colors)]
                
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
    
    def predict_image(self, image_path, confidence=0.5, save_path=None):
        """Predict objects in an image."""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Run prediction
        results = self.model(image, conf=confidence)
        
        # Extract predictions
        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = box.conf[0].cpu().numpy().item()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                    
                    predictions.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        print(f"Found {len(predictions)} objects:")
        for pred in predictions:
            print(f"  - {pred['class']}: {pred['confidence']:.2f}")
        
        # Draw predictions
        annotated_image = self.draw_predictions(image, results, confidence)
        
        # Save result
        if save_path is None:
            save_path = f"output_{Path(image_path).stem}.jpg"
        
        cv2.imwrite(save_path, annotated_image)
        print(f"Annotated image saved: {save_path}")
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "output_path": save_path
        }
    
    def predict_video(self, video_path, confidence=0.5, save_path=None, max_frames=None):
        """Predict objects in a video."""
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return None
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Video specs: {width}x{height}, {fps}fps, {total_frames} frames")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"Limiting to first {total_frames} frames")
        
        # Create output video
        if save_path is None:
            save_path = f"output_{Path(video_path).stem}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_predictions = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break
                
                # Run prediction
                results = self.model(frame, conf=confidence)
                
                # Count predictions
                frame_predictions = 0
                for result in results:
                    if result.boxes is not None:
                        frame_predictions += len(result.boxes)
                
                total_predictions += frame_predictions
                
                # Draw predictions
                annotated_frame = self.draw_predictions(frame, results, confidence)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress update every 30 frames
                    print(f"Processed {frame_count}/{total_frames} frames, {total_predictions} total detections")
        
        finally:
            cap.release()
            out.release()
        
        print(f"Video processing complete!")
        print(f"Processed {frame_count} frames with {total_predictions} total detections")
        print(f"Average detections per frame: {total_predictions/frame_count:.2f}")
        print(f"Annotated video saved: {save_path}")
        
        return {
            "frames_processed": frame_count,
            "total_predictions": total_predictions,
            "avg_predictions_per_frame": total_predictions / frame_count if frame_count > 0 else 0,
            "output_path": save_path
        }

def main():
    parser = argparse.ArgumentParser(description="Levizion Basketball Detection - Local Testing")
    parser.add_argument("input", help="Path to input image or video")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, 
                       help="Confidence threshold (0.1-1.0, default: 0.5)")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--model", "-m", help="Path to custom model file")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process (for videos)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Initialize detector
    detector = BasketballDetector(args.model)
    if detector.model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Determine if input is image or video
    input_path = Path(args.input)
    is_video = input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    try:
        if is_video:
            result = detector.predict_video(
                args.input, 
                confidence=args.confidence, 
                save_path=args.output,
                max_frames=args.max_frames
            )
        else:
            result = detector.predict_image(
                args.input, 
                confidence=args.confidence, 
                save_path=args.output
            )
        
        if result:
            print("\n✅ Processing completed successfully!")
            print(f"📁 Output saved to: {result.get('output_path', 'unknown')}")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    main()