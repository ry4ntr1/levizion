#!/usr/bin/env python3
"""
Run multiple optimization experiments with different configurations
This simulates a WandB sweep by running several training configurations
"""

import subprocess
import time
import os
import json

def run_training_experiment(config, experiment_id):
    """Run a single training experiment with given configuration"""
    
    print(f"\n🚀 Starting Experiment {experiment_id}")
    print(f"📊 Config: {config}")
    print("-" * 60)
    
    cmd = [
        "python", "train_detector_local.py",
        "--data_uri", "gs://levizion-ai-oob-data/datasets/bball_obb_v1",
        "--out_uri", f"gs://levizion-ai-oob-data/models/optimization_exp_{experiment_id}",
        "--base", config["base"],
        "--imgsz", str(config["imgsz"]),
        "--epochs", str(config["epochs"])
    ]
    
    print(f"💻 Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Experiment {experiment_id} completed successfully in {duration:.1f}s")
            return {
                "experiment_id": experiment_id,
                "config": config,
                "status": "success",
                "duration": duration,
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-500:]   # Last 500 chars
            }
        else:
            print(f"❌ Experiment {experiment_id} failed with code {result.returncode}")
            return {
                "experiment_id": experiment_id,
                "config": config,
                "status": "failed",
                "duration": duration,
                "error_code": result.returncode,
                "stdout": result.stdout[-1000:],
                "stderr": result.stderr[-500:]
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {experiment_id} timed out after 2 hours")
        return {
            "experiment_id": experiment_id,
            "config": config,
            "status": "timeout",
            "duration": time.time() - start_time
        }
    except Exception as e:
        print(f"💥 Experiment {experiment_id} crashed: {e}")
        return {
            "experiment_id": experiment_id,
            "config": config,
            "status": "error",
            "duration": time.time() - start_time,
            "error": str(e)
        }

def main():
    """Run optimization experiments with different configurations"""
    
    # Basketball-optimized configurations to test
    # Based on common optimal settings for object detection
    experiment_configs = [
        # Fast baseline experiments
        {"base": "yolov8n.pt", "imgsz": 640, "epochs": 25},
        {"base": "yolov8s.pt", "imgsz": 640, "epochs": 25},
        
        # Higher resolution experiments
        {"base": "yolov8n.pt", "imgsz": 832, "epochs": 20},
        {"base": "yolov8s.pt", "imgsz": 832, "epochs": 20},
        
        # Medium model with optimal settings
        {"base": "yolov8m.pt", "imgsz": 640, "epochs": 30},
        
        # High resolution + longer training (best quality)
        {"base": "yolov8s.pt", "imgsz": 1024, "epochs": 35},
    ]
    
    print("🎯 Basketball Object Detection Optimization Batch")
    print(f"📊 Running {len(experiment_configs)} optimization experiments")
    print(f"🏀 Full dataset: gs://levizion-ai-oob-data/datasets/bball_obb_v1")
    print(f"📤 Results: gs://levizion-ai-oob-data/models/optimization_exp_*")
    print("=" * 80)
    
    results = []
    start_time = time.time()
    
    for i, config in enumerate(experiment_configs, 1):
        experiment_result = run_training_experiment(config, i)
        results.append(experiment_result)
        
        # Save intermediate results
        with open("optimization_results.json", "w") as f:
            json.dump({
                "total_experiments": len(experiment_configs),
                "completed": i,
                "total_duration": time.time() - start_time,
                "results": results
            }, f, indent=2)
        
        print(f"\n📊 Progress: {i}/{len(experiment_configs)} experiments completed")
        
        # Short break between experiments
        if i < len(experiment_configs):
            print("⏸️  Brief pause before next experiment...")
            time.sleep(10)
    
    total_duration = time.time() - start_time
    successful = len([r for r in results if r["status"] == "success"])
    
    print("\n" + "=" * 80)
    print("🏁 OPTIMIZATION BATCH COMPLETE")
    print(f"✅ Successful: {successful}/{len(experiment_configs)}")
    print(f"⏱️  Total time: {total_duration/3600:.1f} hours")
    print(f"📊 Results saved to: optimization_results.json")
    
    # Summary of best performing configs
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        print("\n🎯 Next steps:")
        print("1. Check optimization_results.json for detailed results")
        print("2. Compare models in GCS: gs://levizion-ai-oob-data/models/optimization_exp_*")
        print("3. Select best configuration for production training")
        print("4. Use optimal settings for final model training")
    
    return results

if __name__ == "__main__":
    results = main()