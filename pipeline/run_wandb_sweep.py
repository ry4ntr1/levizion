#!/usr/bin/env python3
"""
WandB Sweep Runner for Basketball Object Detection Optimization

This script initializes and runs hyperparameter sweeps using WandB to find
optimal training configurations for basketball object detection.
"""

import argparse
import os
import sys
import subprocess
import yaml
import wandb
from pathlib import Path

def validate_gcs_uri(uri):
    """Validate GCS URI format"""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}. Must start with 'gs://'")
    return uri

def create_sweep(config_file, project_name="levizion-basketball-detector"):
    """Initialize a new WandB sweep"""
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Sweep config file not found: {config_file}")
    
    # Load sweep configuration
    with open(config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    print(f"Creating sweep with config: {config_file}")
    print(f"Method: {sweep_config.get('method', 'random')}")
    print(f"Metric: {sweep_config.get('metric', {})}")
    
    # Initialize WandB and create sweep
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    print(f"✅ Sweep created successfully!")
    print(f"📊 Sweep ID: {sweep_id}")
    print(f"🌐 View at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    return sweep_id

def run_sweep_agent(sweep_id, data_uri, out_uri, count=10):
    """Run WandB sweep agent"""
    
    print(f"🚀 Starting sweep agent for: {sweep_id}")
    print(f"📁 Dataset: {data_uri}")
    print(f"📤 Output: {out_uri}")
    print(f"🔄 Max runs: {count}")
    
    # Set environment variables for the sweep
    env = os.environ.copy()
    env['WANDB_PROJECT'] = 'levizion-basketball-detector'
    
    # Prepare command for sweep agent
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_detector_sweep.py"
    
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    
    # Run sweep agent
    cmd = [
        "wandb", "agent", sweep_id,
        "--count", str(count)
    ]
    
    print(f"💻 Running: {' '.join(cmd)}")
    print("📝 Training logs will appear below...\n")
    
    try:
        # Add data URIs as environment variables so sweep can access them
        env['DATA_URI'] = data_uri
        env['OUT_URI'] = out_uri
        
        result = subprocess.run(cmd, env=env, check=True)
        print("\n✅ Sweep completed successfully!")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Sweep failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Sweep interrupted by user")
        return 130

def list_sweeps(project_name="levizion-basketball-detector"):
    """List existing sweeps in the project"""
    
    try:
        wandb.login()
        api = wandb.Api()
        
        sweeps = api.project(project_name).sweeps()
        
        print(f"📊 Active sweeps in {project_name}:")
        print("-" * 80)
        
        for sweep in sweeps:
            status = "🟢 Running" if sweep.state == "running" else "🔴 Finished"
            print(f"{status} | {sweep.id} | {sweep.name}")
            print(f"   Method: {sweep.config.get('method', 'unknown')}")
            print(f"   Runs: {sweep.run_count}")
            print(f"   URL: https://wandb.ai/{sweep.entity}/{sweep.project}/sweeps/{sweep.id}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to list sweeps: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="WandB Sweep Runner for Basketball Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new sweep
  python run_wandb_sweep.py create --config wandb_sweep_config.yaml

  # Run sweep agent
  python run_wandb_sweep.py run <sweep_id> \\
    --data_uri gs://bucket/datasets/basketball \\
    --out_uri gs://bucket/models/sweep_results \\
    --count 20

  # List existing sweeps
  python run_wandb_sweep.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create sweep command
    create_parser = subparsers.add_parser('create', help='Create a new WandB sweep')
    create_parser.add_argument('--config', required=True, help='Path to sweep config YAML file')
    create_parser.add_argument('--project', default='levizion-basketball-detector', help='WandB project name')
    
    # Run sweep command  
    run_parser = subparsers.add_parser('run', help='Run sweep agent')
    run_parser.add_argument('sweep_id', help='WandB sweep ID')
    run_parser.add_argument('--data_uri', required=True, help='GCS URI for dataset')
    run_parser.add_argument('--out_uri', required=True, help='GCS URI for output models')
    run_parser.add_argument('--count', type=int, default=10, help='Maximum number of runs')
    
    # List sweeps command
    list_parser = subparsers.add_parser('list', help='List existing sweeps')
    list_parser.add_argument('--project', default='levizion-basketball-detector', help='WandB project name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'create':
            sweep_id = create_sweep(args.config, args.project)
            print(f"\n🎯 Next step: Run the sweep agent with:")
            print(f"python run_wandb_sweep.py run {sweep_id} \\")
            print(f"  --data_uri gs://your-bucket/datasets/basketball \\")
            print(f"  --out_uri gs://your-bucket/models/sweep_results")
            
        elif args.command == 'run':
            validate_gcs_uri(args.data_uri)
            validate_gcs_uri(args.out_uri)
            return run_sweep_agent(args.sweep_id, args.data_uri, args.out_uri, args.count)
            
        elif args.command == 'list':
            list_sweeps(args.project)
            
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())