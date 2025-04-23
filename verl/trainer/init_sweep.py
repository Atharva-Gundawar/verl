import wandb
import yaml
import sys
import os

def init_sweep(config_path, project_name):
    # Load sweep config
    with open(config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Initialize sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    
    # Get the full sweep path (entity/project/sweep_id)
    entity = wandb.api.default_entity
    full_sweep_path = f"{entity}/{project_name}/{sweep_id}"
    print(f"SWEEP_ID={full_sweep_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python init_sweep.py <config_path> <project_name>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    project_name = sys.argv[2]
    init_sweep(config_path, project_name) 