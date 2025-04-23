import torch
import psutil
import time
import threading
from typing import Dict, List, Optional
import csv
from datetime import datetime
import wandb

class ResourceProfiler:
    def __init__(self, output_file: str = "resource_usage.csv", interval: float = 1.0, 
                 project_name: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Initialize the resource profiler.
        
        Args:
            output_file (str): Path to save the profiling data
            interval (float): Time interval between measurements in seconds
            project_name (str): Name of the wandb project
            experiment_name (str): Name of the wandb experiment
        """
        self.output_file = output_file
        self.interval = interval
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.csv_file = None
        self.csv_writer = None
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.wandb_run = None
        self.current_step = 0
        self.last_logged_step = 0
        
    def _get_gpu_usage(self) -> Dict[str, float]:
        """Get GPU memory usage and utilization."""
        gpu_usage = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_usage[f"gpu/{i}/memory_used"] = torch.cuda.memory_allocated(i) / 1024**2  # MB
                gpu_usage[f"gpu/{i}/memory_cached"] = torch.cuda.memory_reserved(i) / 1024**2  # MB
                gpu_usage[f"gpu/{i}/utilization"] = torch.cuda.utilization(i)  # Percentage
        return gpu_usage
    
    def _get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage and memory information."""
        cpu_usage = {
            "cpu/percent": psutil.cpu_percent(),
            "cpu/memory/used": psutil.virtual_memory().used / 1024**2,  # MB
            "cpu/memory/percent": psutil.virtual_memory().percent,
        }
        return cpu_usage
    
    def _monitor_resources(self):
        """Background thread function to monitor resources."""
        while self.is_running:
            try:
                # Only log if the step has changed since last log
                if self.current_step > self.last_logged_step:
                    # Get current timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Get resource usage
                    gpu_usage = self._get_gpu_usage()
                    cpu_usage = self._get_cpu_usage()
                    
                    # Combine all metrics
                    metrics = {
                        "timestamp": timestamp,
                        **gpu_usage,
                        **cpu_usage
                    }
                    
                    # Write to CSV
                    if self.csv_writer:
                        self.csv_writer.writerow(metrics)
                        self.csv_file.flush()
                    
                    # Log to wandb
                    if self.wandb_run:
                        wandb_metrics = {
                            "timestamp": timestamp,
                            "step": self.current_step,
                            **gpu_usage,
                            **cpu_usage
                        }
                        self.wandb_run.log(wandb_metrics)
                        self.last_logged_step = self.current_step
                
                time.sleep(self.interval)
                
            except Exception as e:
                if self.wandb_run:
                    self.wandb_run.log({"error": str(e)})
                break
    
    def update_step(self, step: int):
        """Update the current step counter."""
        self.current_step = step
    
    def start(self):
        """Start the resource monitoring."""
        if self.is_running:
            return
            
        # Initialize wandb if project and experiment names are provided
        if self.project_name and self.experiment_name:
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config={
                    "interval": self.interval,
                    "output_file": self.output_file
                }
            )
            
        # Open CSV file for writing
        self.csv_file = open(self.output_file, 'w', newline='')
        fieldnames = ["timestamp"]
        
        # Add GPU fields if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                fieldnames.extend([
                    f"gpu/{i}/memory_used",
                    f"gpu/{i}/memory_cached",
                    f"gpu/{i}/utilization"
                ])
        
        # Add CPU fields
        fieldnames.extend([
            "cpu/percent",
            "cpu/memory/used",
            "cpu/memory/percent"
        ])
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        
        # Start monitoring thread
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_resources)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the resource monitoring."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.thread:
            self.thread.join()
            
        if self.csv_file:
            self.csv_file.close()
            
        if self.wandb_run:
            self.wandb_run.finish()

def profile_training(func):
    """
    Decorator to profile resource usage during training.
    
    Usage:
        @profile_training
        def train(...):
            ...
    """
    def wrapper(*args, **kwargs):
        # Get project_name and experiment_name from the trainer instance
        trainer = args[0]  # First argument is the trainer instance
        project_name = getattr(trainer.config.trainer, 'project_name', None)
        experiment_name = getattr(trainer.config.trainer, 'experiment_name', None)
        
        profiler = ResourceProfiler(
            project_name=project_name,
            experiment_name=experiment_name
        )
        # Store profiler instance on trainer
        trainer._profiler = profiler
        try:
            profiler.start()
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.stop()
            # Clean up profiler instance
            delattr(trainer, '_profiler')
    return wrapper 