import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from pathlib import Path
import glob
import threading
import time
from collections import defaultdict

class EpochLossDumper:
    """
    Class to dump loss values for all training samples every epoch
    """
    def __init__(self, dump_dir='epoch_losses', max_files=50):
        """
        Args:
            dump_dir: Directory to dump loss files
            max_files: Maximum number of files to keep (for memory management)
        """
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(exist_ok=True)
        self.max_files = max_files
        self.file_counter = 0
    

    def dump_epoch_losses(self, epoch, loss_values, sample_indices=None, additional_data=None):
        """
        Dump loss values for all training samples in an epoch
        
        Args:
            epoch: Epoch number
            loss_values: Array of loss values for all samples
            sample_indices: Optional array of sample indices
            additional_data: Optional dict with additional metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epoch_{epoch:04d}_{timestamp}.json"
        filepath = self.dump_dir / filename
        
        # Prepare filtered data to dump
        # only consider losses above the 99th percentile
        # top 1% loss filtering 
        if len(loss_values) > 100:
            threshold = np.percentile(loss_values, 95)
            filtered_indices = np.where(loss_values >= threshold)[0]    # bad losses only
            top_loss_array = np.array(loss_values)[filtered_indices]
            top_loss_values = top_loss_array.tolist()

        # Prepare all sample data to dump
        dump_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            'loss_values': loss_values,
            'sample_count': len(loss_values),
            'top_loss_values': top_loss_values,
            'top_loss_sample_indices': filtered_indices.tolist(),
            'stats': {
                'mean': float(np.mean(loss_values)),
                'std': float(np.std(loss_values)),
                'min': float(np.min(loss_values)),
                'max': float(np.max(loss_values)),
                'median': float(np.median(loss_values))
            }
        }
        
        if sample_indices is not None:
            dump_data['sample_indices'] = sample_indices.tolist()
        
        if additional_data:
            dump_data.update(additional_data)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(dump_data, f, indent=2)
        
        print(f"Dumped epoch {epoch} losses to {filepath}")
        
        # Manage file count
        self._manage_file_count()
    
    def _manage_file_count(self):
        """
        Keep only the most recent files to manage disk space
        """
        files = list(self.dump_dir.glob("epoch_*.json"))
        if len(files) > self.max_files:
            # Sort by timestamp and remove oldest
            files.sort(key=lambda x: x.stat().st_mtime)
            for old_file in files[:-self.max_files]:
                old_file.unlink()
                print(f"Removed old file: {old_file}")

class RealTimeScatterPlotter:
    """
    Class to create real-time scatter plots of loss values
    """
    def __init__(self, data_dir='epoch_losses', refresh_interval=10, max_points=10000):
        """
        Args:
            data_dir: Directory containing dumped loss files
            refresh_interval: Refresh interval in seconds
            max_points: Maximum points to plot (for performance)
        """
        self.data_dir = Path(data_dir)
        self.refresh_interval = refresh_interval
        self.max_points = max_points
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.scatter = self.ax.scatter([], [], s=20, alpha=0.6, c='blue')
        self.ax.set_xlabel('Sample Index')
        self.ax.set_ylabel('Loss Value')
        self.ax.set_title('Real-time Training Loss Distribution')
        self.ax.grid(True, alpha=0.3)
        
        # Track last modified time to avoid re-reading same files
        self.last_file_times = {}
        self.processed_files = set()
        
        self.ani = None
        self.all_data = []
    
    def get_latest_files(self):
        """
        Get the latest loss files from the directory
        """
        pattern = self.data_dir / "epoch_*.json"
        files = list(pattern.parent.glob(pattern.name))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files
    
    def load_latest_data(self):
        """
        Load data from the most recent files
        """
        files = self.get_latest_files()
        
        new_data = []
        for file_path in files:
            if str(file_path) in self.processed_files:
                continue
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract loss values and create scatter plot data
                epoch = data['epoch']
                loss_values = np.array(data['loss_values'])
                
                # Create sample indices if not provided
                if 'sample_indices' in data:
                    sample_indices = np.array(data['sample_indices'])
                else:
                    sample_indices = np.arange(len(loss_values))
                
                # Add epoch information to each point
                for i, (idx, loss) in enumerate(zip(sample_indices, loss_values)):
                    new_data.append({
                        'epoch': epoch,
                        'sample_idx': idx,
                        'loss': loss,
                        'timestamp': data['timestamp']
                    })
                
                self.processed_files.add(str(file_path))
                
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
        
        return new_data
    
    def update_scatter_plot(self, frame):
        """
        Update function for animation
        """
        # Load new data
        new_data = self.load_latest_data()
        
        if new_data:
            self.all_data.extend(new_data)
        
        if not self.all_data:
            return self.scatter,
        
        # Convert to arrays for plotting
        if len(self.all_data) > self.max_points:
            # Keep most recent points
            recent_data = self.all_data[-self.max_points:]
        else:
            recent_data = self.all_data
        
        sample_indices = np.array([d['sample_idx'] for d in recent_data])
        loss_values = np.array([d['loss'] for d in recent_data])
        epochs = np.array([d['epoch'] for d in recent_data])
        
        # Update scatter plot
        self.scatter.set_offsets(np.column_stack([sample_indices, loss_values]))
        
        # Color by epoch
        colors = epochs
        self.scatter.set_array(colors)
        
        # Update axes limits
        self.ax.set_xlim(sample_indices.min(), sample_indices.max())
        self.ax.set_ylim(loss_values.min(), loss_values.max())
        
        # Update title with current info
        current_epoch = max(epochs) if len(epochs) > 0 else 0
        self.ax.set_title(f'Training Loss Distribution - Epoch {current_epoch}, Points: {len(recent_data)}')
        
        return self.scatter,
    
    def start_plotting(self):
        """
        Start the real-time scatter plotting
        """
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_scatter_plot,
            interval=self.refresh_interval * 1000,  # Convert to milliseconds
            blit=False
        )
        
        plt.tight_layout()
        plt.show()
    
    def stop_plotting(self):
        """
        Stop the real-time plotting
        """
        if self.ani:
            self.ani.event_source.stop()



if __name__ == "__main__":

    rtsp = RealTimeScatterPlotter('/home/mingjie/python--playground/python-playground/epoch_losses')
    rtsp.start_plotting()
    rtsp.start_plotting()