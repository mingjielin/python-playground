import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class LossAnalyzer:
    """
    Analyze loss distribution during training to identify problematic samples
    """
    def __init__(self):
        self.loss_history = []
        self.sample_losses = defaultdict(list)
        self.high_loss_samples = set()
    
    def analyze_batch_losses(self, predictions, targets, sample_indices=None):
        """
        Analyze losses for a batch of samples
        """
        losses = (predictions - targets) ** 2
        losses_np = losses.detach().cpu().numpy()
        
        # Store individual sample losses
        for i, loss_val in enumerate(losses_np):
            idx = sample_indices[i] if sample_indices else i
            self.sample_losses[idx].append(loss_val)
        
        return losses_np
    
    def identify_outliers(self, percentile=95):
        """
        Identify samples with consistently high losses
        """
        final_losses = {}
        for sample_idx, losses in self.sample_losses.items():
            final_losses[sample_idx] = np.mean(losses)
        
        # Find outliers
        all_final_losses = list(final_losses.values())
        threshold = np.percentile(all_final_losses, percentile)
        
        outlier_samples = {idx for idx, loss in final_losses.items() if loss > threshold}
        self.high_loss_samples = outlier_samples
        
        return outlier_samples, threshold
    
    def plot_loss_distribution(self):
        """
        Plot loss distribution analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        all_losses = []
        for losses in self.sample_losses.values():
            all_losses.extend(losses)
        
        # 1. Overall loss distribution
        axes[0, 0].hist(all_losses, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Overall Loss Distribution')
        axes[0, 0].set_xlabel('Loss Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Sample-wise loss box plot (first 100 samples for clarity)
        sample_ids = list(self.sample_losses.keys())[:100]
        sample_avg_losses = [np.mean(self.sample_losses[idx]) for idx in sample_ids]
        
        axes[0, 1].boxplot(sample_avg_losses)
        axes[0, 1].set_title('Sample-wise Loss Distribution')
        axes[0, 1].set_ylabel('Average Loss per Sample')
        
        # 3. Loss evolution over time for high-loss samples
        if self.high_loss_samples:
            high_loss_data = [self.sample_losses[idx] for idx in list(self.high_loss_samples)[:10]]
            axes[1, 0].boxplot(high_loss_data)
            axes[1, 0].set_title('High-Loss Sample Evolution')
            axes[1, 0].set_ylabel('Loss Value')
        
        # 4. Cumulative loss distribution
        sorted_losses = np.sort(all_losses)
        cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        axes[1, 1].plot(sorted_losses, cumulative)
        axes[1, 1].set_title('Cumulative Loss Distribution')
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        
        plt.tight_layout()
        plt.show()