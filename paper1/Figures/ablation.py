import matplotlib.pyplot as plt
import numpy as np

# Ablation data
configs = [
    'Sequence only\n(Transformer)',
    'Structure only\n(GNN)',
    'GNN + Trans\n(concat, no diffusion)',
    'GNN + Trans\n(learned fusion)',
    '+ Attention\ndiffusion only',
    '+ Graph\ndiffusion only',
    '+ Bidirectional\ndiffusion (full)'
]
pearson = [0.74, 0.70, 0.79, 0.81, 0.84, 0.82, 0.87]
rmse = [1.59, 1.71, 1.42, 1.38, 1.31, 1.35, 1.21]

# Set up figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(9, 5))

# Colors
color_rho = '#2E5984'      # Deep blue for Pearson
color_rmse = '#C14A4A'     # Terracotta for RMSE
highlight_color = '#4A7B4A'  # Forest green for full model

# X positions
x = np.arange(len(configs))
width = 0.35

# Create bars
bars_rho = ax1.bar(x - width/2, pearson, width, 
                   label='Pearson $\\rho$', 
                   color=[highlight_color if i == len(configs)-1 else color_rho for i in range(len(configs))],
                   edgecolor='white', linewidth=0.5)

ax2 = ax1.twinx()
bars_rmse = ax2.bar(x + width/2, rmse, width,
                    label='RMSE (kcal/mol)',
                    color=[highlight_color if i == len(configs)-1 else color_rmse for i in range(len(configs))],
                    edgecolor='white', linewidth=0.5)

# Labels and title
ax1.set_xlabel('Model Configuration', fontsize=12)
ax1.set_ylabel('Pearson Correlation ($\\rho$)', color=color_rho, fontsize=12)
ax2.set_ylabel('RMSE (kcal/mol)', color=color_rmse, fontsize=12)
ax1.set_title('Ablation Study on ProTherm Test Set', fontsize=14, pad=20)

# X-ticks
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=0, ha='center')

# Set axis limits
ax1.set_ylim(0.65, 0.90)
ax2.set_ylim(1.15, 1.75)

# Add value labels on bars
def add_labels(ax, bars, values, color, offset=0):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, color=color)

add_labels(ax1, bars_rho, pearson, color_rho, offset=3)
add_labels(ax2, bars_rmse, rmse, color_rmse, offset=3)

# Grid and spines
ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_rho, edgecolor='white', label='Pearson $\\rho$'),
    Patch(facecolor=color_rmse, edgecolor='white', label='RMSE'),
    Patch(facecolor=highlight_color, edgecolor='white', label='Full Model')
]
ax1.legend(handles=legend_elements, loc='upper left', frameon=True, framealpha=0.95)

# Adjust layout and save
plt.tight_layout()
plt.savefig('ablation.pdf', dpi=300, bbox_inches='tight')
plt.show()
