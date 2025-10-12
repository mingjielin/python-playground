import matplotlib.pyplot as plt
import numpy as np

# Data from the table
methods = ['DeepDDG', 'MutFormer', 'ESM-1v', 'DGTN']
skempi = [0.62, 0.67, 0.71, 0.78]
ssym = [0.58, 0.63, 0.66, 0.73]
fireprotdb = [0.66, 0.70, 0.74, 0.80]

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Set width of bars and positions
bar_width = 0.25
index = np.arange(len(methods))

# Create bars
bars1 = ax.bar(index - bar_width, skempi, bar_width, label='SKEMPI 2.0', color='#4E79A7')
bars2 = ax.bar(index, ssym, bar_width, label='Ssym', color='#F28E2B')
bars3 = ax.bar(index + bar_width, fireprotdb, bar_width, label='FireProtDB', color='#E15759')

# Add labels, title, and legend
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Pearson Correlation (ρ)', fontsize=12)
ax.set_title('Cross-Dataset Generalization (Trained on ProTherm)', fontsize=14, pad=20)
ax.set_xticks(index)
ax.set_xticklabels(methods)
ax.legend()

# Add value labels on top of each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Set y-axis limits and grid
ax.set_ylim(0.5, 0.85)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
# plt.savefig('cross_dataset_generalization.pdf', dpi=300, bbox_inches='tight')
plt.savefig('plot1.pdf', 
            format='pdf', 
            dpi=300,               # ignored for PDF but harmless
            bbox_inches='tight',
            transparent=False)     # set True if you need transparent background
plt.show()
