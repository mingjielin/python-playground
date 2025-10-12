import matplotlib.pyplot as plt
import numpy as np

# Simulate attention matrices (replace with real data)
n = 50
np.random.seed(42)

# Early: local attention
early = np.zeros((n, n))
for i in range(n):
    for j in range(max(0, i-3), min(n, i+4)):
        early[i, j] = np.exp(-abs(i-j)/2.0)
early += 0.05 * np.random.rand(n, n)

# Middle: emerging long-range
middle = early.copy()
middle[10:15, 35:40] = 0.6
middle[35:40, 10:15] = 0.6
middle += 0.03 * np.random.rand(n, n)

# Late: structure-aware long-range
late = middle.copy()
late[5, 45] = late[45, 5] = 0.85
late[20, 48] = late[48, 20] = 0.78
late += 0.02 * np.random.rand(n, n)

# Normalize
early /= early.max()
middle /= middle.max()
late /= late.max()

# Create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# Plot heatmaps
im0 = axes[0].imshow(early, cmap='viridis', vmin=0, vmax=1, aspect='auto')
im1 = axes[1].imshow(middle, cmap='viridis', vmin=0, vmax=1, aspect='auto')
im2 = axes[2].imshow(late, cmap='viridis', vmin=0, vmax=1, aspect='auto')

# Titles and labels
titles = ['Early Layer\n(Local Focus)', 'Middle Layer\n(Emerging Long-Range)', 'Late Layer\n(Diffused Attention)']
for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Residue Index')
axes[0].set_ylabel('Residue Index')

# Adjust layout to make room for colorbar
plt.subplots_adjust(right=0.85)  # Leave 15% space on the right

# Add colorbar in the reserved space
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label('Attention Weight', rotation=270, labelpad=15)

# Save
plt.savefig('attention.pdf', dpi=300, bbox_inches='tight')
plt.show()
