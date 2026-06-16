import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import matplotlib.patches as patches

# Create a figure and axis
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')  # Hide axes for a clean diagram

# Helper function to add a box
def add_box(x, y, width, height, text, color='lightblue', edgecolor='blue'):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", fc=color, ec=edgecolor, lw=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, wrap=True)

# Helper function to add arrow
def add_arrow(start, end, color='black'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=ArrowStyle('->', head_length=0.4, head_width=0.3), color=color, lw=2))

# Left Side: Inputs and Encoding
add_box(0.5, 5.5, 2.5, 1.5, "Wild-Type\nEnzyme Sequence", color='lightgreen', edgecolor='green')
add_arrow((3, 6.25), (3.5, 6.25))  # Arrow to DGTN
add_box(4, 5, 3, 2, "DGTN Encoding\n- Sequence Embeddings (ESM-2)\n- Structural Graph Diffusion", color='lightblue', edgecolor='blue')
add_arrow((7.1, 6), (7.6, 6))  # Small arrow to latent graph
add_box(8, 5.5, 2.5, 1, "Latent Graph\nRepresentation", color='lightyellow', edgecolor='orange')

# Center: Core Framework
add_box(4, 2, 6, 2.5, "Enzyme Mutation Game\nMonte Carlo Tree Search (MCTS)\n- Selection\n- Expansion\n- Simulation (DGTN Fitness)\n- Backpropagation\nNavigates Epistatic Dependencies", color='lightcoral', edgecolor='red')
# Inset for landscape
ax.add_patch(patches.Ellipse((9, 2.5), 1, 1, color='gray', alpha=0.3))  # Placeholder for rugged landscape icon
ax.text(9, 2.5, "Rugged\nLandscape", ha='center', va='center', fontsize=8)

# Arrows from encoding to center
add_arrow((5.5, 5), (5.5, 4.5), color='blue')

# Right Side: Outputs
add_box(11, 4.5, 2.5, 1.5, "High-Fitness\nMutation Pathways", color='lightgreen', edgecolor='green')
add_box(11, 2, 2.5, 1.5, "Reconstructed\nFitness Landscape\n(Active Sampling +\nGaussian Process)", color='lightyellow', edgecolor='orange')
add_box(11, 0.5, 2.5, 1, "Explainable Insights\n- Residue Interactions\n- Mechanistic Logic", color='lightblue', edgecolor='blue')

# Arrows from center to outputs
add_arrow((10, 3.5), (10.9, 5.25))  # To pathways
add_arrow((10, 3), (10.9, 3))  # To landscape
add_arrow((10, 2.5), (10.9, 1.25))  # To insights

# Title
ax.text(7, 7.5, "Overview of MCTS-DGTN Framework for Enzyme Mutagenesis", ha='center', fontsize=14, fontweight='bold')

# Save or show the diagram
plt.savefig('methodology_diagram.png')  # Save to file
plt.show()  # Or display in interactive mode
