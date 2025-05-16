import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable

# Example data and angle-to-color conversion function
def angle_to_color(angle):
    """Convert angle (in radians or degrees) to an RGB color."""
    # Normalize angle to 0-1 range (assuming angle is in [0, 2π] or [0, 360])
    # Adjust this normalization based on your angle range
    normalized = (angle % (2*np.pi)) / (2*np.pi)  # For radians
    # normalized = (angle % 360) / 360.0  # For degrees
    
    # Use a colormap to convert to RGB
    cmap = plt.cm.hsv  # You can use other colormaps: jet, viridis, etc.
    return cmap(normalized)

# Sample data: angles in radians
angles = np.random.uniform(0, 2*np.pi, 1000)

# Create the histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the histogram
n, bins, patches = ax.hist(angles, bins=36)

# Color each bin patch according to its angle value
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for bin_center, patch in zip(bin_centers, patches):
    color = angle_to_color(bin_center)
    patch.set_facecolor(color)
    patch.set_edgecolor('white')  # Optional: white edges between bars

# Set up the axes
ax.set_xlabel('Angle')
ax.set_ylabel('Frequency')
ax.set_title('Histogram with Color-Coded Angle Bins')

# Add a custom colorbar as a color reference
cmap = plt.cm.hsv
norm = mcolors.Normalize(vmin=0, vmax=2*np.pi)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2)
cbar.set_label('Angle (radians)')

# Use radians or degrees formatting for the colorbar ticks
cbar.set_ticks(np.linspace(0, 2*np.pi, 5))
cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])  # For radians
# cbar.set_ticks(np.linspace(0, 360, 5))  # For degrees
# cbar.set_ticklabels(['0°', '90°', '180°', '270°', '360°'])  # For degrees

plt.savefig('asdf')
plt.show()