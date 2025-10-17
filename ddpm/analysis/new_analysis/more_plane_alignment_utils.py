import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy import stats

def create_plane_mesh(normal, center, size=100, points=10):
    """
    Create a mesh grid for visualizing a plane in 3D.
    
    Parameters:
    -----------
    normal : np.ndarray
        Normal vector to the plane, shape [3]
    center : np.ndarray
        Center point of the plane, shape [3]
    size : float
        Size of the plane mesh
    points : int
        Number of points in each dimension of the mesh
        
    Returns:
    --------
    X, Y, Z : np.ndarray
        Mesh grid coordinates for the plane
    """
    # Find two vectors perpendicular to the normal
    if np.abs(normal[0]) > np.abs(normal[1]):
        # If first component is larger, use second component for perpendicular
        v1 = np.array([-normal[2], 0, normal[0]])
    else:
        # Otherwise use first component
        v1 = np.array([0, -normal[2], normal[1]])
    
    # Normalize v1
    v1 = v1 / np.linalg.norm(v1)
    
    # v2 is perpendicular to both normal and v1
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Create a grid of points around the center
    grid_size = size / 2
    grid = np.linspace(-grid_size, grid_size, points)
    
    # Initialize meshgrid arrays
    X = np.zeros((points, points))
    Y = np.zeros((points, points))
    Z = np.zeros((points, points))
    
    # Fill the meshgrid by adding scaled v1 and v2 to the center
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            point = center + a * v1 + b * v2
            X[i, j] = point[0]
            Y[i, j] = point[1]
            Z[i, j] = point[2]
    
    return X, Y, Z


def get_plane_from_pca(pca_obj, data):
    """
    Extract plane information (normal vector and center) from PCA object.
    
    Parameters:
    -----------
    pca_obj : sklearn.decomposition.PCA
        Fitted PCA object
    data : np.ndarray
        Data array that was used to fit the PCA
        
    Returns:
    --------
    normal : np.ndarray
        Normal vector to the best-fitting plane
    center : np.ndarray
        Center point of the data
    """
    # Get the normal vector (3rd principal component)
    normal = pca_obj.components_[2]
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Get the center of the data
    center = np.mean(data, axis=0)
    
    return normal, center


def estimate_plane_size(data):
    """
    Estimate appropriate size for the plane based on data spread.
    
    Parameters:
    -----------
    data : np.ndarray
        Data points, shape [n_points, 3]
        
    Returns:
    --------
    size : float
        Suggested size for the plane visualization
    """
    # Calculate the range of data in first two principal components
    pca = PCA(n_components=2)
    projected = pca.fit_transform(data)
    
    # Calculate the span of the data in the projected space
    x_range = np.max(projected[:, 0]) - np.min(projected[:, 0])
    y_range = np.max(projected[:, 1]) - np.min(projected[:, 1])
    
    # Use the larger of the two ranges, with some margin
    size = 0.5 * max(x_range, y_range)
    
    return size


def plot_planes_with_data(mean_activity_per_bin_stim1, mean_bin_values_stim1, 
                          mean_activity_per_bin_stim2, mean_bin_values_stim2,
                          alpha=0.3, figsize=(10, 8)):
    """
    Create a 3D plot with both planes and data points.
    
    Parameters:
    -----------
    mean_activity_per_bin_stim1 : np.ndarray
        Activity data for stimulus 1, shape [n_bins, 3]
    mean_bin_values_stim1 : np.ndarray
        Bin values for stimulus 1, shape [n_bins]
    mean_activity_per_bin_stim2 : np.ndarray
        Activity data for stimulus 2, shape [n_bins, 3]
    mean_bin_values_stim2 : np.ndarray
        Bin values for stimulus 2, shape [n_bins]
    alpha : float
        Transparency level for the planes
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis object
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Fit PCA to data
    pca_obj_stim1 = PCA(n_components=3).fit(mean_activity_per_bin_stim1)
    pca_obj_stim2 = PCA(n_components=3).fit(mean_activity_per_bin_stim2)
    
    # Get the normal vectors and centers
    normal1, center1 = get_plane_from_pca(pca_obj_stim1, mean_activity_per_bin_stim1)
    normal2, center2 = get_plane_from_pca(pca_obj_stim2, mean_activity_per_bin_stim2)
    
    # Determine appropriate plane size
    all_data = np.vstack([mean_activity_per_bin_stim1, mean_activity_per_bin_stim2])
    plane_size = estimate_plane_size(all_data)
    
    # Create plane mesh grids
    X1, Y1, Z1 = create_plane_mesh(normal1, center1, size=plane_size)
    X2, Y2, Z2 = create_plane_mesh(normal2, center2, size=plane_size)
    
    # Plot data points and connecting lines
    ax.scatter(*mean_activity_per_bin_stim1.T, c=mean_bin_values_stim1, 
               cmap='hsv', marker='o', s=120, label='Selected stimulus 1')
    closed_data1 = np.vstack([mean_activity_per_bin_stim1, mean_activity_per_bin_stim1[0:1]])
    ax.plot(*closed_data1.T, color='grey', alpha=0.7)
    
    ax.scatter(*mean_activity_per_bin_stim2.T, c=mean_bin_values_stim2, 
               cmap='hsv', marker='v', s=120, label='Selected stimulus 2')
    closed_data2 = np.vstack([mean_activity_per_bin_stim2, mean_activity_per_bin_stim2[0:1]])
    ax.plot(*closed_data2.T, color='grey', alpha=0.7)
    
    # Plot planes
    plane1 = ax.plot_surface(X1, Y1, Z1, color='grey', alpha=alpha, shade=False)
    plane2 = ax.plot_surface(X2, Y2, Z2, color='grey', alpha=alpha, shade=False)
    
    # Remove fill effect to make them look like the paper's style
    plane1._facecolors2d = plane1._facecolor3d
    plane1._edgecolors2d = plane1._edgecolor3d
    plane2._facecolors2d = plane2._facecolor3d
    plane2._edgecolors2d = plane2._edgecolor3d
    
    # Calculate angle between planes
    cos_angle = np.abs(np.dot(normal1, normal2))
    angle_degrees = np.arccos(cos_angle) * 180 / np.pi
    
    # Add title with angle information
    # ax.set_title(f'Angle between planes: {angle_degrees:.1f}°')
    
    # Set equal aspect ratio for all axes
    # This makes the visualization look more like the paper
    max_range = max([
        np.max(mean_activity_per_bin_stim1[:, 0]) - np.min(mean_activity_per_bin_stim1[:, 0]),
        np.max(mean_activity_per_bin_stim1[:, 1]) - np.min(mean_activity_per_bin_stim1[:, 1]),
        np.max(mean_activity_per_bin_stim1[:, 2]) - np.min(mean_activity_per_bin_stim1[:, 2])
    ])
    
    mid_x = (np.max(all_data[:, 0]) + np.min(all_data[:, 0])) / 2
    mid_y = (np.max(all_data[:, 1]) + np.min(all_data[:, 1])) / 2
    mid_z = (np.max(all_data[:, 2]) + np.min(all_data[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Set axis labels
    ax.set_xlabel('PC 1', fontsize=18, labelpad=10)
    ax.set_ylabel('PC 2', fontsize=18, labelpad=10)
    ax.set_zlabel('PC 3', fontsize=18, labelpad=10)

    # 2. Increase tick label sizes
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    # 3. Reduce number of ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set maximum of 5 ticks on x-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Set maximum of 5 ticks on y-axis
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))  # Set maximum of 5 ticks on z-axis

    # Add legend
    ax.legend(fontsize = 18)
    
    return fig, ax
