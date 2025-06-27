import numpy as np
from scipy import stats
from sklearn.decomposition import PCA


def fit_plane_to_data(data):
    """
    Fit a best 2D plane to 3D data points using PCA.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array of shape [n_points, 3] containing 3D points
        
    Returns:
    --------
    normal_vector : np.ndarray
        Normal vector to the best-fitting plane, shape [3]
    center : np.ndarray
        Center point of the data, shape [3]
    """
    # Center the data
    center = np.mean(data, axis=0)
    centered_data = data - center
    
    # Apply PCA to find the principal components
    pca = PCA(n_components=3)
    pca.fit(centered_data)
    
    # The normal vector to the best-fitting plane is the 3rd principal component
    # (the direction with the least variance)
    normal_vector = pca.components_[2]
    
    # Ensure the normal vector is normalized
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    return normal_vector, center


def compute_plane_normal_svd(data):
    """
    Alternative method to find plane normal using SVD.
    May be more numerically stable for some datasets.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array of shape [n_points, 3] containing 3D points
        
    Returns:
    --------
    normal : np.ndarray
        Normal vector to the best-fitting plane, shape [3]
    center : np.ndarray
        Center point of the data, shape [3]
    """
    # Center the data
    center = np.mean(data, axis=0)
    centered_data = data - center
    
    # Compute SVD
    U, S, Vh = np.linalg.svd(centered_data, full_matrices=False)
    
    # The right singular vector corresponding to the smallest singular value
    # gives the normal to the best-fitting plane
    normal = Vh[2, :]
    
    # Ensure the normal vector is normalized
    normal = normal / np.linalg.norm(normal)
    
    return normal, center


def cosine_between_planes(normal1, normal2):
    """
    Calculate the cosine of the angle between two planes defined by their normal vectors.
    
    Parameters:
    -----------
    normal1 : np.ndarray
        Normal vector of the first plane, shape [3]
    normal2 : np.ndarray
        Normal vector of the second plane, shape [3]
        
    Returns:
    --------
    cos_angle : float
        Cosine of the angle between the planes
    """
    # The angle between two planes is the angle between their normal vectors
    # We use the absolute value of the dot product because planes with opposite 
    # normal vectors are the same plane
    cos_angle = np.abs(np.dot(normal1, normal2))
    
    # Ensure numerical stability
    cos_angle = min(1.0, max(0.0, cos_angle))
    
    return cos_angle


def angle_between_planes_degrees(normal1, normal2):
    """
    Calculate the angle between two planes in degrees.
    
    Parameters:
    -----------
    normal1 : np.ndarray
        Normal vector of the first plane, shape [3]
    normal2 : np.ndarray
        Normal vector of the second plane, shape [3]
        
    Returns:
    --------
    angle : float
        Angle between the planes in degrees
    """
    cos_angle = cosine_between_planes(normal1, normal2)
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle


def compare_planes(data1, data2):
    """
    Compare two sets of 3D points by finding their best-fitting planes
    and calculating the cosine of the angle between them.
    
    Parameters:
    -----------
    data1 : np.ndarray
        First data array of shape [n_points, 3]
    data2 : np.ndarray
        Second data array of shape [n_points, 3]
        
    Returns:
    --------
    cos_angle : float
        Cosine of the angle between the planes
    angle_degrees : float
        Angle between the planes in degrees
    normal1 : np.ndarray
        Normal vector of the first plane
    normal2 : np.ndarray
        Normal vector of the second plane
    """
    # Fit planes to both datasets
    normal1, center1 = fit_plane_to_data(data1)
    normal2, center2 = fit_plane_to_data(data2)
    
    # Calculate the cosine of the angle between the planes
    cos_angle = cosine_between_planes(normal1, normal2)
    
    # Calculate the angle in degrees
    angle_degrees = angle_between_planes_degrees(normal1, normal2)
    
    return cos_angle, angle_degrees, normal1, normal2



def bin_values(activity_tensor, binning_tensor, n_bins = 8):
    
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize arrays to store results
    mean_bin_values = np.zeros(n_bins)
    mean_activity_per_bin = np.zeros((n_bins, activity_tensor.shape[1]))

    for i in range(n_bins):
        if i < n_bins - 1:
            # Standard bin
            mask = (binning_tensor >= bin_edges[i]) & (binning_tensor < bin_edges[i+1])
        else:
            # Last bin needs to handle the circular wrap-around
            mask = (binning_tensor >= bin_edges[i]) | (binning_tensor < bin_edges[0])
        
        if mask.sum() > 0:  # Check if the bin has any elements
            # Calculate circular mean for the bin values
            bin_values = binning_tensor[mask]
            mean_bin_values[i] = stats.circmean(bin_values, low=-np.pi, high=np.pi)
            
            # Calculate mean of activity_tensor for this bin
            mean_activity_per_bin[i] = np.mean(activity_tensor[mask], axis=0)
    
    return mean_activity_per_bin, mean_bin_values

