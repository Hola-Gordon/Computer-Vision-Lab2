import numpy as np
from skimage.feature import local_binary_pattern

class LBPExtractor:
    """Extract Local Binary Pattern (LBP) features from images."""
    
    def __init__(self, radius=3, n_points=24, method='uniform'):
        """Initialize LBP feature extractor.
        
        Args:
            radius (int): Radius of circle for sampling points
            n_points (int): Number of sampling points
            method (str): Type of LBP operator
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
    
    def extract_features(self, image):
        """Extract LBP features from an image.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Normalized histogram of LBP codes
        """
        # Compute LBP
        lbp = local_binary_pattern(image, 
                                 self.n_points, 
                                 self.radius, 
                                 method=self.method)
        
        # Calculate histogram
        n_bins = self.n_points + 2 if self.method == 'uniform' else 256
        hist, _ = np.histogram(lbp.ravel(), 
                             bins=n_bins, 
                             range=(0, n_bins),
                             density=True)
        
        return hist