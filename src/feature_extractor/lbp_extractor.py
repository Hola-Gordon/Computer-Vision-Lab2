import numpy as np
import cv2
from skimage.feature import local_binary_pattern


class LBPExtractor:
    """Extract Local Binary Pattern (LBP) features from images."""
    
    def __init__(self):
        self.radius = 2
        self.n_points = 16
        self.method = 'uniform'

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