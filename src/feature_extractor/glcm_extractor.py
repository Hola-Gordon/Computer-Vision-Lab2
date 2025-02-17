import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops


class GLCMExtractor:
    """Extract Gray Level Co-occurrence Matrix (GLCM) features from images."""
    
    def __init__(self):
        self.distances = [1, 2]  
        self.angles = [0, np.pi/2] 
        self.properties = [
            'contrast', 'homogeneity', 
            'energy', 'correlation' 
        ]

    def extract_features(self, image):
        """Extract GLCM features from an image.
        
        Args:
            image (numpy.ndarray): Grayscale image
            
        Returns:
            numpy.ndarray: Feature vector of GLCM properties
        """
        # Calculate GLCM
        glcm = graycomatrix(image, 
                           distances=self.distances,
                           angles=self.angles,
                           levels=256,
                           symmetric=True,
                           normed=True)
        
        # Extract features from GLCM
        features = []
        for prop in self.properties:
            feature = graycoprops(glcm, prop).ravel()
            features.extend(feature)
            
        return np.array(features)