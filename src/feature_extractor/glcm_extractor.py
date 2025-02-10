import numpy as np
from skimage.feature import graycomatrix, graycoprops

class GLCMExtractor:
    """Extract Gray Level Co-occurrence Matrix (GLCM) features from images."""
    
    def __init__(self, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Initialize GLCM feature extractor.
        
        Args:
            distances (list): List of pixel pair distances
            angles (list): List of pixel pair angles in radians
        """
        self.distances = distances
        self.angles = angles
        self.properties = ['contrast', 'dissimilarity', 'homogeneity', 
                          'energy', 'correlation', 'ASM']
    
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