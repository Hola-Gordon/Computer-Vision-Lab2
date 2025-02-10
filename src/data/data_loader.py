import os
import cv2
import numpy as np
from src.utils.preprocessing import preprocess_image


class DataLoader:
    """Handle loading and preprocessing of texture images."""
    
    def __init__(self, data_path):
        """Initialize DataLoader with path to data directory.
        
        Args:
            data_path (str): Path to directory containing texture classes
        """
        self.data_path = data_path
        self.classes = ['stone', 'brick', 'wood']
        
    def load_dataset(self):
        """Load all images from the data directory.
        
        Returns:
            tuple: (images, labels) where images is a numpy array of preprocessed images
                  and labels is a numpy array of corresponding class indices
        """
        images = []
        labels = []
        
        # Load images from each class directory
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            
            # Skip if directory doesn't exist
            if not os.path.exists(class_path):
                print(f"Warning: Directory {class_path} not found")
                continue
                
            # Process each image in the class directory
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        # Read and preprocess image
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = preprocess_image(img)
                            images.append(img)
                            labels.append(class_idx)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        
        if not images:
            raise ValueError("No valid images found in the data directory")
            
        return np.array(images), np.array(labels)
    
    def save_processed_image(self, image, class_name, filename):
        """Save a processed image to the processed directory.
        
        Args:
            image (numpy.ndarray): Processed image
            class_name (str): Name of the class (stone/brick/wood)
            filename (str): Name for the processed image file
        """
        processed_dir = os.path.join('data/processed', class_name)
        os.makedirs(processed_dir, exist_ok=True)
        
        output_path = os.path.join(processed_dir, filename)
        cv2.imwrite(output_path, image)