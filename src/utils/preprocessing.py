import cv2
import numpy as np


def preprocess_image(image, target_size=(200, 200)):
    """Preprocess an image for feature extraction.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Desired output size (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Ensure image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to normalize brightness
    image = cv2.equalizeHist(image)
    
    # Optional: Add noise reduction
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image


def create_data_splits(features, labels, train_size=0.7, val_size=0.15):
    """Split data into training, validation, and test sets.
    
    Args:
        features (numpy.ndarray): Feature vectors
        labels (numpy.ndarray): Corresponding labels
        train_size (float): Proportion for training
        val_size (float): Proportion for validation
        
    Returns:
        tuple: (train_data, val_data, test_data) where each is a (features, labels) tuple
    """
    # First split: separate training set
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, train_size=train_size, stratify=labels
    )
    
    # Second split: separate validation and test from temp
    val_ratio = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_ratio, stratify=y_temp
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)