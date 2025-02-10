import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TextureClassifier:
    """Classifier for texture recognition using extracted features."""
    
    def __init__(self, feature_extractor):
        """Initialize the classifier.
        
        Args:
            feature_extractor: Feature extraction object (GLCM or LBP)
        """
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()
        self.classifier = SVC(probability=True, kernel='rbf')
    
    def extract_features_batch(self, images):
        """Extract features from a batch of images.
        
        Args:
            images (list): List of images to process
            
        Returns:
            numpy.ndarray: Array of feature vectors
        """
        features = []
        for image in images:
            feat = self.feature_extractor.extract_features(image)
            features.append(feat)
        return np.array(features)
    
    def train(self, images, labels, test_size=0.2):
        """Train the classifier.
        
        Args:
            images (numpy.ndarray): Training images
            labels (numpy.ndarray): Corresponding labels
            test_size (float): Proportion of dataset to use for testing
            
        Returns:
            float: Classification accuracy on test set
        """
        # Extract features from all images
        features = self.extract_features_batch(images)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Return accuracy
        return self.classifier.score(X_test, y_test)
    
    def predict(self, image):
        """Predict the texture class of an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Class probabilities
        """
        features = self.feature_extractor.extract_features(image)
        features = features.reshape(1, -1)
        features = self.scaler.transform(features)
        return self.classifier.predict_proba(features)[0]
    
    def save(self, path):
        """Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'scaler': self.scaler,
            'classifier': self.classifier
        }
        joblib.dump(model_data, path)
    
    def load(self, path):
        """Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        model_data = joblib.load(path)
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']