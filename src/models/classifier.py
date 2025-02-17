import numpy as np
import joblib
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.evaluation.evaluator import Evaluator
from src.utils.preprocessing import preprocess_image
from src.feature_extractor.glcm_extractor import GLCMExtractor


class TextureClassifier:
    def __init__(self, feature_extractor, classifier_type='svm'):
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()
        
        # Choose different classifiers based on the feature extractor type
        if isinstance(feature_extractor, GLCMExtractor):
            self.classifier = SVC(probability=True, kernel='rbf')
        else:  # LBPExtractor
            self.classifier = RandomForestClassifier(n_estimators=100)
    
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
    
    def train(self, features, labels):
        """Train the classifier using pre-extracted features.
        
        Args:
            features: Pre-extracted feature vectors (already extracted, don't extract again)
            labels: Corresponding labels
        """
        # No need to extract features again since they're already extracted
        
        # Scale features
        self.scaler.fit(features)
        X_scaled = self.scaler.transform(features)
        
        # Train classifier
        self.classifier.fit(X_scaled, labels)
        
    def predict(self, image):
        """Predict the texture class of an image."""
        try:
            if image is None:
                raise ValueError("No image provided")
                
            # Preprocess image
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                else:  # RGB
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Extract features
            features = self.feature_extractor.extract_features(image)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            
            return probabilities
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
    
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