import numpy as np
import joblib
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.feature_extractor.glcm_extractor import GLCMExtractor


class TextureClassifier:
    def __init__(self, feature_extractor, classifier_type='svm'):
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()
        
        # Configure classifiers with appropriate parameters for each feature type
        if isinstance(feature_extractor, GLCMExtractor):
            self.classifier = SVC(
                probability=True, 
                kernel='rbf', 
                C=10,
                gamma='scale',
                class_weight='balanced'
            )
        else:  # LBPExtractor
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )
    
    def train(self, features, labels):
        """Train the classifier with feature normalization."""
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Print feature statistics for debugging
        print(f"\nFeature statistics for {self.feature_extractor.__class__.__name__}:")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Feature mean before scaling: {np.mean(features):.3f}")
        print(f"Feature std before scaling: {np.std(features):.3f}")
        print(f"Feature mean after scaling: {np.mean(X_scaled):.3f}")
        print(f"Feature std after scaling: {np.std(X_scaled):.3f}")
        
        # Train classifier
        self.classifier.fit(X_scaled, labels)
    
    def predict(self, image):
        """Predict with proper feature scaling."""
        # Ensure image is grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Extract and scale features
        features = self.feature_extractor.extract_features(image)
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        return probabilities
    
    def save(self, path):
        """Save the trained model to disk."""
        model_data = {
            'scaler': self.scaler,
            'classifier': self.classifier
        }
        joblib.dump(model_data, path)
    
    def load(self, path):
        """Load a trained model from disk."""
        model_data = joblib.load(path)
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']