import os
import cv2 
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from src.data.data_loader import DataLoader
from src.feature_extractor.glcm_extractor import GLCMExtractor
from src.feature_extractor.lbp_extractor import LBPExtractor
from src.models.classifier import TextureClassifier
from src.evaluation.evaluator import Evaluator
from sklearn.model_selection import train_test_split
import json


class TextureClassifierApp:
    def __init__(self):
        self.data_loader = DataLoader("data/raw")
        self.glcm_extractor = GLCMExtractor()
        self.lbp_extractor = LBPExtractor()
        self.glcm_classifier = TextureClassifier(self.glcm_extractor, 'svm')
        self.lbp_classifier = TextureClassifier(self.lbp_extractor, 'rf')

    def predict(self, image, method):
        """Classify an image using selected method."""
        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Preprocess and predict
        image = cv2.resize(image, (200, 200))
        if method == "GLCM":
            probabilities = self.glcm_classifier.predict(image)
        else:
            probabilities = self.lbp_classifier.predict(image)
        
        return {self.class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)}

    def train_models(self):
        """Train models and store metrics."""
        images, labels = self.data_loader.load_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.3, random_state=42
        )
        
        evaluator = Evaluator()
        self.metrics = {}  # Store metrics for display
        
        # Train and evaluate both methods
        for name, clf in [("GLCM", self.glcm_classifier), ("LBP", self.lbp_classifier)]:
            # Extract features
            X_train_features = []
            X_test_features = []
            
            for image in X_train:
                features = clf.feature_extractor.extract_features(image)
                X_train_features.append(features)
            for image in X_test:
                features = clf.feature_extractor.extract_features(image)
                X_test_features.append(features)
                
            X_train_features = np.array(X_train_features)
            X_test_features = np.array(X_test_features)
            
            # Train
            clf.train(X_train_features, y_train)
            
            # Evaluate and store metrics
            self.metrics[name] = evaluator.evaluate_model(
                clf.classifier,
                X_test_features,
                y_test,
                name
            )
        
        self.save_models()
        return self.metrics

    def save_models(self):
        """Save trained models."""
        os.makedirs("trained_models", exist_ok=True)
        self.glcm_classifier.save("trained_models/glcm_model.pkl")
        self.lbp_classifier.save("trained_models/lbp_model.pkl")
    
    def load_models(self):
        """Load trained models."""
        self.glcm_classifier.load("trained_models/glcm_model.pkl")
        self.lbp_classifier.load("trained_models/lbp_model.pkl")

    def create_interface(self):
        """Create Gradio interface with metrics display."""
        with gr.Blocks() as iface:
            gr.Markdown("# Texture Classifier")
            
            # Display metrics if available
            if self.metrics:
                with gr.Row():
                    for method, metrics in self.metrics.items():
                        with gr.Column():
                            gr.Markdown(f"### {method} Performance")
                            gr.Markdown(f"Accuracy: {metrics['accuracy']}%")
                            gr.Markdown(f"Precision: {metrics['precision']}%")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Image")
                    method = gr.Radio(
                        ["GLCM", "LBP"], 
                        label="Method",
                        value="GLCM"
                    )
                    classify_btn = gr.Button("Classify")
                
                output_label = gr.Label(label="Classification Result")
            
            classify_btn.click(
                fn=self.predict,
                inputs=[input_image, method],
                outputs=output_label
            )
            
        return iface

def main():
    app = TextureClassifierApp()
    
    # Train or load models
    if os.path.exists("trained_models/glcm_model.pkl"):
        app.load_models()
        # Load stored metrics if they exist
        if os.path.exists("trained_models/metrics.json"):
            with open("trained_models/metrics.json", 'r') as f:
                app.metrics = json.load(f)
    else:
        app.metrics = app.train_models()
        # Save metrics
        with open("trained_models/metrics.json", 'w') as f:
            json.dump(app.metrics, f)
    
    # Launch interface
    interface = app.create_interface()
    interface.launch()

if __name__ == "__main__":
    main()