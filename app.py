import os
import cv2  # Added missing import
import numpy as np
import gradio as gr
from src.data.data_loader import DataLoader
from src.feature_extractor.glcm_extractor import GLCMExtractor
from src.feature_extractor.lbp_extractor import LBPExtractor
from src.models.classifier import TextureClassifier
from src.utils.preprocessing import preprocess_image

class TextureClassifierApp:
    def __init__(self):
        self.data_loader = DataLoader("data/raw")
        self.glcm_extractor = GLCMExtractor()
        self.lbp_extractor = LBPExtractor()
        self.glcm_classifier = TextureClassifier(self.glcm_extractor)
        self.lbp_classifier = TextureClassifier(self.lbp_extractor)
        self.class_names = ['Stone', 'Brick', 'Wood']
        
    def train_models(self):
        """Train both GLCM and LBP classifiers."""
        print("Loading dataset...")
        images, labels = self.data_loader.load_dataset()
        
        if len(np.unique(labels)) < 2:
            raise ValueError("Need at least 2 classes with images to train. Please check your data/raw directory.")
        
        print("\nTraining GLCM classifier...")
        glcm_accuracy = self.glcm_classifier.train(images, labels)
        
        print("Training LBP classifier...")
        lbp_accuracy = self.lbp_classifier.train(images, labels)
        
        print(f"\nResults:")
        print(f"GLCM Accuracy: {glcm_accuracy:.2f}")
        print(f"LBP Accuracy: {lbp_accuracy:.2f}")
        
        self.save_models()
    
    def predict(self, image, method):
        """Process image and return prediction."""
        try:
            # Convert from numpy array to correct format
            if image is None:
                raise ValueError("No image provided")
                
            # Convert image to grayscale if it's RGB
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                else:  # RGB
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Get prediction
            if method == "GLCM":
                probabilities = self.glcm_classifier.predict(processed_image)
            else:  # LBP
                probabilities = self.lbp_classifier.predict(processed_image)
            
            # Create results dictionary
            return {self.class_names[i]: float(prob) 
                    for i, prob in enumerate(probabilities)}
                    
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {class_name: 0.0 for class_name in self.class_names}
    
    def save_models(self):
        """Save trained models to disk."""
        os.makedirs("trained_models", exist_ok=True)
        self.glcm_classifier.save("trained_models/glcm_model.pkl")
        self.lbp_classifier.save("trained_models/lbp_model.pkl")
        
    def load_models(self):
        """Load trained models from disk."""
        self.glcm_classifier.load("trained_models/glcm_model.pkl")
        self.lbp_classifier.load("trained_models/lbp_model.pkl")
    
    def create_interface(self):
        """Create and return Gradio interface."""
        iface = gr.Interface(
            fn=self.predict,
            inputs=[
                gr.Image(label="Upload Image"),
                gr.Radio(["GLCM", "LBP"], label="Method", value="GLCM")
            ],
            outputs=gr.Label(label="Texture Classification"),
            title="Texture Classifier",
            description="Upload an image to classify its texture as Stone, Brick, or Wood"
        )
        return iface

def main():
    try:
        print("Initializing application...")
        app = TextureClassifierApp()
        
        # Check data directory structure
        raw_path = "data/raw"
        if not os.path.exists(raw_path):
            raise ValueError(f"Data directory not found: {raw_path}")
            
        # Print number of images in each class
        print("\nChecking dataset:")
        for class_name in ['stone', 'brick', 'wood']:
            class_path = os.path.join(raw_path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"{class_name}: {len(images)} images")
            else:
                print(f"{class_name}: directory not found")
        
        if os.path.exists("trained_models/glcm_model.pkl"):
            print("\nLoading existing models...")
            app.load_models()
        else:
            print("\nTraining new models...")
            app.train_models()
        
        print("\nLaunching interface...")
        interface = app.create_interface()
        interface.launch()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure your data directory structure is as follows:")
        print("data/")
        print("  └── raw/")
        print("      ├── stone/   (containing .jpg, .jpeg, or .png files)")
        print("      ├── brick/   (containing .jpg, .jpeg, or .png files)")
        print("      └── wood/    (containing .jpg, .jpeg, or .png files)")

if __name__ == "__main__":
    main()