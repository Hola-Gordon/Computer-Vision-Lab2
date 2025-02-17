import os
import cv2 
import numpy as np
import gradio as gr
from src.data.data_loader import DataLoader
from src.feature_extractor.glcm_extractor import GLCMExtractor
from src.feature_extractor.lbp_extractor import LBPExtractor
from src.models.classifier import TextureClassifier
from src.evaluation.evaluator import Evaluator
from sklearn.model_selection import train_test_split
import json

class TextureClassifierApp:
    def __init__(self):
        self.data_loader = DataLoader("data/raw")  # Use raw data for training
        self.class_names = self.data_loader.classes
        self.glcm_extractor = GLCMExtractor()
        self.lbp_extractor = LBPExtractor()
        self.glcm_classifier = TextureClassifier(self.glcm_extractor, 'svm')
        self.lbp_classifier = TextureClassifier(self.lbp_extractor, 'rf')

    def predict(self, image, method):
        if image is None:
            return None
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.resize(image, (200, 200))
        if method == "GLCM":
            probabilities = self.glcm_classifier.predict(image)
        else:
            probabilities = self.lbp_classifier.predict(image)
        
        return {self.class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)}

    def load_example_images(self):
        """Load example images from data/examples directory only"""
        examples = {}
        for class_name in self.class_names:
            class_path = os.path.join("data/examples", class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (200, 200))
                            examples[class_name] = img
                            break
        return examples

    def create_interface(self):
        example_images = self.load_example_images()
        
        with gr.Blocks() as iface:
            with gr.Row():
                gr.Markdown("# Texture Classifier")
            
            # Top section: Image upload and classification
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="Upload Image", type="numpy", height=300)
                    method_choice = gr.Dropdown(
                        choices=["GLCM", "LBP"],
                        label="Classification Method",
                        value="GLCM"
                    )
                    classify_btn = gr.Button("Classify")
                    result_label = gr.Label(label="Classification Result")
            
            # Middle section: Example images
            gr.Markdown("### Example Images from Each Category")
            with gr.Row():
                for class_name, img in example_images.items():
                    with gr.Column():
                        gr.Markdown(f"##### {class_name.capitalize()}")
                        example_image = gr.Image(value=img, height=200)
                        example_image.select(
                            fn=lambda x: x,
                            inputs=example_image,
                            outputs=input_img
                        )
            
            # Bottom section: Model metrics
            gr.Markdown("### Model Performance Metrics")
            with gr.Row():
                for method in ["GLCM", "LBP"]:
                    with gr.Column():
                        gr.Markdown(f"#### {method} Model Performance")
                        metrics = self.metrics.get(method, {})
                        confusion_matrix = np.array(metrics.get('confusion_matrix', []))
                        
                        metrics_md = f"""
                        - **Accuracy**: {metrics.get('accuracy', 0):.2f}%
                        - **Precision**: {metrics.get('precision', 0):.2f}%
                        - **Recall**: {metrics.get('recall', 0):.2f}%
                        
                        Per-class Accuracy:
                        """
                        for i, acc in enumerate(metrics.get('class_accuracies', [])):
                            metrics_md += f"- {self.class_names[i]}: {acc:.2f}%\n"
                        
                        if confusion_matrix.size > 0:
                            metrics_md += "\nConfusion Matrix:\n```\n"
                            for row in confusion_matrix:
                                metrics_md += f"{row}\n"
                            metrics_md += "```"
                        
                        gr.Markdown(metrics_md)
            
            classify_btn.click(
                fn=self.predict,
                inputs=[input_img, method_choice],
                outputs=result_label
            )
            
        return iface

    def train_models(self):
        """Train models using data from raw directory"""
        images, labels = self.data_loader.load_dataset()  # Uses raw data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.3, random_state=42
        )
        
        evaluator = Evaluator()
        self.metrics = {}
        
        for name, clf in [("GLCM", self.glcm_classifier), ("LBP", self.lbp_classifier)]:
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
            
            clf.train(X_train_features, y_train)
            self.metrics[name] = evaluator.evaluate_model(
                clf.classifier,
                clf.scaler.transform(X_test_features),
                y_test,
                name
            )
        
        self.save_models()
        return self.metrics

    def save_models(self):
        os.makedirs("trained_models", exist_ok=True)
        self.glcm_classifier.save("trained_models/glcm_model.pkl")
        self.lbp_classifier.save("trained_models/lbp_model.pkl")
        with open("trained_models/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load_models(self):
        self.glcm_classifier.load("trained_models/glcm_model.pkl")
        self.lbp_classifier.load("trained_models/lbp_model.pkl")
        if os.path.exists("trained_models/metrics.json"):
            with open("trained_models/metrics.json", 'r') as f:
                self.metrics = json.load(f)

def main():
    app = TextureClassifierApp()
    
    if os.path.exists("trained_models/glcm_model.pkl"):
        app.load_models()
    else:
        app.metrics = app.train_models()
    
    interface = app.create_interface()
    interface.launch()

if __name__ == "__main__":
    main()