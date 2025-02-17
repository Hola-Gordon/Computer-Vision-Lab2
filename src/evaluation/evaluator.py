import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

class Evaluator:
    """Minimal evaluator for texture classification."""
    
    def evaluate_model(self, model, X_test, y_test, class_names, method_name):
        """
        Simple evaluation using just accuracy and confusion matrix.
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\n{method_name} Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return {
            'method': method_name,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
    
    def compare_methods(self, methods_results):
        """Simple comparison of method accuracies."""
        print("\nMethod Comparison:")
        for method, results in methods_results.items():
            print(f"{method} Accuracy: {results['accuracy']:.3f}")