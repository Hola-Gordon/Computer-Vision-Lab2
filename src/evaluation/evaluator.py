from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

class Evaluator:
    def evaluate_model(self, model, X_test, y_test, method_name):
        """Evaluate model performance with comprehensive metrics."""
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate confusion matrix and per-class accuracy
        conf_matrix = confusion_matrix(y_test, y_pred)
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        return {
            'method': method_name,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'class_accuracies': [round(acc * 100, 2) for acc in per_class_acc],
            'confusion_matrix': conf_matrix.tolist()
        }