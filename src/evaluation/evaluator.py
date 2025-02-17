from sklearn.metrics import accuracy_score, precision_score

class Evaluator:
    def evaluate_model(self, model, X_test, y_test, method_name):
        """Calculate accuracy and precision only."""
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Add zero_division parameter to handle the warning
        precision = precision_score(y_test, y_pred, 
                                 average='weighted', 
                                 zero_division=0)
        
        return {
            'method': method_name,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2)
        }