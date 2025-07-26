import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from keras.models import load_model

class PaperResultsGenerator:
    def __init__(self, sequence_length=10):
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / "Data"
        self.model_path = self.base_path / "Model"
        self.plots_path = self.base_path / "Paper_Results"
        self.plots_path.mkdir(exist_ok=True)
        self.sequence_length = sequence_length
        self.class_names = None
        self.model = None

    def load_data_and_model(self):
        # Load test data
        test_df = pd.read_csv(self.data_path / "Splitted_Data" / "test.csv")
        
        # Load label encoder
        with open(self.model_path / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        self.class_names = le.classes_
        
        # Create sequences from test data
        X_test, y_test = self.create_sequences(test_df, le)
        
        # Load trained model
        self.model = load_model(self.model_path / "tcn_gesture_model.h5")
        
        return X_test, y_test

    def create_sequences(self, df, le):
        features = df.drop(columns=['Label']).values
        labels = df['Label'].values
        y_enc = le.transform(labels)
        
        X, y = [], []
        for i in range(len(features) - self.sequence_length + 1):
            X.append(features[i:i+self.sequence_length])
            y.append(y_enc[i+self.sequence_length-1])
        return np.array(X), np.array(y)

    def generate_results(self):
        X_test, y_test = self.load_data_and_model()
        
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        
        # Generate predictions
        y_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=1)
        
        # 1. Classification Report
        self.save_classification_report(y_test, y_pred)
        
        # 2. Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # 3. ROC Curves
        self.plot_roc_curves(y_test, y_prob)
        
        # 4. Model Architecture Summary
        # self.save_model_summary()
        
        # 5. Performance Metrics Summary
        self.save_metrics_summary(test_acc, test_loss)
        
        print("\nAll research paper results generated successfully!")

    def save_classification_report(self, y_true, y_pred):
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        report_df = pd.DataFrame(report).T
        report_df.to_csv(self.plots_path / 'classification_report.csv', float_format='%.4f')
        
        # Create publication-ready table
        class_table = report_df.loc[self.class_names][['precision', 'recall', 'f1-score']]
        class_table = class_table.rename_axis('Gesture').reset_index()
        class_table.to_csv(self.plots_path / 'paper_classification_table.csv', index=False, float_format='%.4f')
        
        print("Classification report saved")

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=self.class_names, 
            yticklabels=self.class_names
        )
        
        plt.title('Normalized Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Gesture', fontsize=12)
        plt.ylabel('True Gesture', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.plots_path / 'confusion_matrix.png', dpi=300)
        plt.close()
        print("Confusion matrix saved")

    def plot_roc_curves(self, y_true, y_prob):
        n_classes = len(self.class_names)
        y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
        
        # Calculate ROC curves and AUC scores
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i], tpr[i], color=color, lw=1.5,
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )
        
        # Plot micro-average ROC
        plt.plot(
            fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4
        )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Multi-class ROC Curves', fontsize=14)
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(self.plots_path / 'roc_curves.png', dpi=300)
        plt.close()
        print("ROC curves saved")

    # def save_model_summary(self):
    #     summary = []
    #     for layer in self.model.layers:
    #         # Use the built-in output_shape attribute which works for all layers
    #         output_shape = layer.output_shape
            
    #         # For layers with multiple outputs, convert to a readable format
    #         if isinstance(output_shape, list):
    #             output_shape = [tuple(shape) for shape in output_shape]
    #         elif isinstance(output_shape, tuple):
    #             # Handle nested tuples from TimeDistributed layers
    #             if any(isinstance(dim, tuple) for dim in output_shape):
    #                 output_shape = tuple(output_shape)
            
    #         summary.append({
    #             'Layer': layer.name,
    #             'Type': layer.__class__.__name__,
    #             'Output Shape': output_shape,
    #             'Parameters': layer.count_params()
    #         })
        
    #     summary_df = pd.DataFrame(summary)
    #     summary_df.to_csv(self.plots_path / 'model_architecture.csv', index=False)
        
        # Create compact version for paper
        # compact_df = summary_df[['Type', 'Parameters']]
        # compact_df = compact_df.rename(columns={
        #     'Type': 'Layer Type',
        #     'Parameters': '# Parameters'
        # })
        # total_params = compact_df['# Parameters'].sum()
        # total_row = pd.DataFrame([{'Layer Type': 'Total', '# Parameters': total_params}])
        # compact_df = pd.concat([compact_df, total_row], ignore_index=True)
        # compact_df.to_csv(self.plots_path / 'paper_model_table.csv', index=False)
        # print("Model architecture summary saved")

    def save_metrics_summary(self, test_acc, test_loss):
        # Get TFLite model size if available
        tflite_size = "N/A"
        tflite_path = self.model_path / "tcn_gesture_model.tflite"
        if tflite_path.exists():
            size_kb = os.path.getsize(tflite_path) / 1024
            tflite_size = f"{size_kb:.1f} KB"
        
        metrics = {
            'Test Accuracy': test_acc,
            'Test Loss': test_loss,
            'Model Size (H5)': self.get_model_size(),
            'TFLite Size': tflite_size,
            'Parameters': self.model.count_params(),
            'Sequence Length': self.sequence_length
        }
        
        with open(self.plots_path / 'performance_metrics.txt', 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        
        # Save as CSV for easier import into papers
        pd.DataFrame([metrics]).to_csv(self.plots_path / 'paper_metrics.csv', index=False)
        print("Performance metrics saved")

    def get_model_size(self):
        model_path = self.model_path / "tcn_gesture_model.h5"
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return f"{size_mb:.2f} MB"


if __name__ == "__main__":
    print("Generating research paper results...")
    generator = PaperResultsGenerator()
    generator.generate_results()