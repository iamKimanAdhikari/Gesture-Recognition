import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from pathlib import Path
from data_preprocessing._setup_logging import SetupLogs  
from . import build_tcn_model

class TrainTCNModel:
    def __init__(self, sequence_length=10):
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / "Data"
        self.model_path = self.base_path / "Model"
        self.model_path.mkdir(exist_ok=True)
        self.plots_path = self.base_path / "Plots"
        self.plots_path.mkdir(exist_ok=True)
        self.sequence_length = sequence_length
        
        # Setup logging
        setup = SetupLogs("TCN Training")
        self.logger = setup.setup_logging()
        
        # Load normalized data
        self.load_data()
        
    def load_data(self):
        """Load normalized datasets"""
        self.train_df = pd.read_csv(self.data_path / "Splitted_Data" / "train.csv")
        self.val_df = pd.read_csv(self.data_path / "Splitted_Data" / "val.csv")
        self.test_df = pd.read_csv(self.data_path / "Splitted_Data" / "test.csv")
        
        # Load scaler for future inference
        scaler_path = self.data_path / "Normalized_Data" / "_normalized_dfs.pkl"
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        self.logger.info("Loaded normalized datasets and scaler")
        
    def create_sequences(self, df):
        """Create sequences with temporal ordering"""
        features = df.drop(columns=['Label']).values
        labels = df['Label'].values
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        # Create sequences with temporal ordering
        sequences = []
        sequence_labels = []
        
        for i in range(len(features) - self.sequence_length + 1):
            seq = features[i:i+self.sequence_length]
            label = encoded_labels[i+self.sequence_length-1]  # Label for the last element
            sequences.append(seq)
            sequence_labels.append(label)
            
        return np.array(sequences), np.array(sequence_labels), le
    
    def train_and_evaluate(self):
        """Train and evaluate the TCN model"""
        start_time = time.time()
        
        # Create sequences
        X_train, y_train, le = self.create_sequences(self.train_df)
        X_val, y_val, _ = self.create_sequences(self.val_df)
        X_test, y_test, _ = self.create_sequences(self.test_df)
        
        # Save label encoder
        with open(self.model_path / "label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        
        # Model parameters
        input_shape = (self.sequence_length, X_train.shape[2])
        num_classes = len(le.classes_)
        
        # Build TCN model
        model = build_tcn_model(
            input_shape=input_shape,
            num_classes=num_classes,
            num_blocks=3,
            filters=16,  # Reduced for ESP32 compatibility
            kernel_size=3,
            weight_decay=1e-4
        )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        self.logger.info("Model summary:")
        model.summary(print_fn=self.logger.info)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        self.logger.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save model
        model_path = self.model_path / "tcn_gesture_model.h5"
        model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Generate reports and plots
        self.generate_reports(model, X_test, y_test, le, history)
        
        # Convert to TensorFlow Lite for ESP32
        self.convert_to_tflite(model)
        
        duration = time.time() - start_time
        self.logger.info(f"Training completed in {duration:.2f} seconds")
        
        return model
    
    def generate_reports(self, model, X_test, y_test, le, history):
        # Classification report
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        report = classification_report(
            y_test, 
            y_pred_classes,
            target_names=le.classes_
        )
        self.logger.info("Classification Report:\n" + report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.plots_path / 'confusion_matrix.png')
        plt.close()
        
        # Training history plots
        plt.figure(figsize=(12, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'training_history.png')
        plt.close()

        self.plot_roc_curve(y_test, y_pred, le)
        
    def plot_roc_curve(self, y_test, y_pred, le):
        """Generate multi-class ROC curve with AUC values"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # Binarize the true labels
        classes = le.classes_
        n_classes = len(classes)
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)
        
        # Plot class-specific ROC curves
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of {classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        # Plot random guessing line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Format plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        plt.savefig(self.plots_path / 'roc_curve.png')
        plt.close()
        self.logger.info("ROC curve plot saved to roc_curve.png")
    
    def convert_to_tflite(self, model):
        """Convert model to TensorFlow Lite format for ESP32"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_path = self.model_path / "tcn_gesture_model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        self.logger.info(f"Model converted to TFLite and saved to {tflite_path}")
        
        # Calculate model size
        model_size_kb = len(tflite_model) / 1024
        self.logger.info(f"TFLite model size: {model_size_kb:.2f} KB")

