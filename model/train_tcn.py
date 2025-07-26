import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode

import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing._setup_logging import SetupLogs
from tcn_model import build_tcn_model  # assumes tcn_model.py is in the same folder

class TrainTCNModel:
    def __init__(self, sequence_length=10):
        self.base_path       = Path(__file__).parent
        self.data_path       = self.base_path / "Data"
        self.model_path      = self.base_path / "Model";      self.model_path.mkdir(exist_ok=True)
        self.plots_path      = self.base_path / "Plots";      self.plots_path.mkdir(exist_ok=True)
        self.sequence_length = sequence_length

        setup = SetupLogs("TCN Training")
        self.logger = setup.setup_logging()
        self.load_data()

    def load_data(self):
        self.train_df = pd.read_csv(self.data_path / "Splitted_Data" / "train.csv")
        self.val_df   = pd.read_csv(self.data_path / "Splitted_Data" / "val.csv")
        self.test_df  = pd.read_csv(self.data_path / "Splitted_Data" / "test.csv")
        with open(self.data_path / "Normalized_Data" / "_normalized_dfs.pkl","rb") as f:
            self.scaler = pickle.load(f)
        self.logger.info("Loaded normalized datasets and scaler")

    def create_sequences(self, df):
        features = df.drop(columns=['Label']).values
        labels   = df['Label'].values
        le       = LabelEncoder()
        y_enc    = le.fit_transform(labels)
        X, y = [], []
        for i in range(len(features) - self.sequence_length + 1):
            X.append(features[i:i+self.sequence_length])
            y.append(y_enc[i+self.sequence_length-1])
        return np.array(X), np.array(y), le

    def print_model_table(self, model):
        """Print architecture as a table"""
        rows = []
        for layer in model.layers:
            rows.append({
                'Layer': f"{layer.name} ({layer.__class__.__name__})",
                'Output Shape': layer.output_shape,
                'Params': layer.count_params()
            })
        df = pd.DataFrame(rows)
        print("\n=== Model Architecture ===")
        print(df.to_string(index=False))

    def train_and_evaluate(self):
        t0 = time.time()
        X_tr, y_tr, le = self.create_sequences(self.train_df)
        X_va, y_va, _  = self.create_sequences(self.val_df)
        X_te, y_te, _  = self.create_sequences(self.test_df)

        # Save label encoder
        with open(self.model_path / "label_encoder.pkl","wb") as f:
            pickle.dump(le, f)

        # Build & compile
        model = build_tcn_model(
            input_shape=(self.sequence_length, X_tr.shape[2]),
            num_classes=len(le.classes_),
            num_blocks=3, filters=16, kernel_size=3
        )
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Print architecture table
        self.print_model_table(model)

        # Callbacks
        cbs = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]

        # Fit
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=100, batch_size=32,
            callbacks=cbs, verbose=1
        )

        # Capture final losses & accuracies
        tr_acc, tr_loss = hist.history['accuracy'][-1], hist.history['loss'][-1]
        va_acc, va_loss = hist.history['val_accuracy'][-1], hist.history['val_loss'][-1]
        te_loss, te_acc = model.evaluate(X_te, y_te, verbose=0)

        # Summary table
        df_metrics = pd.DataFrame([
            {'Dataset':'Train',      'Accuracy':tr_acc, 'Loss':tr_loss},
            {'Dataset':'Validation', 'Accuracy':va_acc, 'Loss':va_loss},
            {'Dataset':'Test',       'Accuracy':te_acc, 'Loss':te_loss},
        ])
        print("\n=== Overall Accuracy & Loss ===")
        print(df_metrics.to_string(index=False, float_format="%.4f"))

        # Save model
        model.save(self.model_path / "tcn_gesture_model.h5")

        # Reports & plots
        self.generate_reports(model, X_te, y_te, le, hist)
        self.convert_to_tflite(model)

        self.logger.info(f"Training done in {time.time()-t0:.1f}s")
        return model

    def generate_reports(self, model, X_test, y_test, le, history):
        # Predictions
        y_prob = model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=1)

        # Classification report table
        rpt_dict = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True
        )
        rpt_df = pd.DataFrame(rpt_dict).T[['precision','recall','f1-score','support']].round(4)
        print("\n=== Per‑Class Metrics ===")
        print(rpt_df.to_string(float_format="%.4f"))
        rpt_df.to_csv(self.plots_path / 'classification_report.csv')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Gesture')
        plt.ylabel('True Gesture')
        plt.tight_layout()
        plt.savefig(self.plots_path / 'confusion_matrix.png')
        plt.close()

        # ROC curves
        self.plot_roc_curve(y_test, y_prob, le)

    def plot_roc_curve(self, y_test, y_prob, le):
        n = len(le.classes_)
        y_bin = np.eye(n)[y_test]

        # Per-class & micro
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:,i], y_prob[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        # Macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n
        fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
        roc_auc['macro'] = auc(all_fpr, mean_tpr)

        # Plot
        plt.figure(figsize=(10,8))
        plt.plot(fpr['micro'], tpr['micro'],
                 linestyle=':', linewidth=4,
                 label=f'Micro‑avg (AUC = {roc_auc["micro"]:.2f})')
        plt.plot(fpr['macro'], tpr['macro'],
                 linestyle='--', linewidth=4,
                 label=f'Macro‑avg (AUC = {roc_auc["macro"]:.2f})')
        colors = plt.cm.rainbow(np.linspace(0,1,n))
        for i, c in zip(range(n), colors):
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'{le.classes_[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0,1],[0,1],'k--', lw=2)
        plt.xlim(0,1); plt.ylim(0,1.05)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi‑class ROC Curves')
        plt.legend(loc='lower right', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(self.plots_path / 'roc_curve.png')
        plt.close()

    def convert_to_tflite(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        tfl = converter.convert()
        path = self.model_path / "tcn_gesture_model.tflite"
        with open(path,'wb') as f:
            f.write(tfl)
        size_kb = len(tfl)/1024
        self.logger.info(f"TFLite model saved ({size_kb:.1f} KB)")

if __name__ == "__main__":
    trainer = TrainTCNModel(sequence_length=10)
    trainer.train_and_evaluate()
