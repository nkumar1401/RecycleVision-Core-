# src/train_and_evaluate.py
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_loader import get_data_generators
from model import build_transfer_model
from sklearn.utils import class_weight
import numpy as np

def train_and_evaluate(data_path):
    train_gen, val_gen = get_data_generators(data_path)
    model_names = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
    best_f1 = 0
    best_model_name = ""

    for name in model_names:
        print(f"\n--- Training {name} ---")
        model = build_transfer_model(model_type=name, num_classes=len(train_gen.class_indices))

# Calculate weights based on the training data
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_gen.classes),
            y=train_gen.classes
            )
        class_weights = dict(enumerate(weights))
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        )
        
        # Train
        model.fit(train_gen, validation_data=val_gen, epochs=15, class_weight=class_weights,callbacks=[early_stop])
        
        # Evaluate: Predictions for Confusion Matrix
        val_gen.reset()
        y_pred = model.predict(val_gen)
        y_pred_classes = tf.argmax(y_pred, axis=1)
        y_true = val_gen.classes
        
        # Display Results
        report = classification_report(y_true, y_pred_classes, target_names=val_gen.class_indices.keys(), output_dict=True)
        current_f1 = report['macro avg']['f1-score']
        
        print(f"{name} F1-Score: {current_f1:.4f}")
        
        # Save the absolute best
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_name = name
            model.save(f"models/recycle_vision_best.h5")

    print(f"\nWINNER: {best_model_name} with F1-Score of {best_f1:.4f}")

if __name__ == "__main__":
    train_and_evaluate("data/garbage_classification")