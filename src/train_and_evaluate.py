# src/train_and_evaluate.py
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_loader import get_data_generators
from model import build_transfer_model

def train_and_evaluate(data_path):
    train_gen, val_gen = get_data_generators(data_path)
    model_names = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
    best_f1 = 0
    best_model_name = ""

    for name in model_names:
        print(f"\n--- Training {name} ---")
        model = build_transfer_model(model_type=name, num_classes=len(train_gen.class_indices))
        
        # Train
        model.fit(train_gen, validation_data=val_gen, epochs=10)
        
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