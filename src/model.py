from tensorflow.keras.applications import MobileNetV2, ResNet50 , EfficientNetB0
from tensorflow.keras import layers, models

def build_transfer_model(model_type='MobileNetV2', num_classes=6):
    """Builds a modular model based on specified architecture."""
    if model_type == 'MobileNetV2':
        base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    elif model_type == 'ResNet50':
        base = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    elif model_type == 'EfficientNetB0':
        base = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    else:
        raise ValueError("Unsupported model type")
    
    base.trainable = False  # Freeze pre-trained weights

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),  # Prevents overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model