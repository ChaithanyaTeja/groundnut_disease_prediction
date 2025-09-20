from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models

def build_model(num_classes, img_size=(224,224)):
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=img_size+(3,))
    base_model.trainable = False  # freeze initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model
