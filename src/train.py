import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
BASE_DIR = r"D:\groundnut_leaf_disease_detection\Dataset"
MODEL_DIR = r"D:\groundnut_leaf_disease_detection\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "val"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# ✅ Save class indices
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_gen.class_indices, f)

print("Class indices saved:", train_gen.class_indices)

# Build model (MobileNetV2 transfer learning)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[checkpoint, earlystop]
)

print("✅ Training complete. Model + class indices saved in:", MODEL_DIR)
