import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os

# Paths
MODEL_PATH = r"D:\groundnut_leaf_disease_detection\models\best_model.h5"
CLASS_INDICES_PATH = r"D:\groundnut_leaf_disease_detection\models\class_indices.json"
TEST_DIR = r"D:\groundnut_leaf_disease_detection\Dataset\test"

# Load model
model = load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

class_labels = list(class_indices.keys())

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict
pred_probs = model.predict(test_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes

# 🔥 Classification report (Precision, Recall, F1-score)
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix (Optional)
print("\n🔎 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
