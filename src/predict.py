import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = os.path.join("..", "models", "best_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = ['early_leaf_spot', 'healthy_leaf', 'late_leaf_spot', 'nutrition_deficiency', 'rust']

def predict_image(img_path):
    image = Image.open(img_path).resize((224,224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return pred_class, confidence

if __name__ == "__main__":
    test_img = "../Dataset/test/healthy_leaf/some_image.jpg"
    label, conf = predict_image(test_img)
    print(f"Prediction: {label} ({conf:.2f}%)")
