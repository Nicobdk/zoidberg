import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


class HierarchicalClassifier:
    def __init__(self, binary_model_path, subtype_model_path, img_size=(224, 224)):
        self.binary_model = tf.keras.models.load_model(binary_model_path)
        self.subtype_model = tf.keras.models.load_model(subtype_model_path)
        self.img_size = img_size

        self.binary_classes = ["NORMAL", "PNEUMONIA"]
        self.subtype_classes = ["BACTERIA", "VIRUS"]

    def preprocess(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, img_path):
        img = self.preprocess(img_path)

        # ----- Étape 1 : Binaire -----
        binary_prob = self.binary_model.predict(img)[0][0]
        binary_pred = 1 if binary_prob > 0.5 else 0

        if self.binary_classes[binary_pred] == "NORMAL":
            return {
                "final_prediction": "NORMAL",
                "confidence": float(binary_prob if binary_pred == 1 else 1 - binary_prob),
                "stage": "binary_only"
            }

        # ----- Étape 2 : Subtype (Binaire) -----
        # Le subtype_model a été entraîné spécifiquement sur BACTERIA vs VIRUS avec un Sigmoid.
        # Si Proba > 0.5 => VIRUS. Sinon => BACTERIA.
        subtype_prob = self.subtype_model.predict(img)[0][0]
        
        # 0 = BACTERIA, 1 = VIRUS (Selon l'ordre alphabétique des dossiers dans data_loader)
        subtype_pred = 1 if subtype_prob > 0.5 else 0
        subtype_classes = ["BACTERIA", "VIRUS"]

        return {
            "final_prediction": subtype_classes[subtype_pred],
            "confidence": float(subtype_prob if subtype_pred == 1 else 1 - subtype_prob),
            "stage": "hierarchical"
        }