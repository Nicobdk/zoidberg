import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def get_img_array(img_path, size, preprocess_func=None):
    # `img` est une image PIL
    img = load_img(img_path, target_size=size)
    # `array` est un array float32 de forme (size[0], size[1], 3)
    array = img_to_array(img)
    # On ajoute une dimension pour avoir (1, size[0], size[1], 3)
    array = np.expand_dims(array, axis=0)
    
    if preprocess_func:
        array = preprocess_func(array)
        
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if isinstance(last_conv_layer_name, tuple):
        inner_model_name, conv_name = last_conv_layer_name
        inner_model = model.get_layer(inner_model_name)
        grad_model = tf.keras.models.Model(
            inner_model.inputs, 
            [inner_model.get_layer(conv_name).output, inner_model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, inner_preds = grad_model(img_array)
            tape.watch(last_conv_layer_output)
            x = inner_preds
            inner_idx = next(i for i, l in enumerate(model.layers) if l.name == inner_model_name)
            for layer in model.layers[inner_idx+1:]:
                x = layer(x)
            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0]
            else:
                class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
    else:
        # On crée un modèle qui prend l'image et recrache l'activation de la dernière couche de conv
        # ET la prédiction finale
        grad_model = tf.keras.models.Model(
            model.inputs, 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # On calcule le gradient de la classe de prédiction par rapport à la sortie de la dernière couche conv
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0]
            else:
                class_channel = preds[:, pred_index]

        # C'est le gradient de la sortie du neurone prédictif
        grads = tape.gradient(class_channel, last_conv_layer_output)

    # On fait une moyenne des gradients par carte de caractéristiques
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # On pondère les activations de la couche conv par "l'importance" de cette activation (son gradient)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisation pour l'affichage entre 0 et 1 (On retire les valeurs négatives avec ReLU)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4, display=True, save_path=None):
    # Charger l'image d'origine
    img = img_to_array(load_img(img_path))

    # Redimensionner la heatmap aux dimensions de l'image
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")

    # Utiliser les couleurs RGB de la colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Mettre la heatmap à la taille de l'image originale
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superposer la heatmap
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    if display:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(superimposed_img)
        ax.axis("off")
        ax.set_title("Grad-CAM Interpretability")
        
        # Ajout de la légende (colorbar)
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Degré d\'attention du modèle', rotation=270, labelpad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"✅ Grad-CAM sauvegardé sous : {save_path}")
        plt.show()

    return superimposed_img

def find_last_conv_layer(model):
    """Trouve dynamiquement la dernière couche de convolution"""
    for layer in reversed(model.layers):
        # Cas pour l'EfficientNet imbriqué dans un Sequential (ton transfert_model.py)
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D) or "conv" in sub_layer.name.lower() and not "bn" in sub_layer.name.lower():
                    # Dans ce cas il faudra appliquer Grad-Cam au sous-modèle!
                    return layer.name, sub_layer.name
                    
        # Cas pour les modèles Sequential normaux (ton cnn_model.py)
        elif isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower() and not "bn" in layer.name.lower():
            return layer.name
                    
    raise ValueError("Impossible de trouver une couche de convolution dans le modèle.")

