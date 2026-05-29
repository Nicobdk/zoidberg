import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob

# On prend la même image que pour le Grad-CAM
bacteria_images = glob.glob("data/raw/chest_Xray/test/2_BACTERIA/*.*")
img_path = bacteria_images[0]

# Chargement
img = img_to_array(load_img(img_path)) / 255.0

# Application de crops (C'est mathématiquement ce qui se passe quand on "zoom" au centre)
# Fraction 0.8 correspond à un zoom in de ~20% (on garde 80% du centre)
# Fraction 0.5 correspond à un zoom in de 50%
img_zoom_20 = tf.image.central_crop(img, central_fraction=0.8).numpy()
img_zoom_30 = tf.image.central_crop(img, central_fraction=0.7).numpy()
img_zoom_40 = tf.image.central_crop(img, central_fraction=0.6).numpy() # Zoom 40% (on garde 60%)
img_zoom_50 = tf.image.central_crop(img, central_fraction=0.5).numpy()

# Affichage côte à côte
plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
plt.imshow(img)
plt.title("Originale")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(img_zoom_20)
plt.title("Zoom ~20% (Ce qu'on a fait)")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(img_zoom_30)
plt.title("Zoom ~30% (Ce qu'on a fait)")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(img_zoom_40)
plt.title("Zoom ~40%")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(img_zoom_50)
plt.title("Zoom ~50%")
plt.axis("off")

plt.tight_layout()
save_path = "reports/figures/zoom_preview.png"
plt.savefig(save_path)
print(f"✅ Prévisualisation sauvegardée sous : {save_path}")
