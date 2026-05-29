import json
import re

with open("notebooks/04_interpretability.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "tf.keras.models.load_model" in source:
            source = re.sub(r'model_path = ".*?"', 'model_path = "models/trained/zoidberg_cnn_best_crop_v1.h5"', source)
            cell["source"] = [s + "\n" for s in source.split("\n")][:-1]
            break

with open("notebooks/04_interpretability.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
