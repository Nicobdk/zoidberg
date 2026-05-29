import json

with open("notebooks/04_interpretability.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "make_gradcam_heatmap" in source and "submodel" in source:
            new_source = """# 5. Génération de la preuve de résultat (Heatmap Grad-CAM)
# Géré automatiquement par "make_gradcam_heatmap" désormais !

# On force l'évaluation des gradients sur cette image particulière.
heatmap = make_gradcam_heatmap(img_array, model, last_conv)

# 6. On l'affiche joliment pour le dossier médical !
# Et on le sauvegarde pour ton README si tu le désires : 
save_destination = "reports/figures/grad_cam_bacteria_example.png"
display_gradcam(img_path, heatmap, alpha=0.5, display=True, save_path=save_destination)
"""
            cell["source"] = [s + "\n" for s in new_source.split("\n")][:-1]

with open("notebooks/04_interpretability.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
