#!/usr/bin/env python
"""
Gnre le Flow Diagram du Pipeline Hirarchique Zoidberg
Visualise le flux des prdictions  travers les 2 stages
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Configuration
OUTPUT_FILE = "reports/figures/hierarchical_flow_diagram.png"

# Donnes du pipeline (bases sur final_report.md)
TOTAL_IMAGES = 1402

# Stage 1 : Binaire (Normal vs Pneumonia)
STAGE1_NORMAL = int(TOTAL_IMAGES * 0.52)  # 52% recall Normal
STAGE1_PNEUMONIA = TOTAL_IMAGES - STAGE1_NORMAL

# Stage 2 : Sous-type (Bacteria vs Virus) - sur les pneumonies dtectes
STAGE2_BACTERIA = int(STAGE1_PNEUMONIA * 0.95)  # 95% recall Bacteria
STAGE2_VIRUS = STAGE1_PNEUMONIA - STAGE2_BACTERIA

# Mtriques
AUC_STAGE1 = 0.9897
RECALL_STAGE1 = 0.98
ACC_STAGE2 = 0.76
RECALL_BACTERIA = 0.98

# Vraies distributions (ground truth)
TRUE_NORMAL = 200
TRUE_BACTERIA = 688
TRUE_VIRUS = 514

print("=" * 70)
print(" GNRATION DU FLOW DIAGRAM - PIPELINE HIRARCHIQUE")
print("=" * 70)
print(f"\nDataset Test : {TOTAL_IMAGES} images")
print(f"   Normal    : {TRUE_NORMAL} images ({TRUE_NORMAL/TOTAL_IMAGES*100:.1f}%)")
print(f"   Bactrie  : {TRUE_BACTERIA} images ({TRUE_BACTERIA/TOTAL_IMAGES*100:.1f}%)")
print(f"   Virus     : {TRUE_VIRUS} images ({TRUE_VIRUS/TOTAL_IMAGES*100:.1f}%)")

print(f"\nFlux Prdit :")
print(f"  Stage 1  Normal    : {STAGE1_NORMAL} images ({STAGE1_NORMAL/TOTAL_IMAGES*100:.1f}%)")
print(f"  Stage 1  Pneumonie : {STAGE1_PNEUMONIA} images ({STAGE1_PNEUMONIA/TOTAL_IMAGES*100:.1f}%)")
print(f"    Stage 2  Bactrie : {STAGE2_BACTERIA} images ({STAGE2_BACTERIA/TOTAL_IMAGES*100:.1f}%)")
print(f"    Stage 2  Virus    : {STAGE2_VIRUS} images ({STAGE2_VIRUS/TOTAL_IMAGES*100:.1f}%)")


def create_flow_diagram():
    """Cre le diagramme de flux du pipeline"""

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Couleurs
    color_input = '#E8F4F8'
    color_stage1 = '#B8E6F0'
    color_stage2 = '#7EC8E3'
    color_output = '#4A90A4'
    color_normal = '#81C784'
    color_bacteria = '#E57373'
    color_virus = '#9575CD'

    # === TITRE ===
    ax.text(5, 11.5, 'Pipeline Hirarchique  2 tages',
           ha='center', va='top', fontsize=18, fontweight='bold')
    ax.text(5, 11, f'Dataset Test : {TOTAL_IMAGES} Radiographies',
           ha='center', va='top', fontsize=12, style='italic')

    # === INPUT ===
    input_box = FancyBboxPatch((3.5, 9.5), 3, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=color_input, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 9.9, f'{TOTAL_IMAGES} Images', ha='center', va='center',
           fontsize=14, fontweight='bold')

    # Flche vers Stage 1
    arrow1 = FancyArrowPatch((5, 9.5), (5, 8.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='black')
    ax.add_patch(arrow1)

    # === STAGE 1 : BINAIRE ===
    stage1_box = FancyBboxPatch((2.5, 6.5), 5, 1.8,
                                boxstyle="round,pad=0.15",
                                facecolor=color_stage1, edgecolor='black', linewidth=2.5)
    ax.add_patch(stage1_box)

    ax.text(5, 7.9, ' STAGE 1 : Dtection Binaire ',
           ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(5, 7.5, 'EfficientNetB0 (Transfer Learning)',
           ha='center', va='center', fontsize=10, style='italic')
    ax.text(5, 7.1, f'AUC: {AUC_STAGE1:.4f} | Recall: {RECALL_STAGE1*100:.0f}%',
           ha='center', va='center', fontsize=10)
    ax.text(5, 6.7, 'Tche : Normal vs Pneumonie',
           ha='center', va='center', fontsize=9, style='italic', color='#555')

    # Flches de sortie Stage 1
    # Vers Normal (gauche)
    arrow_normal = FancyArrowPatch((3.5, 6.5), (2, 5),
                                  arrowstyle='->', mutation_scale=25,
                                  linewidth=2.5, color=color_normal)
    ax.add_patch(arrow_normal)
    ax.text(2.5, 5.7, f'{STAGE1_NORMAL} images\n({STAGE1_NORMAL/TOTAL_IMAGES*100:.1f}%)',
           ha='center', va='center', fontsize=9, fontweight='bold', color=color_normal)

    # Vers Pneumonie (droite)
    arrow_pneumonia = FancyArrowPatch((6.5, 6.5), (8, 5),
                                     arrowstyle='->', mutation_scale=25,
                                     linewidth=2.5, color='#E57373')
    ax.add_patch(arrow_pneumonia)
    ax.text(7.3, 5.7, f'{STAGE1_PNEUMONIA} images\n({STAGE1_PNEUMONIA/TOTAL_IMAGES*100:.1f}%)',
           ha='center', va='center', fontsize=9, fontweight='bold', color='#E57373')

    # === OUTPUT NORMAL ===
    normal_box = FancyBboxPatch((0.5, 4), 3, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=color_normal, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(normal_box)
    ax.text(2, 4.4, f' NORMAL', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    ax.text(2, 3.9, f'{STAGE1_NORMAL} images', ha='center', va='bottom',
           fontsize=9, color='white')

    # Flche verticale vers Stage 2
    arrow_to_stage2 = FancyArrowPatch((8, 5), (8, 3.5),
                                     arrowstyle='->', mutation_scale=30,
                                     linewidth=3, color='black')
    ax.add_patch(arrow_to_stage2)

    # === STAGE 2 : SOUS-TYPE ===
    stage2_box = FancyBboxPatch((5.5, 1.5), 5, 1.8,
                                boxstyle="round,pad=0.15",
                                facecolor=color_stage2, edgecolor='black', linewidth=2.5)
    ax.add_patch(stage2_box)

    ax.text(8, 2.9, ' STAGE 2 : Classification Sous-type ',
           ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(8, 2.5, 'EfficientNetB0 Fine-tuned (Pneumonies)',
           ha='center', va='center', fontsize=10, style='italic')
    ax.text(8, 2.1, f'Accuracy: {ACC_STAGE2*100:.0f}% | Recall Bact: {RECALL_BACTERIA*100:.0f}%',
           ha='center', va='center', fontsize=10)
    ax.text(8, 1.7, 'Tche : Bactrie vs Virus',
           ha='center', va='center', fontsize=9, style='italic', color='#555')

    # Flches de sortie Stage 2
    # Vers Bactrie (gauche)
    arrow_bacteria = FancyArrowPatch((6.5, 1.5), (5.5, 0.5),
                                    arrowstyle='->', mutation_scale=25,
                                    linewidth=2.5, color=color_bacteria)
    ax.add_patch(arrow_bacteria)
    ax.text(6, 1, f'{STAGE2_BACTERIA} images\n({STAGE2_BACTERIA/TOTAL_IMAGES*100:.1f}%)',
           ha='center', va='center', fontsize=9, fontweight='bold', color=color_bacteria)

    # Vers Virus (droite)
    arrow_virus = FancyArrowPatch((9.5, 1.5), (9.5, 0.5),
                                 arrowstyle='->', mutation_scale=25,
                                 linewidth=2.5, color=color_virus)
    ax.add_patch(arrow_virus)
    ax.text(9.5, 1, f'{STAGE2_VIRUS} images\n({STAGE2_VIRUS/TOTAL_IMAGES*100:.1f}%)',
           ha='center', va='center', fontsize=9, fontweight='bold', color=color_virus)

    # === OUTPUT BACTERIA ===
    bacteria_box = FancyBboxPatch((4, 0), 3, 0.6,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color_bacteria, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(bacteria_box)
    ax.text(5.5, 0.3, f' BACTRIE', ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')

    # === OUTPUT VIRUS ===
    virus_box = FancyBboxPatch((8.5, 0), 2, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=color_virus, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(virus_box)
    ax.text(9.5, 0.3, f' VIRUS', ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')

    # === LGENDE / STATISTIQUES ===
    stats_text = f"""
     Rsultat Final (3 Classes) :
     Normal    : {STAGE1_NORMAL:4d} images ({STAGE1_NORMAL/TOTAL_IMAGES*100:5.1f}%)
     Bactrie  : {STAGE2_BACTERIA:4d} images ({STAGE2_BACTERIA/TOTAL_IMAGES*100:5.1f}%)
     Virus     : {STAGE2_VIRUS:4d} images ({STAGE2_VIRUS/TOTAL_IMAGES*100:5.1f}%)

     Performance Globale : 71% Accuracy
    """

    ax.text(0.5, 11, stats_text, ha='left', va='top',
           fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8, pad=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n Flow diagram sauvegard : {OUTPUT_FILE}")

    return fig


def create_sankey_style_diagram():
    """Alternative : Diagramme style Sankey (flux proportionnels)"""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Titre
    ax.text(5, 9.5, 'Pipeline Hirarchique - Flux des Prdictions',
           ha='center', fontsize=16, fontweight='bold')

    # Calcul des hauteurs proportionnelles (chelle)
    scale = 5 / TOTAL_IMAGES  # 5 units pour reprsenter toutes les images

    h_total = TOTAL_IMAGES * scale
    h_normal = STAGE1_NORMAL * scale
    h_pneumonia = STAGE1_PNEUMONIA * scale
    h_bacteria = STAGE2_BACTERIA * scale
    h_virus = STAGE2_VIRUS * scale

    # Position Y de base
    y_base = 2

    # Barre INPUT
    input_rect = plt.Rectangle((0.5, y_base), 1, h_total,
                               facecolor='#B8E6F0', edgecolor='black', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(1, y_base + h_total/2, f'{TOTAL_IMAGES}\nimages',
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Stage 1 - Sparation
    # Flux vers Normal
    normal_y = y_base + h_total - h_normal
    from matplotlib.patches import Polygon
    flow_normal = Polygon([[1.5, y_base + h_total], [1.5, normal_y + h_normal],
                          [4, normal_y + h_normal], [4, y_base + h_total]],
                         facecolor='#81C784', edgecolor='black', alpha=0.6, linewidth=1)
    ax.add_patch(flow_normal)

    # Flux vers Pneumonie
    flow_pneumonia = Polygon([[1.5, y_base], [1.5, normal_y],
                             [4, normal_y], [4, y_base]],
                            facecolor='#E57373', edgecolor='black', alpha=0.6, linewidth=1)
    ax.add_patch(flow_pneumonia)

    # Barre STAGE 1 OUTPUT
    # Normal
    normal_rect = plt.Rectangle((4, normal_y), 1, h_normal,
                                facecolor='#81C784', edgecolor='black', linewidth=2)
    ax.add_patch(normal_rect)
    ax.text(4.5, normal_y + h_normal/2, f'Normal\n{STAGE1_NORMAL}',
           ha='center', va='center', fontsize=9, fontweight='bold')

    # Pneumonie
    pneumonia_rect = plt.Rectangle((4, y_base), 1, h_pneumonia,
                                   facecolor='#E57373', edgecolor='black', linewidth=2)
    ax.add_patch(pneumonia_rect)
    ax.text(4.5, y_base + h_pneumonia/2, f'Pneumonie\n{STAGE1_PNEUMONIA}',
           ha='center', va='center', fontsize=9, fontweight='bold')

    # Stage 2 - Sous-division Pneumonie
    bacteria_y = y_base + h_pneumonia - h_bacteria

    # Flux vers Bacteria
    flow_bacteria = Polygon([[5, y_base + h_pneumonia], [5, bacteria_y + h_bacteria],
                            [8, bacteria_y + h_bacteria], [8, y_base + h_pneumonia]],
                           facecolor='#E57373', edgecolor='black', alpha=0.6, linewidth=1)
    ax.add_patch(flow_bacteria)

    # Flux vers Virus
    flow_virus = Polygon([[5, y_base], [5, bacteria_y],
                         [8, bacteria_y], [8, y_base]],
                        facecolor='#9575CD', edgecolor='black', alpha=0.6, linewidth=1)
    ax.add_patch(flow_virus)

    # Barres OUTPUT FINAL
    # Bacteria
    bacteria_rect = plt.Rectangle((8, bacteria_y), 1, h_bacteria,
                                  facecolor='#E57373', edgecolor='black', linewidth=2)
    ax.add_patch(bacteria_rect)
    ax.text(8.5, bacteria_y + h_bacteria/2, f'Bactrie\n{STAGE2_BACTERIA}',
           ha='center', va='center', fontsize=9, fontweight='bold')

    # Virus
    virus_rect = plt.Rectangle((8, y_base), 1, h_virus,
                               facecolor='#9575CD', edgecolor='black', linewidth=2)
    ax.add_patch(virus_rect)
    ax.text(8.5, y_base + h_virus/2, f'Virus\n{STAGE2_VIRUS}',
           ha='center', va='center', fontsize=9, fontweight='bold')

    # Labels tapes
    ax.text(1, y_base - 0.5, 'INPUT', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.5, y_base - 0.5, 'STAGE 1', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.5, y_base - 0.5, 'OUTPUT', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    sankey_file = "reports/figures/hierarchical_sankey_diagram.png"
    plt.savefig(sankey_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" Sankey diagram sauvegard : {sankey_file}")

    return fig


if __name__ == "__main__":
    # Gnrer les deux versions
    print("\n[1/2] Gnration du Flow Diagram...")
    create_flow_diagram()

    print("\n[2/2] Gnration du Sankey Diagram...")
    create_sankey_style_diagram()

    print("\n" + "=" * 70)
    print(" SUCCS : 2 diagrammes gnrs !")
    print("=" * 70)
    print(f"\nFichiers crs :")
    print(f"  1. reports/figures/hierarchical_flow_diagram.png")
    print(f"  2. reports/figures/hierarchical_sankey_diagram.png")
    print("\n Les visualisations sont prtes pour le rapport !")
