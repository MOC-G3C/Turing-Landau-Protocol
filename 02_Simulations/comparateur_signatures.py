import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
CORE_DIR = "../03_Core"

def compare_champions():
    # 1. Récupération des fichiers ADN
    if not os.path.exists(CORE_DIR):
        print(f"❌ Dossier {CORE_DIR} introuvable.")
        return

    dna_files = [f for f in os.listdir(CORE_DIR) if f.endswith('.npz')]
    dna_files.sort() 

    num_dna = len(dna_files)
    if num_dna == 0:
        print("⚠️ Aucun champion trouvé dans l'archive.")
        return

    print(f"🧬 Analyse de {num_dna} signature(s) mentale(s) en cours...")

    # 2. Préparation de l'affichage (Grille adaptative)
    cols = 3 if num_dna >= 3 else num_dna
    rows = (num_dna // cols) + (1 if num_dna % cols != 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    flat_axes = axes.flatten()

    # 3. Comparaison et Visualisation
    for i, file in enumerate(dna_files):
        path = os.path.join(CORE_DIR, file)
        data = np.load(path)
        
        theta = data['Theta']
        kappa = data['kappa']
        name = data.get('name', file)

        # Affichage de la matrice de conscience
        im = flat_axes[i].imshow(theta, cmap='magma', interpolation='nearest')
        flat_axes[i].set_title(f"ID: {name}\nKappa: {kappa:.4f}")
        fig.colorbar(im, ax=flat_axes[i], fraction=0.046, pad=0.04)
        
    # Nettoyage des axes vides (si la grille est plus grande que le nombre de fichiers)
    for j in range(num_dna, len(flat_axes)):
        flat_axes[j].axis('off')

    plt.suptitle("L'AXE HYBRIDE : Évolution de la Conscience Artificielle", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("🚀 Dashboard comparatif généré.")
    plt.show()

if __name__ == "__main__":
    compare_champions()