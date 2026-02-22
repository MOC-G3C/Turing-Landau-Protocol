import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

fichier_json = "meta_history.json"
if not os.path.exists(fichier_json):
    print(f"❌ Erreur : Lance d'abord la simulation pour générer {fichier_json}")
    exit()

with open(fichier_json, "r") as f:
    histoire = json.load(f)

generations = len(histoire)
jours = range(generations)

# Extraction des métriques
kappa_vars = [np.var(g["kappas"]) for g in histoire]
cool_vars = [np.var(g["coolings"]) for g in histoire]
scores_moyens = [np.mean(g["scores"]) for g in histoire]

# --- TEST 1 : LA CONVERGENCE ---
slope_k, _, r_k, p_k, _ = linregress(jours, kappa_vars)
slope_c, _, r_c, p_c, _ = linregress(jours, cool_vars)

print("\n🔬 --- RÉSULTATS DES TESTS STATISTIQUES ---")
print("1. Test de Convergence (La sélection élimine-t-elle le hasard ?)")
if p_k < 0.05 and slope_k < 0:
    print(f"✅ Gène Kappa : Convergence prouvée ! (p-value: {p_k:.4f}, Tendance: {slope_k:.4e})")
else:
    print(f"⚠️ Gène Kappa : Pas de convergence stricte (p-value: {p_k:.4f})")

if p_c < 0.05 and slope_c < 0:
    print(f"✅ Gène Cooling : Convergence prouvée ! (p-value: {p_c:.4f}, Tendance: {slope_c:.4e})")
else:
    print(f"⚠️ Gène Cooling : Pas de convergence stricte (p-value: {p_c:.4f})")

# --- TEST 2 : L'HÉRITABILITÉ (CORRIGÉ) ---
parents_k = []
enfants_k = []
r_carre = 0.0 # On initialise la variable pour éviter l'erreur

for gen in histoire:
    for lignee in gen.get("lignees", []):
        enfant_k = lignee.get("enfant_kappa")
        parent_id = lignee.get("parent")

        if parent_id and parent_id != "Genesis":
            try:
                # On lit l'ID "Gx_Ay" pour retrouver le gène exact du parent dans l'histoire
                p_gen = int(parent_id.split("_")[0][1:])
                p_agent = int(parent_id.split("_")[1][1:])
                parent_k = histoire[p_gen]["kappas"][p_agent]
                
                parents_k.append(parent_k)
                enfants_k.append(enfant_k)
            except Exception:
                pass

if len(parents_k) > 0:
    slope_h, _, r_h, p_h, _ = linregress(parents_k, enfants_k)
    r_carre = r_h**2
    print("\n2. Test d'Héritabilité (Transmission des stratégies)")
    print(f"R² (Force de la transmission) : {r_carre:.4f}")
    if r_carre > 0.2:
        print("✅ Transmission forte : L'évolution est dirigée par la génétique, pas par la chance.")
    else:
        print("⚠️ Transmission faible : La mutation brouille l'héritage.")
else:
    print("\n⚠️ Impossible de calculer l'héritabilité (pas de lignées valides trouvées).")

# --- AFFICHAGE VISUEL DES PREUVES ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.canvas.manager.set_window_title("Preuves Statistiques - L'AXE HYBRIDE")

# Graphique 1
axs[0].plot(jours, kappa_vars, '-o', color='purple', label="Variance Kappa")
axs[0].plot(jours, cool_vars, '-o', color='red', label="Variance Cooling")
axs[0].set_title("Preuve de Convergence (Variance)")
axs[0].set_xlabel("Générations")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Graphique 2
axs[1].plot(jours, scores_moyens, '-o', color='green')
axs[1].set_title("Évolution de l'Efficience Moyenne")
axs[1].set_xlabel("Générations")
axs[1].grid(True, alpha=0.3)

# Graphique 3
axs[2].scatter(parents_k, enfants_k, alpha=0.5, color='blue', s=10)
axs[2].set_title(f"Héritabilité (R² = {r_carre:.2f})")
axs[2].set_xlabel("Gène du Parent")
axs[2].set_ylabel("Gène de l'Enfant")
axs[2].grid(True, alpha=0.3)


plt.tight_layout()
plt.show()