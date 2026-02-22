import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# --- LECTURE DES DONNÉES ---
csv_file = "transition_phase_data.csv"

if not os.path.exists(csv_file):
    print(f"❌ Erreur : Fichier {csv_file} introuvable.")
    exit()

df = pd.read_csv(csv_file)

# --- EXTRACTION DES VARIABLES ---
# x = Longévité (combien de temps ils vivent)
# y = Efficience (ratio Information/Chaleur)
x = df["Longevity"]
y = df["Efficiency"]

# --- CALCUL MATHÉMATIQUE (Régression) ---
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r_squared = r_value**2

# --- RÉSULTATS DANS LE TERMINAL ---
print("📊 --- RÉSULTATS DE L'ANALYSE L'AXE HYBRIDE ---")
print(f"📈 Coefficient de détermination (R²) : {r_squared:.4f}")
if r_squared > 0.7:
    print("✅ Corrélation FORTE : L'efficience dicte la survie. Loi de Landauer validée.")
else:
    print("⚠️ Corrélation FAIBLE : D'autres facteurs influencent la survie.")

# --- VISUALISATION ---
plt.figure(figsize=(10, 6))

# Affichage des points par profil pour garder le contexte
colors = {"Stoic": "blue", "Balanced": "purple", "Plastic": "green"}
for p_name in df["Profile"].unique():
    subset = df[df["Profile"] == p_name]
    plt.scatter(subset["Longevity"], subset["Efficiency"], 
                color=colors.get(p_name, "gray"), label=p_name, alpha=0.5)

# Tracé de la Ligne de Tendance Mathématique
plt.plot(x, intercept + slope * x, color='red', linewidth=2, 
         label=f'Loi d\'Évolution (R² = {r_squared:.2f})')

plt.xlabel("Longévité (Cycles de vie)")
plt.ylabel("Efficience (Information / Chaleur)")
plt.title("L'AXE HYBRIDE : Démonstration Mathématique de la Survie")
plt.legend()
plt.grid(True, alpha=0.2)

# Sauvegarde
plt.savefig("preuve_mathematique_R2.png", dpi=300)
print("📸 Graphique de preuve sauvegardé : preuve_mathematique_R2.png")

plt.show()