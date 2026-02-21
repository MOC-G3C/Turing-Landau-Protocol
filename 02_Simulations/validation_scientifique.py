import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy

# --- CONFIGURATION DU PROTOCOLE ---
num_runs = 50
steps = 5000
L = 16
CSV_FILE = "transition_phase_data.csv"

# Profils de l'Axe Hybride
profiles = {
    "Stoic": {"adapt": 0.0001, "color": "blue"},
    "Balanced": {"adapt": 0.002, "color": "purple"},
    "Plastic": {"adapt": 0.02, "color": "green"}
}

def calculate_shannon_entropy(matrix):
    # Mesure de l'information structurée (Complexité)
    counts, _ = np.histogram(matrix, bins=10)
    return entropy(counts)

# --- ACQUISITION DES DONNÉES ---
data_log = []

print(f"🔬 Analyse de robustesse : {num_runs} cycles par profil...")

for run in range(1, num_runs + 1):
    for p_name, p_config in profiles.items():
        theta = 0.001 * np.random.randn(L, L)
        heat = 0.0
        kappa = 1.0
        total_info = 0
        
        for step in range(steps):
            var = np.var(theta)
            info = calculate_shannon_entropy(theta)
            
            # Physique de Landauer simplifiée
            heat += (var * 10) - 0.5
            
            # Seuil de rupture thermique
            if heat > 1000:
                break
            
            total_info += info
            # Adaptation de l'identité (Kappa)
            kappa = np.clip(kappa + p_config["adapt"] * (0.2 - var), 0.5, 2.5)
            theta += np.random.randn(L, L) * 0.01 
            
        # Calcul de l'Efficience (Information / Énergie)
        efficiency = total_info / (heat + 1e-6)
        
        data_log.append({
            "Run": run,
            "Profile": p_name,
            "Longevity": step,
            "Efficiency": efficiency,
            "Final_Heat": heat
        })

# --- EXPORT CSV ---
df = pd.DataFrame(data_log)
df.to_csv(CSV_FILE, index=False)
print(f"✅ Résultats exportés : {CSV_FILE}")

# --- DASHBOARD STATISTIQUE ---
plt.figure(figsize=(12, 6))
for p_name, p_config in profiles.items():
    subset = df[df["Profile"] == p_name]
    plt.scatter(subset["Longevity"], subset["Efficiency"], 
                color=p_config["color"], label=p_name, alpha=0.5)

plt.xlabel("Longévité (Cycles de vie)")
plt.ylabel("Efficience (Information / Chaleur)")
plt.title("L'AXE HYBRIDE : Transition de Phase et Point Critique")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()