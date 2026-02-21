import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy

# --- CONFIGURATION SCIENTIFIQUE ---
num_runs = 50
steps = 5000
L = 16
profiles = ["Stoic", "Balanced", "Plastic"]

def calculate_shannon_entropy(matrix):
    # On discrétise la matrice pour calculer l'entropie
    counts, _ = np.histogram(matrix, bins=10)
    return entropy(counts)

# --- MOTEUR DE VALIDATION ---
results = {p: {"efficiency": [], "survival": []} for p in profiles}

print(f"🔬 Analyse de robustesse sur {num_runs} itérations...")

for run in range(num_runs):
    for p_name in profiles:
        # Initialisation selon le profil
        adapt = {"Stoic": 0.0001, "Balanced": 0.002, "Plastic": 0.02}[p_name]
        theta = 0.001 * np.random.randn(L, L)
        heat = 0.0
        kappa = 1.0
        total_info = 0
        alive = True
        
        for step in range(steps):
            # Dynamique simplifiée
            var = np.var(theta)
            info = calculate_shannon_entropy(theta)
            
            # Loi de Landauer : Chaleur vs Info
            heat += (var * 10) - 0.5
            if heat > 1000: # Seuil de mort arbitraire par run
                alive = False
                break
            
            total_info += info
            # Adaptation
            kappa = np.clip(kappa + adapt * (0.2 - var), 0.5, 2.5)
            theta += np.random.randn(L, L) * 0.01 # Bruit thermique
            
        results[p_name]["efficiency"].append(total_info / (heat + 1e-6))
        results[p_name]["survival"].append(step)

# --- VISUALISATION DES RÉSULTATS STATISTIQUES ---
plt.figure(figsize=(10, 6))
for p_name in profiles:
    plt.scatter(results[p_name]["survival"], results[p_name]["efficiency"], label=p_name, alpha=0.5)

plt.xlabel("Longévité (Steps)")
plt.ylabel("Efficience Informationnelle (Info/Heat)")
plt.title("L'AXE HYBRIDE : Preuve Statistique de l'Équilibre")
plt.legend()
plt.show()