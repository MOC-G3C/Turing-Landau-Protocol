# localized_ablation.py
import numpy as np, copy
from simulation_V8_13_spatial_mix import eval_individu_raw

data = np.load("final_population_v8_13.npz", allow_pickle=True)
champ = data["population"][0]

# Création d'un masque sur les Hotspots
mask = np.zeros((16, 16), dtype=bool)
mask[2:5, 2:5] = True   # Zone Hotspot 1
mask[11:14, 11:14] = True # Zone Hotspot 2

print("📊 Évaluation Baseline...")
base = eval_individu_raw(champ, steps=400)

print("🔪 Ablation FAST globale")
ab = copy.deepcopy(champ)
ab["tau_fast"][:,:] = 0.95 # On rend les zones de mémoire amnésiques
ab_res = eval_individu_raw(ab, steps=400)

print(f"\nLucidité Initiale : {base['luc_avg']:.2f}")
print(f"Lucidité après ablation : {ab_res['luc_avg']:.2f}")
print(f"Delta (Perte de fonction) : {base['luc_avg'] - ab_res['luc_avg']:.2f}")