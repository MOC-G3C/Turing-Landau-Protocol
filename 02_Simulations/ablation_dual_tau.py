# ablation_dual_tau.py
import numpy as np
import copy
from simulation_V8_12_dual_tau import eval_individu_raw

# 1. Charger le champion
data = np.load("final_population_v8_12.npz", allow_pickle=True)
champ_original = data["population"][0]

# 2. Évaluation Baseline
print("📊 Évaluation du champion intact...")
res_base = eval_individu_raw(champ_original, steps=400)

# 3. Ablation du canal SLOW (on force tau_slow à devenir rapide)
print("🔪 Lobotomie du canal SLOW (conversion en mémoire courte)...")
champ_ablate = copy.deepcopy(champ_original)
champ_ablate["tau_slow"] = np.full((16, 16), 0.20) # On détruit la mémoire longue

res_ablate = eval_individu_raw(champ_ablate, steps=400)

# 4. Résultats
delta_luc = res_base["luc_avg"] - res_ablate["luc_avg"]
delta_q = res_base["q_avg"] - res_ablate["q_avg"]

print("\n--- RÉSULTATS DE L'ABLATION ---")
print(f"Lucidité Baseline : {res_base['luc_avg']:.2f}")
print(f"Lucidité après ablation : {res_ablate['luc_avg']:.2f}")
print(f"Perte de Lucidité : {delta_luc:.2f}")
print(f"Variation thermique (Q) : {delta_q:.2e}")

if delta_luc > 2:
    print("\n✅ PREUVE CAUSALE ÉTABLIE : Le canal SLOW est indispensable à la cognition.")
else:
    print("\n⚠️ RÉSULTAT AMBIGU : Le système semble trop dépendre du canal FAST.")