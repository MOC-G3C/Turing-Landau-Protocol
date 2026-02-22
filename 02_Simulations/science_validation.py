import json, numpy as np, matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# 1. CHARGEMENT
if not os.path.exists("meta_history.json"):
    print("❌ meta_history.json manquant.")
    exit()

with open("meta_history.json", "r") as f: histoire = json.load(f)
data_pop = np.load("final_population.npz", allow_pickle=True)
pop_finale = data_pop["population"]

# 2. BOOTSTRAP (CONVERGENCE KAPPA)
kappa_vars = np.array([np.var(g["kappas"]) for g in histoire])
G, B = len(kappa_vars), 5000
slopes = [linregress(np.arange(G), kappa_vars[np.random.choice(range(G), size=G, replace=True)]).slope for _ in range(B)]
p_emp = np.mean(np.array(slopes) >= 0)

# 3. PERMUTATION (HÉRITABILITÉ)
parents, children = [], []
for gen in histoire:
    for pair in gen.get("parent_child_pairs", []):
        parents.append(pair["parent_kappa"])
        children.append(pair["child_kappa"])

parents, children = np.array(parents), np.array(children)

# --- RAPPORT ---
print(f"\n🔬 --- RAPPORT DE VALIDATION SCIENTIFIQUE ---")
print(f"1. Convergence (Bootstrap): Slope {np.mean(slopes):.2e} | p_emp = {p_emp:.4f} " + ("✅" if p_emp < 0.05 else "❌"))

if len(parents) > 0:
    obs_r2 = linregress(parents, children).rvalue**2
    r2s_perm = [linregress(parents, np.random.permutation(children)).rvalue**2 for _ in range(B)]
    p_herit = np.mean(np.array(r2s_perm) >= obs_r2)
    print(f"2. Héritabilité (Permutation): R² {obs_r2:.4f} | p_val = {p_herit:.4f} " + ("✅" if p_herit < 0.05 else "❌"))
else:
    print(f"2. Héritabilité: ⚠️ Pas de données généalogiques trouvées.")

# 4. STRESS TEST
def run_stress(ind, burst_amp=10.0):
    theta, meta = ind["theta"].copy(), ind["meta"]
    heat = 0.0
    for t in range(1, 1000):
        env = np.sin(t/50.) * (1.0 + (burst_amp if np.random.rand() < 0.05 else 1.0) * np.random.randn())
        if t % 9 == 0: theta *= meta["cooling_strength"] # Résonance Tesla
        var = np.var(theta)
        heat = max(0.0, heat + (var*15.0) - 0.3)
        if heat > 500 or var < 0.02: return False
    return True

surv_rate = np.mean([run_stress(ind) for ind in pop_finale])
print(f"3. Robustesse (Stress Test): Taux de survie = {surv_rate*100:.1f}% " + ("✅" if surv_rate > 0.7 else "⚠️"))