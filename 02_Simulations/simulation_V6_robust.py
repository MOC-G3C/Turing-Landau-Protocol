import numpy as np
import copy
import json
import os
from scipy.stats import entropy

# --- 1. UTILS ---
def calculer_entropie(matrice):
    comptes, _ = np.histogram(matrice, bins=32)
    return entropy(comptes + 1e-9)

def creer_individu(adn_base, meta_parent=None, id_parent=None):
    theta = adn_base + np.random.randn(*adn_base.shape) * 0.05
    theta = (theta - theta.mean()) / (theta.std() + 1e-8) * 0.25
    if meta_parent is None:
        meta = {"kappa_gain": 0.15, "cooling_strength": 0.60, "mutation_rate": 0.02, "target_base": 0.04}
        p_id = "Genesis"
    else:
        meta = copy.deepcopy(meta_parent)
        for key in ["kappa_gain", "cooling_strength"]: 
            meta[key] *= (1.0 + np.random.randn() * 0.02)
        p_id = id_parent
    return {"theta": theta, "meta": meta, "parent_id": p_id}

# --- 2. ÉVALUATION MULTI-ENVIRONNEMENTS (K=5) ---
def eval_multi_env(ind, k_env=5, steps=2000):
    theta0, meta = ind["theta"].copy(), ind["meta"]
    scores, survs = [], []
    
    for k in range(k_env):
        np.random.seed(k + 42) # Seeds fixes pour comparer équitablement
        theta = theta0.copy()
        mem = np.zeros_like(theta)
        heat, total_heat, entropies = 0.0, 0.0, []
        
        for t in range(1, steps + 1):
            # Bursts intégrés pendant l'entraînement pour la robustesse
            chaos = 8.0 if np.random.rand() < 0.02 else 1.0
            env = np.sin(t/50.0) * chaos
            mem = 0.95 * mem + 0.05 * env
            target = meta["target_base"] + 0.02 * abs(env - mem.mean())
            if t % 9 == 0: theta *= meta["cooling_strength"] # Tesla 3-6-9
            
            var = np.var(theta)
            heat = max(0.0, heat + (var * 15.0) - 0.3)
            total_heat += heat
            ent = calculer_entropie(theta)
            entropies.append(ent)
            
            if heat > 500 or var < 0.02 or ent < 0.8: break
            
            kappa = np.clip(1.0 + meta["kappa_gain"] * (target - var), 0.1, 4.0)
            theta += (np.random.randn(16, 16) * (0.02 * kappa * meta["mutation_rate"])) + (env * 0.005)
        
        scores.append(np.mean(entropies) / (total_heat + 1e-9) if entropies else 0)
        survs.append(1 if t >= steps else 0)
        
    avg_score = np.mean(scores)
    surv_frac = np.mean(survs)
    # FITNESS : 70% Score + 30% Robustesse (Survie)
    fitness = 0.7 * avg_score + 0.3 * surv_frac
    return {"fitness": fitness, "avg_score": avg_score, "surv_frac": surv_frac, "theta": theta}

# --- 3. BOUCLE ÉVOLUTIVE ---
alpha_path = "../03_Core/alpha_post_arena.npz"
adn_base = np.load(alpha_path)[np.load(alpha_path).files[0]]
nb_gen, pop_size = 30, 20
population = [creer_individu(adn_base) for _ in range(pop_size)]
meta_history = []

print("🌪️ Forge de Résilience V6 (Multi-Env) en cours...")
for gen in range(nb_gen):
    results = [eval_multi_env(ind) for ind in population]
    # Tri par FITNESS
    idx_sorted = np.argsort([r["fitness"] for r in results])[::-1]
    
    print(f"Gen {gen:02d} | Fit: {results[idx_sorted[0]]['fitness']:.3f} | Surv: {results[idx_sorted[0]]['surv_frac']*100:.0f}%")

    meta_history.append({
        "kappas": [population[i]["meta"]["kappa_gain"] for i in range(pop_size)],
        "coolings": [population[i]["meta"]["cooling_strength"] for i in range(pop_size)],
        "fitness": [r["fitness"] for r in results],
        "surv_frac": [r["surv_frac"] for r in results]
    })
    
    # Reproduction (Élitisme + Tournoi)
    next_pop = [population[idx_sorted[0]], population[idx_sorted[1]]]
    while len(next_pop) < pop_size:
        p_idx = max(np.random.choice(pop_size, 3), key=lambda i: results[i]["fitness"])
        p = population[p_idx]
        next_pop.append(creer_individu(p["theta"], p["meta"], f"G{gen}_A{p_idx}"))
    population = next_pop

# --- 4. EXPORT ---
with open("meta_history_v6.json", "w") as f: json.dump(meta_history, f, indent=4)
np.savez("final_population_v6.npz", population=population)
print("✅ Simulation V6 terminée.")