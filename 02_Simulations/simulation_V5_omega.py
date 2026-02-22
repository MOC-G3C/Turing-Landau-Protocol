import numpy as np
import copy
import json
import os
from scipy.stats import entropy

# --- UTILS ---
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
        # Mutation légère pour ne pas casser l'élite
        for key in ["kappa_gain", "cooling_strength"]: 
            meta[key] *= (1.0 + np.random.randn() * 0.01)
        p_id = id_parent
    return {"theta": theta, "meta": meta, "parent_id": p_id}

# --- ARÈNE OMEGA (SUR-ENTRAÎNEMENT) ---
def evaluer_omega(ind, cycles_max=3000):
    theta, meta = ind["theta"].copy(), ind["meta"]
    mem = np.zeros_like(theta)
    heat, total_heat, entropies = 0.0, 0.0, []
    
    for step in range(1, cycles_max + 1):
        # 🌪️ TEMPÊTE NIVEAU 15 (Pire que le test de validation à 10)
        chaos = 15.0 if np.random.rand() < 0.03 else 1.0
        env = np.sin(step/50.0) * chaos
        
        mem = 0.95 * mem + 0.05 * env
        target = meta["target_base"] + 0.02 * abs(env - mem.mean())
        
        if step % 9 == 0: theta *= meta["cooling_strength"] # Tesla 3-6-9
        
        var = np.var(theta)
        heat = max(0.0, heat + (var * 18.0) - 0.3)
        total_heat += heat
        ent = calculer_entropie(theta)
        entropies.append(ent)
        
        if heat > 400 or var < 0.02 or ent < 0.8: break
        
        kappa = np.clip(1.0 + meta["kappa_gain"] * (target - var), 0.1, 4.0)
        theta += (np.random.randn(16, 16) * (0.02 * kappa * meta["mutation_rate"])) + (env * 0.005)
    
    score = (step * np.mean(entropies)) / (total_heat + 1e-9)
    return {"survival": step, "score": score, "meta": meta, "id": ind.get("id"), "parent_id": ind["parent_id"], "theta": theta}

# --- ÉVOLUTION OMEGA ---
alpha_path = "../03_Core/alpha_post_arena.npz"
adn_base = np.load(alpha_path)[np.load(alpha_path).files[0]]
nb_gen, pop_size = 50, 30
population = [creer_individu(adn_base) for _ in range(pop_size)]
meta_history = []

print("🌀 Forgeron de l'OMEGA Master DNA en cours...")
for gen in range(nb_gen):
    for i, ind in enumerate(population): ind["id"] = f"G{gen}_A{i}"
    results = [evaluer_omega(ind) for ind in population]
    
    # Tri par score pour l'Élitisme
    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"Gen {gen:02d} | Top Score: {results[0]['score']:.4f} | Survie: {results[0]['survival']}")

    gen_data = {"kappas": [r["meta"]["kappa_gain"] for r in results],
                "coolings": [r["meta"]["cooling_strength"] for r in results],
                "scores": [r["score"] for r in results],
                "parent_child_pairs": []}
    meta_history.append(gen_data)
    
    # REPRODUCTION AVEC ÉLITISME (Les 2 meilleurs passent intacts)
    next_pop = [creer_individu(results[0]["theta"], results[0]["meta"], results[0]["id"]),
                creer_individu(results[1]["theta"], results[1]["meta"], results[1]["id"])]
    
    while len(next_pop) < pop_size:
        # Sélection par tournoi sur le reste
        p = results[np.random.randint(0, 10)] 
        enfant = creer_individu(p["theta"], p["meta"], p["id"])
        gen_data["parent_child_pairs"].append({"parent_kappa": p["meta"]["kappa_gain"], "child_kappa": enfant["meta"]["kappa_gain"]})
        next_pop.append(enfant)
    population = next_pop

# --- EXPORT ---
with open("meta_history.json", "w") as f: json.dump(meta_history, f, indent=4)
np.savez("final_population.npz", population=population)
print("✅ Master DNA forgé.")