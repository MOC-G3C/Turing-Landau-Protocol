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
        meta = {
            "kappa_gain": float(np.random.uniform(0.01, 0.6)),
            "cooling_strength": float(np.random.uniform(0.95, 0.995)),
            "mutation_rate": float(np.random.uniform(0.01, 0.05)),
            "target_base": float(np.random.uniform(0.02, 0.05))
        }
        p_id = "Genesis"
    else:
        meta = copy.deepcopy(meta_parent)
        for key in meta: meta[key] *= (1.0 + np.random.randn() * 0.02)
        p_id = id_parent
    return {"theta": theta, "meta": meta, "parent_id": p_id}

def evaluer_agent(ind, cycles_max=5000, heat_threshold=500):
    theta, meta = ind["theta"].copy(), ind["meta"]
    mem = np.zeros_like(theta)
    heat, total_heat, entropies = 0.0, 0.0, []
    for step in range(1, cycles_max + 1):
        env = np.sin(step/50.0) * np.cos(step/15.0) * (1.0 + 0.5 * np.random.randn())
        mem = 0.95 * mem + 0.05 * env
        target = meta["target_base"] + 0.02 * abs(env - mem.mean())
        if step % 9 == 0: theta *= meta["cooling_strength"] # Tesla 3-6-9
        var = np.var(theta)
        delta = (var * 15.0) - 0.3
        heat = max(0.0, heat + delta)
        total_heat += heat
        ent = calculer_entropie(theta)
        entropies.append(ent)
        if heat > heat_threshold or var < 0.02 or ent < 0.8: break
        kappa = np.clip(1.0 + meta["kappa_gain"] * (target - var), 0.1, 4.0)
        theta += (np.random.randn(16, 16) * (0.02 * kappa * meta["mutation_rate"])) + (env * 0.005)
    score = np.mean(entropies) / (total_heat + 1e-9) if entropies else 0
    return {"survival": step, "score": score, "meta": meta, "parent_id": ind["parent_id"]}

# --- MAIN ---
alpha_path = "../03_Core/alpha_post_arena.npz"
adn_base = np.load(alpha_path)[np.load(alpha_path).files[0]]
nb_gen, pop_size = 20, 30
population = [creer_individu(adn_base) for _ in range(pop_size)]
meta_history = []

print("🚀 Evolution Méta-Génétique en cours...")
for gen in range(nb_gen):
    # Attribution des IDs
    for i, ind in enumerate(population): ind["id"] = f"G{gen}_A{i}"
    
    results = [evaluer_agent(ind) for ind in population]
    
    # Sérialisation
    gen_data = {
        "kappas": [ind["meta"]["kappa_gain"] for ind in population],
        "coolings": [ind["meta"]["cooling_strength"] for ind in population],
        "scores": [r["score"] for r in results],
        "parent_child_pairs": []
    }
    
    # On cherche les parents dans la population qui vient de finir (celle-ci) pour la GEN SUIVANTE
    # Note: On stocke la pop actuelle pour que la gen suivante puisse s'y référer
    pop_reference = copy.deepcopy(population)
    
    meta_history.append(gen_data)
    
    # Sélection & Reproduction
    parents_idx = [max(np.random.choice(pop_size, 3, replace=False), key=lambda i: results[i]["score"]) for _ in range(int(pop_size*0.25))]
    next_pop = []
    while len(next_pop) < pop_size:
        p_idx = np.random.choice(parents_idx)
        p = population[p_idx]
        enfant = creer_individu(p["theta"], p["meta"], p["id"])
        
        # Enregistrement immédiat du lien de parenté pour la stat d'héritabilité
        if len(meta_history) > 0:
            meta_history[-1]["parent_child_pairs"].append({
                "parent_kappa": p["meta"]["kappa_gain"],
                "child_kappa": enfant["meta"]["kappa_gain"]
            })
        next_pop.append(enfant)
    population = next_pop

# --- EXPORTS ---
with open("meta_history.json", "w") as f: json.dump(meta_history, f, indent=4)
np.savez("final_population.npz", population=population)
print("✅ Exports terminés avec liens génétiques.")