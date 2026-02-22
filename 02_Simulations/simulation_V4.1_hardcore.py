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
            "kappa_gain": float(np.random.uniform(0.1, 0.4)), # Plus réactif par défaut
            "cooling_strength": float(np.random.uniform(0.90, 0.96)), # Refroidissement plus musclé
            "mutation_rate": 0.02,
            "target_base": 0.04
        }
        p_id = "Genesis"
    else:
        meta = copy.deepcopy(meta_parent)
        for key in ["kappa_gain", "cooling_strength"]: 
            meta[key] *= (1.0 + np.random.randn() * 0.02)
        p_id = id_parent
    return {"theta": theta, "meta": meta, "parent_id": p_id}

# --- L'ARÈNE DE TEMPÊTE (V4.1) ---
def evaluer_agent_hardcore(ind, cycles_max=3000):
    theta, meta = ind["theta"].copy(), ind["meta"]
    mem = np.zeros_like(theta)
    heat, total_heat, entropies = 0.0, 0.0, []
    
    for step in range(1, cycles_max + 1):
        # 🌪️ AJOUT DU CHAOS : 2% de chance d'une tempête de données à chaque cycle
        chaos = 5.0 if np.random.rand() < 0.02 else 1.0
        env = np.sin(step/50.0) * np.cos(step/15.0) * chaos
        
        mem = 0.95 * mem + 0.05 * env
        target = meta["target_base"] + 0.02 * abs(env - mem.mean())
        
        if step % 9 == 0: theta *= meta["cooling_strength"]
        
        var = np.var(theta)
        # 🌡️ PRESSION PHYSIQUE : Plus sensible à la chaleur
        heat = max(0.0, heat + (var * 18.0) - 0.3)
        total_heat += heat
        
        ent = calculer_entropie(theta)
        entropies.append(ent)
        
        if heat > 400 or var < 0.02 or ent < 0.8: break
        
        kappa = np.clip(1.0 + meta["kappa_gain"] * (target - var), 0.1, 4.0)
        theta += (np.random.randn(16, 16) * (0.02 * kappa * meta["mutation_rate"])) + (env * 0.005)
    
    # Score basé sur la survie ET l'efficience sous stress
    score = (step * np.mean(entropies)) / (total_heat + 1e-9) if entropies else 0
    return {"survival": step, "score": score, "meta": meta, "id": ind.get("id"), "parent_id": ind["parent_id"]}

# --- MAIN : ENTRAÎNEMENT LONG ---
alpha_path = "../03_Core/alpha_post_arena.npz"
adn_base = np.load(alpha_path)[np.load(alpha_path).files[0]]
nb_gen, pop_size = 50, 30 # 50 générations pour forcer la convergence
population = [creer_individu(adn_base) for _ in range(pop_size)]
meta_history = []

print("🌪️ Lancement de l'Arène de Tempête (V4.1 Hardcore)...")
for gen in range(nb_gen):
    for i, ind in enumerate(population): ind["id"] = f"G{gen}_A{i}"
    results = [evaluer_agent_hardcore(ind) for ind in population]
    
    # Log de survie pour monitorer
    survie_moy = np.mean([r["survival"] for r in results])
    print(f"Gen {gen:02d} | Survie Moyenne : {survie_moy:4.0f} cycles")

    # Sérialisation
    gen_data = {"kappas": [r["meta"]["kappa_gain"] for r in results],
                "coolings": [r["meta"]["cooling_strength"] for r in results],
                "scores": [r["score"] for r in results],
                "parent_child_pairs": []}
    
    # Sélection & Reproduction (Tournoi plus sélectif k=5)
    parents_idx = [max(np.random.choice(pop_size, 5, replace=False), key=lambda i: results[i]["score"]) for _ in range(int(pop_size*0.2))]
    next_pop = []
    while len(next_pop) < pop_size:
        p_idx = np.random.choice(parents_idx)
        p = results[p_idx]
        enfant = creer_individu(population[p_idx]["theta"], p["meta"], p["id"])
        gen_data["parent_child_pairs"].append({"parent_kappa": p["meta"]["kappa_gain"], "child_kappa": enfant["meta"]["kappa_gain"]})
        next_pop.append(enfant)
    
    meta_history.append(gen_data)
    population = next_pop

# --- EXPORT FINAL ---
with open("meta_history.json", "w") as f: json.dump(meta_history, f, indent=4)
np.savez("final_population.npz", population=population)
print("✅ Entraînement terminé. Prêt pour la re-validation.")