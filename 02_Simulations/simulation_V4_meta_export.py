import numpy as np
import copy
import json
import os
from scipy.stats import entropy

# --- FONCTIONS DE BASE ---
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
        parent_id = "Genesis"
    else:
        meta = copy.deepcopy(meta_parent)
        # Mutation réduite à 2% pour stabiliser l'évolution
        for key in meta:
            meta[key] *= (1.0 + np.random.randn() * 0.02) 
            
        meta["kappa_gain"] = np.clip(meta["kappa_gain"], 0.01, 1.0)
        meta["cooling_strength"] = np.clip(meta["cooling_strength"], 0.90, 0.999)
        meta["mutation_rate"] = np.clip(meta["mutation_rate"], 0.001, 0.1)
        meta["target_base"] = np.clip(meta["target_base"], 0.01, 0.1)
        parent_id = id_parent

    return {"theta": theta, "meta": meta, "parent_id": parent_id}

def evaluer_agent(ind, cycles_max=5000, heat_threshold=500):
    theta = ind["theta"].copy()
    meta = ind["meta"]
    mem = np.zeros_like(theta)
    heat, total_heat = 0.0, 0.0
    entropies = []
    
    for step in range(1, cycles_max + 1):
        signal_env = np.sin(step/50.0) * np.cos(step/15.0) * (1.0 + 0.5 * np.random.randn())
        mem = 0.95 * mem + 0.05 * signal_env
        target_var = meta["target_base"] + 0.02 * abs(signal_env - mem.mean())
        
        if step % 9 == 0:
            theta *= meta["cooling_strength"]
            
        var = np.var(theta)
        delta_heat = (var * 15.0) - 0.3
        heat = max(0.0, heat + delta_heat)
        total_heat += heat
        
        ent = calculer_entropie(theta)
        entropies.append(ent)
        
        if heat > heat_threshold or var < 0.02 or ent < 0.8:
            break
            
        kappa = np.clip(1.0 + meta["kappa_gain"] * (target_var - var), 0.1, 4.0)
        theta += (np.random.randn(*theta.shape) * (0.02 * kappa * meta["mutation_rate"])) + (signal_env * 0.005)
        
    avg_ent = np.mean(entropies) if entropies else 0.0
    info_per_heat = avg_ent / (total_heat + 1e-9)
    return {"survival": step, "score": info_per_heat, "meta": meta, "parent_id": ind["parent_id"]}

# --- INITIALISATION ---
chemin_alpha = "../03_Core/alpha_post_arena.npz"
adn_base = np.load(chemin_alpha)[np.load(chemin_alpha).files[0]]

nb_generations = 20
taille_population = 30
population = [creer_individu(adn_base) for _ in range(taille_population)]

# Base de données de l'histoire évolutive
histoire_complete = []

print(f"🌍 Lancement de la simulation (Exportation JSON activée)...")

# --- BOUCLE ÉVOLUTIVE ---
for gen in range(nb_generations):
    # On donne un ID unique à chaque agent de cette génération
    for i, ind in enumerate(population):
        ind["id"] = f"G{gen}_A{i}"
        
    resultats = [evaluer_agent(ind) for ind in population]
    
    # Sauvegarde des données de cette génération
    donnees_gen = {
        "generation": gen,
        "kappas": [r["meta"]["kappa_gain"] for r in resultats],
        "coolings": [r["meta"]["cooling_strength"] for r in resultats],
        "scores": [r["score"] for r in resultats],
        "lignees": [{"enfant": population[i]["id"], "parent": r["parent_id"], "enfant_kappa": r["meta"]["kappa_gain"], "parent_kappa": next((p["meta"]["kappa_gain"] for p in population if p["id"] == r["parent_id"]), None)} for i, r in enumerate(resultats) if r["parent_id"] != "Genesis"]
    }
    histoire_complete.append(donnees_gen)
    
    print(f"Gen {gen+1:02d} traitée et sauvegardée.")
    
    # Sélection par tournoi
    parents_idx = []
    for _ in range(max(2, int(taille_population * 0.25))):
        combattants = np.random.choice(taille_population, 3, replace=False)
        meilleur = max(combattants, key=lambda i: resultats[i]["score"])
        parents_idx.append(meilleur)
        
    # Reproduction
    nouvelle_pop = []
    while len(nouvelle_pop) < taille_population:
        parent_idx = np.random.choice(parents_idx)
        parent_gagnant = population[parent_idx]
        enfant = creer_individu(parent_gagnant["theta"], parent_gagnant["meta"], id_parent=parent_gagnant["id"])
        nouvelle_pop.append(enfant)
        
    population = nouvelle_pop

# --- EXPORTATION ---
with open("meta_history.json", "w") as f:
    json.dump(histoire_complete, f, indent=4)
print("✅ Données génétiques sauvegardées dans 'meta_history.json'.")