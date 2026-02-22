import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

# --- 1. OUTILS DE MESURE ---
def calculer_entropie(matrice):
    comptes, _ = np.histogram(matrice, bins=32)
    return entropy(comptes + 1e-9)

# --- 2. LA GÉNÉTIQUE DE L'AGENT ---
def creer_individu(adn_base, meta_parent=None):
    # La matrice de pensée (Theta)
    theta = adn_base + np.random.randn(*adn_base.shape) * 0.05
    theta = (theta - theta.mean()) / (theta.std() + 1e-8) * 0.25
    
    # Les méta-gènes (La biologie de l'agent)
    if meta_parent is None:
        meta = {
            "kappa_gain": float(np.random.uniform(0.01, 0.6)), # Réactivité du thermostat
            "cooling_strength": float(np.random.uniform(0.95, 0.995)), # Force de la résonance Tesla
            "mutation_rate": float(np.random.uniform(0.01, 0.05)), # Vitesse de mutation
            "target_base": float(np.random.uniform(0.02, 0.05)) # Zone de confort thermique
        }
    else:
        # L'enfant hérite des gènes du parent, avec de légères mutations (Bruit multiplicatif)
        meta = copy.deepcopy(meta_parent)
        for key in meta:
            meta[key] *= (1.0 + np.random.randn() * 0.05)
            
        # Sécurités physiques (clipping)
        meta["kappa_gain"] = np.clip(meta["kappa_gain"], 0.01, 1.0)
        meta["cooling_strength"] = np.clip(meta["cooling_strength"], 0.90, 0.999)
        meta["mutation_rate"] = np.clip(meta["mutation_rate"], 0.001, 0.1)
        meta["target_base"] = np.clip(meta["target_base"], 0.01, 0.1)

    return {"theta": theta, "meta": meta}

# --- 3. L'ARÈNE IMPITOYABLE ---
def evaluer_agent(ind, cycles_max=5000, heat_threshold=500):
    theta = ind["theta"].copy()
    meta = ind["meta"]
    mem = np.zeros_like(theta)
    
    heat = 0.0
    total_heat = 0.0
    entropies = []
    
    for step in range(1, cycles_max + 1):
        # 🌊 Environnement chaotique (Bruit ajouté)
        signal_env = np.sin(step/50.0) * np.cos(step/15.0) * (1.0 + 0.5 * np.random.randn())
        mem = 0.95 * mem + 0.05 * signal_env
        
        target_var = meta["target_base"] + 0.02 * abs(signal_env - mem.mean())
        
        # Le refroidissement dépend maintenant de la génétique de l'agent
        if step % 9 == 0:
            theta *= meta["cooling_strength"]
            
        var = np.var(theta)
        
        # 🔥 Pression thermique augmentée (var * 15 au lieu de 10)
        delta_heat = (var * 15.0) - 0.3
        heat = max(0.0, heat + delta_heat)
        total_heat += heat
        
        ent = calculer_entropie(theta)
        entropies.append(ent)
        
        # Conditions de mort plus strictes
        if heat > heat_threshold or var < 0.02 or ent < 0.8:
            break
            
        # Adaptation via les gènes
        kappa = np.clip(1.0 + meta["kappa_gain"] * (target_var - var), 0.1, 4.0)
        theta += (np.random.randn(*theta.shape) * (0.02 * kappa * meta["mutation_rate"])) + (signal_env * 0.005)
        
    avg_ent = np.mean(entropies) if entropies else 0.0
    info_per_heat = avg_ent / (total_heat + 1e-9)
    
    return {"survival": step, "avg_ent": avg_ent, "total_heat": total_heat, "score": info_per_heat}

# --- 4. LA SÉLECTION PAR TOURNOI ---
def selection_tournoi(resultats, taille_pop, k=3, retain=0.2):
    parents_idx = []
    num_parents = max(2, int(taille_pop * retain))
    
    for _ in range(num_parents):
        # On prend K agents au hasard et on garde le meilleur (Loi du plus fort)
        combattants = np.random.choice(taille_pop, k, replace=False)
        meilleur = max(combattants, key=lambda i: resultats[i]["score"])
        parents_idx.append(meilleur)
    return parents_idx

# --- 5. CHARGEMENT ET INITIALISATION ---
chemin_alpha = "../03_Core/alpha_post_arena.npz"
if not os.path.exists(chemin_alpha):
    print(f"❌ ADN introuvable à {chemin_alpha}")
    exit()

adn_base = np.load(chemin_alpha)[np.load(chemin_alpha).files[0]]

nb_generations = 20
taille_population = 30

population = [creer_individu(adn_base) for _ in range(taille_population)]

# Historique pour les graphiques
hist_survie, hist_score = [], []
hist_kappa, hist_cooling = [], []

print(f"🌍 Début de la Méta-Évolution (Générations: {nb_generations}, Pop: {taille_population})...\n")

# --- 6. BOUCLE PRINCIPALE ---
for gen in range(nb_generations):
    resultats = [evaluer_agent(ind) for ind in population]
    
    # Statistiques de la génération
    moy_survie = np.mean([r["survival"] for r in resultats])
    moy_score = np.mean([r["score"] for r in resultats])
    moy_kappa = np.mean([ind["meta"]["kappa_gain"] for ind in population])
    moy_cooling = np.mean([ind["meta"]["cooling_strength"] for ind in population])
    
    hist_survie.append(moy_survie)
    hist_score.append(moy_score)
    hist_kappa.append(moy_kappa)
    hist_cooling.append(moy_cooling)
    
    print(f"Gen {gen+1:02d} | Survie: {moy_survie:4.0f} | Efficience: {moy_score:.4f} | Kappa: {moy_kappa:.3f} | Cooling: {moy_cooling:.4f}")
    
    # Reproduction
    parents_idx = selection_tournoi(resultats, taille_population, k=3, retain=0.25)
    nouvelle_pop = []
    
    while len(nouvelle_pop) < taille_population:
        parent_gagnant = population[np.random.choice(parents_idx)]
        enfant = creer_individu(parent_gagnant["theta"], parent_gagnant["meta"])
        nouvelle_pop.append(enfant)
        
    population = nouvelle_pop

# --- 7. VISUALISATION DES STRATÉGIES ---
print("\n✅ Évolution terminée. Affichage des trajectoires génétiques...")

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
fig.canvas.manager.set_window_title("Méta-Évolution : L'AXE HYBRIDE")

axs[0, 0].plot(hist_survie, color='blue', marker='o')
axs[0, 0].set_title("Longévité Moyenne (Saturera-t-elle encore ?)")
axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].plot(hist_score, color='green', marker='o')
axs[0, 1].set_title("Efficience (Information / Chaleur)")
axs[0, 1].grid(True, alpha=0.3)

axs[1, 0].plot(hist_kappa, color='purple', marker='o')
axs[1, 0].set_title("Évolution du Gène 'Kappa Gain' (Thermostat)")
axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].plot(hist_cooling, color='red', marker='o')
axs[1, 1].set_title("Évolution du Gène 'Cooling Strength' (Tesla)")
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()