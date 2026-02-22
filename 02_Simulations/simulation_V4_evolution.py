import numpy as np
import zlib
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

# --- 1. OUTILS DE MESURE (SIMPLES ET DIRECTS) ---
def calculer_entropie(matrice):
    # Mesure la quantité d'information (la structure)
    comptes, _ = np.histogram(matrice, bins=32)
    return entropy(comptes + 1e-9)

def ratio_compression(matrice):
    # Plus le ratio est bas, plus la matrice est structurée (moins chaotique)
    b = (matrice - matrice.min()) / (matrice.max() - matrice.min() + 1e-9)
    octets = (b * 255).astype(np.uint8).tobytes()
    compresse = zlib.compress(octets)
    return len(compresse) / len(octets)

# --- 2. PARAMÈTRES DE L'ÉVOLUTION ---
nb_generations = 10
taille_population = 20
cycles_max = 5000

# Historique pour les graphiques de preuve
historique_longevite = []
historique_efficience = []

# --- 3. CHARGEMENT DE L'ANCÊTRE (ALPHA) ---
chemin_alpha = "../03_Core/alpha_post_arena.npz"
if not os.path.exists(chemin_alpha):
    print(f"❌ Erreur : ADN introuvable à {chemin_alpha}")
    exit()

donnees_alpha = np.load(chemin_alpha)
adn_base = donnees_alpha[donnees_alpha.files[0]]

# Création de la toute première population
population_actuelle = []
for _ in range(taille_population):
    enfant = adn_base + (np.random.randn(*adn_base.shape) * 0.05)
    enfant = (enfant - np.mean(enfant)) / (np.std(enfant) + 1e-8) * 0.25
    population_actuelle.append(enfant)

print("🌍 Début du programme d'Évolution Naturelle (V4)...\n")

# --- 4. LA BOUCLE DES GÉNÉRATIONS ---
for generation in range(nb_generations):
    resultats_generation = []
    
    # Test de chaque agent dans l'arène
    for index, adn_agent in enumerate(population_actuelle):
        memoire = np.zeros_like(adn_agent)
        chaleur = 0.0
        kappa = 1.0
        chaleur_totale_accumulee = 0.0
        entropie_moyenne = 0.0
        
        for cycle in range(1, cycles_max + 1):
            # Le bruit de l'environnement (la météo cognitive)
            signal_env = np.sin(cycle / 50.0) * np.cos(cycle / 15.0)
            
            # La mémoire absorbe le choc
            charge_cognitive = abs(signal_env - np.mean(memoire))
            memoire = (0.95 * memoire) + (0.05 * signal_env)
            
            cible_variance = 0.035 + (0.02 * charge_cognitive)
            
            # Résonance Tesla
            if cycle % 9 == 0:
                adn_agent *= 0.98
                
            var = np.var(adn_agent)
            info = calculer_entropie(adn_agent)
            
            # Thermodynamique
            delta_chaleur = (var * 10) - 0.5
            chaleur = max(0.0, chaleur + delta_chaleur)
            chaleur_totale_accumulee += chaleur
            entropie_moyenne += info
            
            # Vérification de la survie
            if chaleur > 1000 or var < 0.03 or info < 1.0:
                break
                
            # Adaptation
            kappa = np.clip(kappa + 0.5 * (cible_variance - var), 0.5, 3.0)
            adn_agent += (np.random.randn(16, 16) * (0.02 * kappa)) + (signal_env * 0.005)

        # Fin de vie de l'agent : on calcule son score final
        entropie_moyenne /= cycle
        # LE SCORE ULTIME : Information générée divisée par la chaleur produite
        score_efficience = entropie_moyenne / (chaleur_totale_accumulee + 1.0)
        
        resultats_generation.append({
            "adn": adn_agent,
            "longevite": cycle,
            "efficience": score_efficience
        })

    # --- 5. SÉLECTION NATURELLE ET REPRODUCTION ---
    # On trie les agents du meilleur au pire selon leur efficience
    resultats_generation.sort(key=lambda x: x["efficience"], reverse=True)
    
    moyenne_longevite = np.mean([r["longevite"] for r in resultats_generation])
    moyenne_efficience = np.mean([r["efficience"] for r in resultats_generation])
    
    historique_longevite.append(moyenne_longevite)
    historique_efficience.append(moyenne_efficience)
    
    print(f"Génération {generation + 1:02d} | Survie moy : {moyenne_longevite:4.0f} | Efficience moy : {moyenne_efficience:.6f}")
    
    # On garde seulement les 20% meilleurs (l'élite)
    taille_elite = max(2, int(taille_population * 0.2))
    elite = resultats_generation[:taille_elite]
    
    # On recrée une nouvelle population à partir de l'élite
    population_actuelle = []
    for _ in range(taille_population):
        # On choisit un parent au hasard dans l'élite
        parent = elite[np.random.randint(0, taille_elite)]["adn"]
        # On le mute légèrement
        enfant = parent + (np.random.randn(*parent.shape) * 0.02)
        population_actuelle.append(enfant)

# --- 6. PREUVE VISUELLE (GRAPHIQUES) ---
print("\n✅ Évolution terminée. Génération des graphiques de preuve...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.canvas.manager.set_window_title("Résultats de l'Évolution - V4")

# Graphique 1 : La Survie
ax1.plot(range(1, nb_generations + 1), historique_longevite, marker='o', color='b')
ax1.set_title("Évolution de la Longévité")
ax1.set_xlabel("Générations")
ax1.set_ylabel("Cycles de survie (Moyenne)")
ax1.grid(True, alpha=0.3)

# Graphique 2 : L'Efficience (Le Graal)
ax2.plot(range(1, nb_generations + 1), historique_efficience, marker='o', color='g')
ax2.set_title("Évolution de l'Efficience Cognitive")
ax2.set_xlabel("Générations")
ax2.set_ylabel("Score (Information / Chaleur)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()