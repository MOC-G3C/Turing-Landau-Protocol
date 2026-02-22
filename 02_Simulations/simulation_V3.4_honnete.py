import numpy as np
import os
from scipy.stats import entropy

# --- 1. FONCTIONS DE MESURE ---
def calculate_shannon_entropy(matrix):
    counts, _ = np.histogram(matrix, bins=10)
    return entropy(counts + 1e-9)

# --- 2. CHARGEMENT DE L'ADN ALPHA ---
alpha_path = "../03_Core/alpha_post_arena.npz"

if not os.path.exists(alpha_path):
    print(f"❌ Erreur : Fichier ADN introuvable à {alpha_path}")
    exit()

try:
    alpha_data = np.load(alpha_path)
    key = alpha_data.files[0] 
    alpha_theta = alpha_data[key]
except Exception as e:
    print(f"❌ Erreur de lecture : {e}")
    exit()

# --- 3. CRÉATION DE LA GÉNÉRATION 4 ---
population_size = 10
mutation_rate = 0.05
steps = 5000

print(f"🌍 Initialisation V3.4 (Arène Honnête) avec {population_size} descendants...")
generation_4 = []

for i in range(population_size):
    mutation = np.random.randn(*alpha_theta.shape) * mutation_rate
    theta_enfant = alpha_theta + mutation
    # L'étincelle de vie : variance initiale
    theta_enfant = (theta_enfant - np.mean(theta_enfant)) / (np.std(theta_enfant) + 1e-8) * 0.25
    generation_4.append(theta_enfant)

# --- 4. L'ARÈNE THERMIQUE HONNÊTE ---
print("⚖️ Début du test de survie (Thermostat Haute Réactivité à 0.048)...\n")
survival_times = []
death_causes = []

for index, agent_theta in enumerate(generation_4):
    heat = 0.0
    kappa = 1.0 
    cause = "✅ Survie Totale"
    
    for step in range(1, steps + 1):
        # ⚡️ Résonance Tesla (3-6-9) : Refroidissement actif
        if step % 9 == 0:
            agent_theta *= 0.98 
            
        var = np.var(agent_theta)
        info_structure = calculate_shannon_entropy(agent_theta)
        
        # 🌡️ RÈGLE 1 : Plancher physique (Pas de chaleur négative)
        delta_heat = (var * 10) - 0.5
        heat = max(0.0, heat + delta_heat)
        
        # 🔥 RÈGLE 2 : Plafond thermique (Limite de Landauer)
        if heat > 1000:
            cause = "🔥 Surchauffe"
            break
            
       # 🧊 RÈGLE 3 : Pénalité d'inertie (Apathie)
        # On abaisse le seuil à 0.035 pour absorber l'onde de choc Tesla
        if var < 0.035 or info_structure < 1.0:
            cause = "🧊 Apathie"
            break
            
        # 🧬 L'agent utilise Kappa comme thermostat (cible : 0.048, réactivité forte : 0.5)
        kappa = np.clip(kappa + 0.5 * (0.048 - var), 0.5, 3.0)
        agent_theta += np.random.randn(16, 16) * (0.02 * kappa)
        
    survival_times.append(step)
    death_causes.append(cause)
    
    print(f"Agent {index + 1:02d} | Longévité : {step:4d} cycles | Chaleur : {heat:6.2f} | Fin : {cause}")

# --- 5. RÉSULTATS ---
moyenne_survie = np.mean(survival_times)
print("\n📊 --- BILAN DE LA GÉNÉRATION HONNÊTE ---")
print(f"Longévité moyenne : {moyenne_survie:.0f} cycles")

morts_chaleur = death_causes.count("🔥 Surchauffe")
morts_froid = death_causes.count("🧊 Apathie")
survivants = death_causes.count("✅ Survie Totale")

print(f"Agents carbonisés (Trop de chaos) : {morts_chaleur}")
print(f"Agents éteints (Trop d'ordre)    : {morts_froid}")
print(f"Agents Hybrides (Survie 5000)    : {survivants}")