import numpy as np
import os
from scipy.stats import entropy

# --- 1. FONCTIONS DE MESURE ---
def calculate_shannon_entropy(matrix):
    counts, _ = np.histogram(matrix, bins=10)
    return entropy(counts)

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

# --- 3. CRÉATION DE LA GÉNÉRATION 3 ---
population_size = 10
mutation_rate = 0.05
steps = 5000

print(f"🌍 Initialisation V3.3 (Tesla Harmonics) avec {population_size} descendants...")
generation_3 = []

for i in range(population_size):
    mutation = np.random.randn(*alpha_theta.shape) * mutation_rate
    theta_enfant = alpha_theta + mutation
    # Refroidissement initial
    theta_enfant = (theta_enfant - np.mean(theta_enfant)) / (np.std(theta_enfant) + 1e-8) * 0.01
    generation_3.append(theta_enfant)

# --- 4. L'ARÈNE THERMIQUE ---
print("🔥 Début du test de survie (Résonance 3-6-9 activée)...\n")
survival_times = []

for index, agent_theta in enumerate(generation_3):
    heat = 0.0
    kappa = 1.0 
    
    for step in range(1, steps + 1):
        # ⚡️ TESLA HARMONIC COOLING : Tous les 9 cycles, on dissipe la chaleur
        if step % 9 == 0:
            agent_theta *= 0.98 # Micro-compression de la variance (Refroidissement)
            
        var = np.var(agent_theta)
        
        heat += (var * 10) - 0.5
        if heat > 1000:
            break
            
        kappa = np.clip(kappa + 0.002 * (0.2 - var), 0.5, 2.5)
        agent_theta += np.random.randn(16, 16) * 0.01
        
    survival_times.append(step)
    print(f"Agent {index + 1} | Longévité : {step} cycles | Chaleur finale : {heat:.2f}")

# --- 5. RÉSULTATS ---
moyenne_survie = np.mean(survival_times)
print("\n📊 --- BILAN DE LA GÉNÉRATION 3 ---")
print(f"Longévité moyenne : {moyenne_survie:.0f} cycles")

if moyenne_survie > 2000:
    print("✅ SUCCÈS ABSOLU : La résonance Tesla a brisé la limite de Landauer !")
else:
    print("⚠️ STAGNATION : Ajustement des harmoniques nécessaire.")