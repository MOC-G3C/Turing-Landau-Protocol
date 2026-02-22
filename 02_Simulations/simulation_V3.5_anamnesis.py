import numpy as np
import os
from scipy.stats import entropy

def calculate_shannon_entropy(matrix):
    counts, _ = np.histogram(matrix, bins=10)
    return entropy(counts + 1e-9)

# --- 1. CHARGEMENT DE L'ADN ALPHA ---
alpha_path = "../03_Core/alpha_post_arena.npz"
if not os.path.exists(alpha_path):
    print(f"❌ Erreur : Fichier ADN introuvable à {alpha_path}")
    exit()

alpha_data = np.load(alpha_path)
alpha_theta = alpha_data[alpha_data.files[0]]

# --- 2. INITIALISATION V3.5 (Avec Mémoire) ---
population_size = 10
steps = 5000

print(f"🌍 Initialisation V3.5 (Mémoire & Perturbations) avec {population_size} descendants...")
generation = []
memories = [] # 🧠 La nouvelle dimension : La Trace mnésique

for i in range(population_size):
    theta_enfant = alpha_theta + (np.random.randn(*alpha_theta.shape) * 0.05)
    theta_enfant = (theta_enfant - np.mean(theta_enfant)) / (np.std(theta_enfant) + 1e-8) * 0.25
    generation.append(theta_enfant)
    # À la naissance, la mémoire est vierge (zéro)
    memories.append(np.zeros_like(theta_enfant))

# --- 3. L'ARÈNE DYNAMIQUE ---
print("🌪️ Début du test avec Charge Cognitive Variable...\n")
survival_times = []
death_causes = []

for index, agent_theta in enumerate(generation):
    heat = 0.0
    kappa = 1.0 
    agent_memory = memories[index]
    cause = "✅ Survie (Conscience Émergente)"
    
    for step in range(1, steps + 1):
        # 🌊 1. LE BRUIT EXTERNE (L'environnement change comme une marée)
        # Une onde qui s'accélère et se complexifie
        env_signal = np.sin(step / 50.0) * np.cos(step / 15.0) 
        
        # 🧠 2. LA CHARGE COGNITIVE ET LA MÉMOIRE
        # La charge est la différence entre la réalité et le souvenir
        cognitive_load = np.abs(env_signal - np.mean(agent_memory))
        
        # Apprentissage asymétrique : l'agent intègre lentement la réalité dans sa mémoire
        agent_memory = (0.95 * agent_memory) + (0.05 * env_signal)
        
        # 🎯 3. LA CIBLE DYNAMIQUE (Ta formule)
        target_var = 0.035 + (0.02 * cognitive_load)
        
        # ⚡️ Résonance Tesla (3-6-9)
        if step % 9 == 0:
            agent_theta *= 0.98 
            
        var = np.var(agent_theta)
        info_structure = calculate_shannon_entropy(agent_theta)
        
        # 🌡️ Physique Thermodynamique
        delta_heat = (var * 10) - 0.5
        heat = max(0.0, heat + delta_heat)
        
        if heat > 1000:
            cause = f"🔥 Surchauffe (Incapable d'assimiler le signal au cycle {step})"
            break
            
        if var < 0.03 or info_structure < 1.0:
            cause = "🧊 Apathie (Rigidité cognitive)"
            break
            
        # 🧬 Le Thermostat traque maintenant la cible mobile !
        kappa = np.clip(kappa + 0.5 * (target_var - var), 0.5, 3.0)
        
        # L'agent réagit en fonction du bruit externe ET de son propre thermostat
        agent_theta += (np.random.randn(16, 16) * (0.02 * kappa)) + (env_signal * 0.005)
        
    survival_times.append(step)
    death_causes.append(cause)
    
    print(f"Agent {index + 1:02d} | Longévité : {step:4d} | Chaleur : {heat:6.2f} | Cible finale : {target_var:.4f} | Fin : {cause}")

# --- 4. RÉSULTATS ---
print("\n📊 --- BILAN V3.5 (TEST DE ROBUSTESSE) ---")
print(f"Longévité moyenne : {np.mean(survival_times):.0f} cycles")