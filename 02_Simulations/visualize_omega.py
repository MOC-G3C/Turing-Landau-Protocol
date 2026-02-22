import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. CHARGEMENT DE L'OMEGA MASTER DNA ---
pop_file = "final_population_v8_7.npz"
if not os.path.exists(pop_file):
    print(f"❌ Erreur : Fichier {pop_file} introuvable.")
    exit()

data = np.load(pop_file, allow_pickle=True)
population = data['population']

# On prend le champion absolu (le premier de la liste triée)
champion = population[0] 
theta0 = champion["theta"].copy()
meta = champion["meta"]

print(f"🧬 Radiographie du Master DNA V8.8 :")
print(f"   - Tau (Mémoire) : {meta.get('tau'):.4f}")
print(f"   - Gain Signal (Calme) : {meta.get('env_gain_calm'):.4f}")
print(f"   - Gain Signal (Spike) : {meta.get('env_gain_spike'):.4f}")
print(f"   - Refroidissement (Tesla) : {meta.get('cooling_strength'):.4f}")

# --- 2. SIMULATION DU TEST (L'Électroencéphalogramme) ---
steps = 500
env_signals = []
variances = []
chaleur = []

theta = theta0.copy()
mem = np.zeros_like(theta)
heat = 0.0

for t in range(1, steps + 1):
    # Création d'un environnement calme avec 3 gros chocs (spikes) aux cycles 100, 250, 400
    is_spike = t in [100, 101, 102, 250, 251, 252, 400, 401, 402]
    env_val = np.sin(t / 20.0) * (8.0 if is_spike else 1.0)
        
    env_signals.append(env_val)
    
    # Mécanique métacellulaire du champion
    gs = meta["env_gain_spike"] if is_spike else meta["env_gain_calm"]
    mem = (1.0 - meta["tau"]) * mem + meta["tau"] * env_val
    target = meta["target_base"] + gs * abs(env_val - np.mean(mem))
    
    if t % 9 == 0: 
        theta *= meta["cooling_strength"]
    
    var = np.var(theta)
    variances.append(var)
    
    heat = max(0.0, heat + (var * 15.0) - 0.3)
    chaleur.append(heat)
    
    kappa = np.clip(1.0 + meta["kappa_gain"] * (target - var), 0.1, 4.0)
    theta += (np.random.randn(16, 16) * (0.02 * kappa * meta["mutation_rate"])) + (env_val * 0.005)

# --- 3. VISUALISATION ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.canvas.manager.set_window_title("Radiographie de l'Axe Hybride (V8.8)")

# Graphique 1 : L'Environnement
ax1.plot(env_signals, color='gray', label="Signal Externe (Spikes)")
ax1.set_title("Météo Cognitive (L'Environnement)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Graphique 2 : La Réponse de l'Agent (La Pensée)
ax2.plot(variances, color='blue', label="Variance Interne (Activité cérébrale)")
ax2.axhline(meta["low_var_thresh"], color='red', linestyle='--', label="Seuil d'Apathie (Mort)")
ax2.set_title("L'Anamnèse en Action")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

# Graphique 3 : La Chaleur (Contrainte de Landauer)
ax3.plot(chaleur, color='orange', label="Chaleur Accumulée (Joules)")
ax3.set_title("Friction Thermodynamique")
ax3.set_xlabel("Cycles (Temps)")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("radiographie_master_dna.png")
print("✅ Graphique généré et sauvegardé sous 'radiographie_master_dna.png'.")