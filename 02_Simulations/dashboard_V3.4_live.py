import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy.stats import entropy

# --- 1. FONCTIONS DE MESURE ---
def calculate_shannon_entropy(matrix):
    counts, _ = np.histogram(matrix, bins=10)
    return entropy(counts + 1e-9)

# --- 2. CHARGEMENT DE L'ADN ALPHA ---
alpha_path = "../03_Core/alpha_post_arena.npz"
if not os.path.exists(alpha_path):
    print(f"❌ Erreur : ADN introuvable à {alpha_path}")
    exit()

alpha_data = np.load(alpha_path)
alpha_theta = alpha_data[alpha_data.files[0]]

# --- 3. INITIALISATION (10 Agents) ---
population_size = 10
generation = []
heat_levels = [0.0] * population_size
kappas = [1.0] * population_size
active_agents = [True] * population_size

for i in range(population_size):
    mutation = np.random.randn(*alpha_theta.shape) * 0.05
    theta_enfant = alpha_theta + mutation
    # Étincelle de vie
    theta_enfant = (theta_enfant - np.mean(theta_enfant)) / (np.std(theta_enfant) + 1e-8) * 0.25
    generation.append(theta_enfant)

# --- 4. CONFIGURATION DE L'ÉCRAN RADAR ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title("L'AXE HYBRIDE - V3.4 Live Dashboard")

lines = []
for i in range(population_size):
    line, = ax.plot([], [], alpha=0.8, linewidth=1.5)
    lines.append(line)

# Les limites physiques visibles à l'écran
ax.axhline(1000, color='red', linestyle='--', linewidth=2, label="🔥 Plafond de Landauer (Mort)")
ax.axhline(0, color='blue', linestyle=':', linewidth=2, label="❄️ Plancher Physique (Zéro Absolu)")

ax.set_xlim(0, 300)
ax.set_ylim(-50, 1050) # On voit légèrement en dessous de zéro pour vérifier le plancher
ax.set_title("Kybernetes V4 : Survie Thermodynamique Honnête en Temps Réel", fontsize=14)
ax.set_xlabel("Cycles de vie", fontsize=12)
ax.set_ylabel("Chaleur Accumulée", fontsize=12)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.2)

x_data = []
y_data = [[] for _ in range(population_size)]

# --- 5. LE MOTEUR D'ANIMATION (La Vraie Physique) ---
def update(frame):
    step = frame + 1
    x_data.append(step)
    
    for i in range(population_size):
        # Si l'agent est mort, sa ligne devient plate
        if not active_agents[i]:
            y_data[i].append(heat_levels[i])
            continue

        # ⚡️ Résonance Tesla (3-6-9)
        if step % 9 == 0:
            generation[i] *= 0.98 
            
        var = np.var(generation[i])
        info_structure = calculate_shannon_entropy(generation[i])
        
        # 🌡️ Physique Thermodynamique Honnête
        delta_heat = (var * 10) - 0.5
        heat_levels[i] = max(0.0, heat_levels[i] + delta_heat)
        
        # 💀 Vérification de la mort (Surchauffe ou Apathie)
        if heat_levels[i] > 1000 or var < 0.035 or info_structure < 1.0:
            active_agents[i] = False
            
        # 🧬 Thermostat adaptatif
        kappas[i] = np.clip(kappas[i] + 0.5 * (0.048 - var), 0.5, 3.0)
        generation[i] += np.random.randn(16, 16) * (0.02 * kappas[i])
        
        y_data[i].append(heat_levels[i])
        lines[i].set_data(x_data, y_data[i])
        
    # Le radar avance avec le temps (scrolling)
    if step > 300:
        ax.set_xlim(step - 300, step + 10)
        
    return lines

print("🚀 Lancement du tableau de bord... (Ferme la fenêtre pour arrêter la simulation)")
ani = animation.FuncAnimation(fig, update, frames=5000, interval=20, blit=False)
plt.show()