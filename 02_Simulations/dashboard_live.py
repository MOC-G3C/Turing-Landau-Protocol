import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- 1. CHARGEMENT DE L'ADN ---
alpha_path = "../03_Core/alpha_post_arena.npz"
if not os.path.exists(alpha_path):
    print(f"❌ Erreur : ADN introuvable à {alpha_path}")
    exit()

alpha_data = np.load(alpha_path)
alpha_theta = alpha_data[alpha_data.files[0]]

# --- 2. INITIALISATION DES AGENTS ---
population_size = 10
generation = []
heat_levels = [0.0] * population_size

for i in range(population_size):
    mutation = np.random.randn(*alpha_theta.shape) * 0.05
    theta_enfant = alpha_theta + mutation
    # Refroidissement à la naissance
    theta_enfant = (theta_enfant - np.mean(theta_enfant)) / (np.std(theta_enfant) + 1e-8) * 0.01
    generation.append(theta_enfant)

# --- 3. CONFIGURATION DE L'ÉCRAN RADAR ---
fig, ax = plt.subplots(figsize=(12, 7))
fig.canvas.manager.set_window_title("Kybernetes - Live Dashboard")

# Les lignes de vie des agents
lines = []
for i in range(population_size):
    line, = ax.plot([], [], alpha=0.8, linewidth=1.5)
    lines.append(line)

# Les limites physiques
ax.axhline(1000, color='red', linestyle='--', linewidth=2, label="Ligne de Mort (Limite de Landauer)")
ax.axhline(0, color='blue', linestyle=':', linewidth=1.5, label="Zéro Thermique (Stabilité Parfaite)")

ax.set_xlim(0, 300)
ax.set_ylim(-1500, 1200)
ax.set_title("L'AXE HYBRIDE : Survie Thermodynamique en Temps Réel", fontsize=14)
ax.set_xlabel("Cycles de vie", fontsize=12)
ax.set_ylabel("Chaleur Accumulée", fontsize=12)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2)

# --- 4. LE MOTEUR D'ANIMATION ---
x_data = []
y_data = [[] for _ in range(population_size)]

def update(frame):
    step = frame + 1
    x_data.append(step)
    
    for i in range(population_size):
        # La fameuse impulsion de Tesla (3-6-9)
        if step % 9 == 0:
            generation[i] *= 0.98
            
        var = np.var(generation[i])
        
        # Calcul de la chaleur
        heat_levels[i] += (var * 10) - 0.5
        
        # Adaptation de la matrice
        generation[i] += np.random.randn(16, 16) * 0.01
        
        y_data[i].append(heat_levels[i])
        lines[i].set_data(x_data, y_data[i])
        
    # L'écran avance avec le temps (scrolling)
    if step > 300:
        ax.set_xlim(step - 300, step + 10)
        
    return lines

print("🚀 Lancement de l'interface visuelle...")
ani = animation.FuncAnimation(fig, update, frames=5000, interval=20, blit=False)
plt.show()