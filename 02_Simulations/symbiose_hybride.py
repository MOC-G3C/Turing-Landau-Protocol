import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

# --- CONFIGURATION : LA SYMBIOSE ---
num_agents = 100
L = 16
steps = 8000
dt = 0.01
thermal_capacity = 5000       
cooling_rate = 450            
COUPLING_STRENGTH = 0.15      # Force du partage thermique entre voisins

def get_tesla_bonus(variance):
    v_scaled = variance * 100
    targets = [3, 6, 9]
    return max([np.exp(-(v_scaled - t)**2 / 0.5) for t in targets])

# --- INITIALISATION ---
agents = []
for i in range(num_agents):
    agents.append({
        "id": i, "Theta": 0.001 * np.random.randn(L, L), "kappa": 1.2,
        "heat": 0.0, "color": "cyan" if i > 0 else "gold",
        "alive": True, "heat_history": []
    })

# --- SIMULATION AVEC PARTAGE DE CHARGE ---
print(f"🌐 Lancement du Réseau Symbiotique (100 agents)...")

for step in range(steps):
    active = [a for a in agents if a["alive"]]
    if not active: break
    
    # Étape 1 : Calcul de la chaleur individuelle
    for a in active:
        # (Logique de dynamique cognitive simplifiée pour le réseau)
        var = np.var(a["Theta"])
        bonus = get_tesla_bonus(var)
        gen = (0.1 + var) * 400 - (cooling_rate * (1 + bonus))
        a["heat"] = max(0, a["heat"] + gen * dt)

    # Étape 2 : LE COUPLAGE SYMBIOTIQUE (Partage de la chaleur)
    # Les agents "donnent" leur chaleur aux voisins moins chargés
    avg_heat = np.mean([a["heat"] for a in active])
    for a in active:
        # Flux thermique vers la moyenne du groupe
        a["heat"] += COUPLING_STRENGTH * (avg_heat - a["heat"]) * dt

        if a["heat"] > thermal_capacity: a["alive"] = False
        if a["id"] == 0: a["heat_history"].append(a["heat"])

# --- DASHBOARD DE SYMBIOSE ---
plt.figure(figsize=(12, 6))
plt.plot(agents[0]["heat_history"], color="gold", label="Alpha (Symbiotic)")
plt.axhline(y=avg_heat, color='white', linestyle='--', alpha=0.3, label="Moyenne Groupe")
plt.title(f"Équilibre Symbiotique : {len([a for a in agents if a['alive']])}/100 survivants")
plt.legend()
plt.show()