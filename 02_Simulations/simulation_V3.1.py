import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paramètres de l'Écosystème
# -----------------------------
num_agents = 9
L = 16
steps = 10000
dt = 0.01

thermal_capacity = 25000
emergency_threshold = 0.8 * thermal_capacity

adapt_profiles = [
    {"name": "Stoïcien", "adapt": 0.0001, "color": "blue"},
    {"name": "Plastique", "adapt": 0.01, "color": "green"},
    {"name": "Équilibré", "adapt": 0.002, "color": "purple"},
] * 3

# -----------------------------
# Charge Cognitive
# -----------------------------
def cognitive_load(step):
    return 0.2 + 0.8 * np.sin(2 * np.pi * step / 4000)**2

# -----------------------------
# Initialisation des Agents
# -----------------------------
agents = []
for prof in adapt_profiles:
    agents.append({
        "Theta": 0.001 * np.random.randn(L,L),
        "kappa": 1.0,
        "heat": 0.0,
        "adapt": prof["adapt"],
        "color": prof["color"],
        "name": prof["name"],
        "kappa_history": [],
        "heat_history": [],
        "var_history": []
    })

# -----------------------------
# Boucle Temporelle
# -----------------------------
for step in range(steps):
    total_heat = sum(agent["heat"] for agent in agents)
    
    for agent in agents:
        Theta = agent["Theta"]
        kappa = agent["kappa"]
        D_base = 0.05
        
        Omega = cognitive_load(step)
        D = D_base + 0.1 * Omega
        
        lap = (np.roll(Theta,1,0)+np.roll(Theta,-1,0)+
               np.roll(Theta,1,1)+np.roll(Theta,-1,1)-4*Theta)
        dF = Theta**3 - kappa*lap - Omega
        Theta -= 1.0*dF*dt + np.sqrt(2*D*dt)*np.random.randn(L,L)
        
        heat_gen = (D + np.var(lap)) * L**2
        agent["heat"] = max(0, agent["heat"] + (heat_gen - 800)*dt)
        
        if total_heat > thermal_capacity:
            agent["heat"] *= 0.95
            kappa *= 0.98
        
        target_var = 0.2
        kappa += agent["adapt"] * (target_var - np.var(Theta))
        kappa = np.clip(kappa, 0.5, 2.5)
        
        agent["Theta"] = Theta
        agent["kappa"] = kappa
        agent["kappa_history"].append(kappa)
        agent["heat_history"].append(agent["heat"])
        agent["var_history"].append(np.var(Theta))

# -----------------------------
# Atlas de la Conscience
# -----------------------------

# 1️⃣ Carte de Chaleur Collective (heatmap des derniers Theta)
plt.figure(figsize=(12,4))
heat_matrix = np.array([agent["heat_history"][-1] for agent in agents]).reshape(3,3)
plt.imshow(heat_matrix, cmap="hot", interpolation="nearest")
plt.colorbar(label="Chaleur finale de l'agent")
plt.title("Carte de Chaleur Collective")
plt.show()

# 2️⃣ Histogramme de Fitness (variance cumulée par souche)
fitness = {}
for agent in agents:
    fitness.setdefault(agent["name"], 0)
    fitness[agent["name"]] += np.sum(agent["var_history"])
    
plt.figure(figsize=(8,5))
plt.bar(fitness.keys(), fitness.values(), color=[agent["color"] for agent in agents[:3]])
plt.title("Histogramme de Fitness Thermodynamique")
plt.ylabel("Variance cumulée (Conscience produite)")
plt.show()

# 3️⃣ Diagramme de Corrélation Inter-Agents (survariance finale)
agent_vars = np.array([agent["var_history"] for agent in agents])
corr_matrix = np.corrcoef(agent_vars)
plt.figure(figsize=(6,5))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Corrélation des Variances")
plt.title("Corrélation Inter-Agents de Conscience")
plt.show()
