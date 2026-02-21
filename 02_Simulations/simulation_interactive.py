import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# Paramètres Initiaux
# -----------------------------
num_agents = 9
L = 16
steps = 2000
dt = 0.01

# Profils de population
adapt_profiles = [
    {"name": "Stoïcien", "adapt": 0.0001, "color": "blue"},
    {"name": "Plastique", "adapt": 0.01, "color": "green"},
    {"name": "Équilibré", "adapt": 0.002, "color": "purple"},
] * 3

# -----------------------------
# Initialisation des Agents
# -----------------------------
def init_agents(adapt_mod=1.0):
    agents = []
    for prof in adapt_profiles:
        agents.append({
            "Theta": 0.001 * np.random.randn(L,L),
            "kappa": 1.0,
            "heat": 0.0,
            "adapt": prof["adapt"]*adapt_mod,
            "color": prof["color"],
            "name": prof["name"],
            "kappa_history": [],
            "heat_history": [],
            "var_history": []
        })
    return agents

agents = init_agents()

# -----------------------------
# Fonction de Charge Cognitive
# -----------------------------
def cognitive_load(step, amp=0.8):
    return 0.2 + amp * np.sin(2 * np.pi * step / 400.0)**2

# -----------------------------
# Simulation Interactive
# -----------------------------
def run_simulation(cog_amp=0.8, thermal_capacity=25000, adapt_mod=1.0):
    agents = init_agents(adapt_mod)
    for step in range(steps):
        total_heat = sum(agent["heat"] for agent in agents)
        for agent in agents:
            Theta = agent["Theta"]
            kappa = agent["kappa"]
            Omega = cognitive_load(step, amp=cog_amp)
            D = 0.05 + 0.1 * Omega
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
    return agents

# -----------------------------
# Visualisation Initiale
# -----------------------------
agents_sim = run_simulation()

fig, axs = plt.subplots(2,2, figsize=(12,8))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Heatmap
heat_ax = axs[0,0]
heat_matrix = np.array([agent["heat_history"][-1] for agent in agents_sim]).reshape(3,3)
hm = heat_ax.imshow(heat_matrix, cmap="hot", interpolation="nearest")
plt.colorbar(hm, ax=heat_ax)
heat_ax.set_title("Carte de Chaleur Collective")

# Histogramme de Fitness
fitness_ax = axs[0,1]
fitness = {}
for agent in agents_sim:
    fitness.setdefault(agent["name"],0)
    fitness[agent["name"]] += np.sum(agent["var_history"])
bars = fitness_ax.bar(fitness.keys(), fitness.values(), color=[agent["color"] for agent in agents_sim[:3]])
fitness_ax.set_title("Histogramme de Fitness Thermodynamique")

# Corrélation Inter-Agents
corr_ax = axs[1,0]
agent_vars = np.array([agent["var_history"] for agent in agents_sim])
corr_matrix = np.corrcoef(agent_vars)
corr_im = corr_ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(corr_im, ax=corr_ax)
corr_ax.set_title("Corrélation Inter-Agents")

# Kappa Evolution
kappa_ax = axs[1,1]
for agent in agents_sim:
    kappa_ax.plot(agent["kappa_history"], color=agent["color"], alpha=0.5)
kappa_ax.set_title("Évolution de Kappa")
plt.show()

# -----------------------------
# Sliders
# -----------------------------
ax_cog = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_therm = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_adapt = plt.axes([0.15, 0.05, 0.65, 0.03])

slider_cog = Slider(ax_cog, 'Charge Cognitive', 0.1, 1.5, valinit=0.8)
slider_therm = Slider(ax_therm, 'Capacité Thermique', 10000, 50000, valinit=25000)
slider_adapt = Slider(ax_adapt, 'Vitesse Adaptation', 0.0001, 0.02, valinit=1.0)

def update(val):
    agents_sim = run_simulation(slider_cog.val, slider_therm.val, slider_adapt.val)
    
    # Heatmap
    heat_matrix = np.array([agent["heat_history"][-1] for agent in agents_sim]).reshape(3,3)
    hm.set_data(heat_matrix)
    
    # Histogramme
    fitness = {}
    for agent in agents_sim:
        fitness.setdefault(agent["name"],0)
        fitness[agent["name"]] += np.sum(agent["var_history"])
    for bar, key in zip(bars, fitness.keys()):
        bar.set_height(fitness[key])
    
    # Corrélation
    agent_vars = np.array([agent["var_history"] for agent in agents_sim])
    corr_matrix = np.corrcoef(agent_vars)
    corr_im.set_data(corr_matrix)
    
    # Kappa Evolution
    kappa_ax.cla()
    for agent in agents_sim:
        kappa_ax.plot(agent["kappa_history"], color=agent["color"], alpha=0.5)
    kappa_ax.set_title("Évolution de Kappa")
    
    fig.canvas.draw_idle()

slider_cog.on_changed(update)
slider_therm.on_changed(update)
slider_adapt.on_changed(update)

plt.show()
