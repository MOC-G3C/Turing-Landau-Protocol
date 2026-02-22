import json, numpy as np, matplotlib.pyplot as plt

with open("meta_history_v7.json", "r") as f: h = json.load(f)

# Extraction des courbes
fit_avg = [np.mean(g["fitness"]) for g in h]
kappa_var = [np.var(g["kappas"]) for g in h]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(fit_avg, color='green', label='Efficience η (Landauer)')
ax1.set_ylabel('Information / Joules')
ax1.set_xlabel('Générations')

ax2 = ax1.twinx()
ax2.plot(kappa_var, color='purple', linestyle='--', label='Convergence (Variance)')
ax2.set_ylabel('Variance Génétique')

plt.title("Alignement Causal : Efficience vs Dissipation")
fig.tight_layout()
plt.show()