import json, numpy as np, matplotlib.pyplot as plt

with open("meta_history_v6.json", "r") as f: histoire = json.load(f)

# Extraction
fit_means = [np.mean(g["fitness"]) for g in histoire]
surv_means = [np.mean(g["surv_frac"]) for g in histoire]
kappa_vars = [np.var(g["kappas"]) for g in histoire]

# 🔬 TEST DE TENDANCE (Simple Mann-Kendall)
def trend_test(data):
    n = len(data)
    s = sum(np.sign(data[j] - data[i]) for i in range(n) for j in range(i + 1, n))
    return "Baisse" if s < 0 else "Hausse", s

t_kappa, s_kappa = trend_test(kappa_vars)

# --- VISUALISATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(fit_means, label="Fitness", color="blue")
ax1.plot(surv_means, label="Survie", color="red")
ax1.set_title("Évolution de la Résilience")
ax1.legend()

ax2.plot(kappa_vars, color="purple")
ax2.set_title(f"Convergence Gène Kappa ({t_kappa})")
plt.tight_layout()
plt.show()

print(f"Rapport : Tendance Variance Kappa = {t_kappa} (S-score: {s_kappa})")