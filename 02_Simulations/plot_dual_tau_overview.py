# plot_dual_tau_overview.py
import numpy as np
import matplotlib.pyplot as plt

# Chargement du champion V8.12
try:
    data = np.load("final_population_v8_12.npz", allow_pickle=True)
    pop = data["population"]
    champ = pop[0] # Le champion est le premier
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")
    exit()

tau_f = champ["tau_fast"]
tau_s = champ["tau_slow"]
w_f = champ["meta"].get("w_fast", 0.5)
w_s = 1.0 - w_f

plt.figure(figsize=(15, 5))

# Canal Rapide
plt.subplot(1, 3, 1)
plt.title(f"Canal FAST (tau: 0.15-0.40)\nPoids global : {w_f:.2f}")
im1 = plt.imshow(tau_f, cmap="cool")
plt.colorbar(im1)

# Canal Lent
plt.subplot(1, 3, 2)
plt.title(f"Canal SLOW (tau: 0.70-0.98)\nPoids global : {w_s:.2f}")
im2 = plt.imshow(tau_s, cmap="hot")
plt.colorbar(im2)

# Analyse de la dispersion
plt.subplot(1, 3, 3)
plt.hist(tau_f.flatten(), bins=15, alpha=0.5, label="Fast", color="cyan")
plt.hist(tau_s.flatten(), bins=15, alpha=0.5, label="Slow", color="red")
plt.title("Distribution des échelles de temps")
plt.legend()

plt.tight_layout()
plt.show()