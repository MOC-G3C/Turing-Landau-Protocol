# analyze_spatial_mix.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Chargement V8.13
data = np.load("final_population_v8_13.npz", allow_pickle=True)
champ = data["population"][0]

w = champ["w_fast_mat"]
tau_f = champ["tau_fast"]
tau_s = champ["tau_slow"]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Mix Spatial (w_fast)\nJaune = Réflexe | Violet = Mémoire")
plt.imshow(w, cmap="viridis")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Canal FAST (tau)")
plt.imshow(tau_f, cmap="cool")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Canal SLOW (tau)")
plt.imshow(tau_s, cmap="hot")
plt.colorbar()

plt.tight_layout()
plt.show()

# Calcul de corrélation
wf_flat, ts_flat = w.flatten(), tau_s.flatten()
rho, p = spearmanr(wf_flat, ts_flat)
print(f"\n📊 STATS : Spearman(w_fast, tau_slow) = {rho:.3f} (p={p:.2e})")