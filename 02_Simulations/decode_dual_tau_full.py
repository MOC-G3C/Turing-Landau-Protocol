# decode_dual_tau_full.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# --- Configuration identique à la simulation ---
HOTSPOTS = [(3, 3, 10.0, 0.0), (12, 12, 8.0, np.pi/2)]
XS, YS = np.arange(16)[:, None], np.arange(16)[None, :]
T = 1000

print("🔮 Régénération du flux cognitif pour analyse prédictive...")
data = np.load("final_population_v8_13.npz", allow_pickle=True)
champ = data["population"][0]

h_f, h_s = np.zeros((16,16)), np.zeros((16,16))
theta = champ["theta"].copy()
hs_comb = np.zeros((T, 16, 16))
envs = np.zeros((T, 16, 16))

# Simulation
for t in range(T):
    env_field = np.zeros((16,16))
    f_comp = (1.0 if np.random.rand() < 0.12 else 0.1) * np.sin(t / 8.0)
    s_comp = (1.0 if np.random.rand() < 0.02 else 0.1) * np.sin(t / 200.0)
    for (x, y, amp, ph) in HOTSPOTS:
        dist2 = (XS - x)**2 + (YS - y)**2
        env_field += amp * np.exp(-dist2 / 9.0) * (f_comp + s_comp)
    
    h_f = champ["tau_fast"] * h_f + (1.0 - champ["tau_fast"]) * env_field
    h_s = champ["tau_slow"] * h_s + (1.0 - champ["tau_slow"]) * env_field
    h_comb = (champ["w_fast_mat"] * h_f) + ((1.0 - champ["w_fast_mat"]) * h_s)
    
    hs_comb[t] = h_comb
    envs[t] = env_field

# Test de prédiction sur 3 lags
lags = [5, 10, 25]
plt.figure(figsize=(15, 5))

for idx, lag in enumerate(lags):
    X = hs_comb[:-lag].reshape(-1, 256)
    Y = envs[lag:].reshape(-1, 256)
    
    # On entraîne un décodeur Ridge sur tout le cerveau
    model = Ridge(alpha=1.0)
    model.fit(X[:800], Y[:800])
    Y_pred = model.predict(X[800:])
    
    r2_per_pixel = r2_score(Y[800:], Y_pred, multioutput='raw_values').reshape(16, 16)
    
    plt.subplot(1, 3, idx+1)
    plt.title(f"Pouvoir Prédictif à t+{lag}\n(R² moyen: {np.mean(r2_per_pixel):.3f})")
    plt.imshow(r2_per_pixel, cmap="plasma", vmin=0, vmax=1)
    plt.colorbar()

plt.tight_layout()
plt.show()