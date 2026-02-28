import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# --- 1. Chargement du Champion ---
data = np.load("final_population_v8_11.npz", allow_pickle=True)
champ = data["population"][0]

theta = champ["theta"]
tau = champ["tau_matrix"]
nx, ny = tau.shape

# --- 2. Recréation de l'Environnement V8.11.1 ---
HOTSPOTS = [(3, 3, 10.0, 0.0), (12, 12, 8.0, np.pi/2)]
XS = np.arange(nx)[:, None]
YS = np.arange(ny)[None, :]
SPIKE_PROB_FAST = 0.12
SPIKE_PROB_SLOW = 0.02

T = 800
hs = np.zeros((T, nx, ny))
envs = np.zeros_like(hs)

h = np.zeros_like(theta)
print("🧠 Simulation de 800 cycles pour extraire l'Information Mutuelle...")
for t in range(1, T + 1):
    env_field = np.zeros_like(theta)
    fast_comp = (1.0 if np.random.rand() < SPIKE_PROB_FAST else 0.1) * np.sin(t / 8.0)
    slow_comp = (1.0 if np.random.rand() < SPIKE_PROB_SLOW else 0.1) * np.sin(t / 200.0)
    for (x, y, amp, phase) in HOTSPOTS:
        dist2 = (XS - x)**2 + (YS - y)**2
        env_field += amp * np.exp(-dist2 / 9.0) * (fast_comp + slow_comp + 0.2 * np.sin(t/10.0 + phase))
    env_field += 0.3 * np.sin(t/100.0)
    
    h = tau * h + (1.0 - tau) * env_field
    hs[t-1] = h
    envs[t-1] = env_field

# --- 3. Calcul de l'Information Mutuelle par Neurone ---
mi_map = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        x = hs[:, i, j].reshape(-1, 1)
        y = envs[:, i, j].ravel()
        try:
            # Calcul de la dépendance entre la mémoire locale (h) et l'environnement local (y)
            mi = mutual_info_regression(x, y, random_state=0)
            mi_map[i, j] = mi[0]
        except Exception:
            mi_map[i, j] = 0.0

# --- 4. Affichage ---
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.title("Anatomie : Matrice Tau (Mémoire)")
im0 = plt.imshow(tau, cmap="magma")
plt.colorbar(im0)

plt.subplot(1, 3, 2)
plt.title("Fonction : Information Mutuelle (MI)")
im1 = plt.imshow(mi_map, cmap="viridis")
plt.colorbar(im1)

plt.subplot(1, 3, 3)
plt.title("Corrélation Tau vs MI")
plt.scatter(tau.flatten(), mi_map.flatten(), s=15, alpha=0.7, c="cyan", edgecolors="black")
plt.xlabel("Valeur de Tau (Mémoire)")
plt.ylabel("Information Mutuelle (MI)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()