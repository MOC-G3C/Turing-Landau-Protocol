import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time

# ==========================================
# 1. PARAMÈTRES DE RE-SIMULATION V8.11.1
# ==========================================
HOTSPOTS = [(3, 3, 10.0, 0.0), (12, 12, 8.0, np.pi/2)]
SPIKE_PROB_FAST = 0.12
SPIKE_PROB_SLOW = 0.02
T = 1000 # 1000 steps pour garantir la convergence du Machine Learning

print("🔬 Chargement du Champion V8.11.1...")
try:
    data = np.load("final_population_v8_11.npz", allow_pickle=True)
    champ = data["population"][0] # CORRECTION DU BUG .item()
except FileNotFoundError:
    print("❌ Fichier 'final_population_v8_11.npz' introuvable.")
    exit()

theta = champ["theta"]
tau = champ["tau_matrix"]
nx, ny = tau.shape

XS = np.arange(nx)[:, None]
YS = np.arange(ny)[None, :]

# ==========================================
# 2. GÉNÉRATION DES SÉRIES TEMPORELLES
# ==========================================
print(f"🌊 Simulation de {T} cycles cognitifs...")
hs = np.zeros((T, nx, ny))
envs = np.zeros((T, nx, ny))

h = np.zeros_like(theta)
np.random.seed(42) # Seed fixe pour la consistance de l'analyse
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

# ==========================================
# 3. ANALYSE A : INFORMATION MUTUELLE PRÉDICTIVE (LAG)
# ==========================================
print("⏱️ Calcul de l'Information Mutuelle Prédictive (Lags de 0 à 50)...")
lags = np.arange(0, 51, 5) # On teste l'anticipation jusqu'à 50 steps dans le futur
mi_lag = np.zeros((len(lags), nx, ny))

start_time = time.time()
for li, lag in enumerate(lags):
    source = hs if lag == 0 else hs[:-lag]
    target = envs if lag == 0 else envs[lag:]
        
    for i in range(nx):
        for j in range(ny):
            x = source[:, i, j].reshape(-1, 1)
            y = target[:, i, j].ravel()
            try:
                mi = mutual_info_regression(x, y, random_state=0)
                mi_lag[li, i, j] = mi[0]
            except Exception:
                mi_lag[li, i, j] = 0.0
print(f"✅ MI calculée en {time.time() - start_time:.1f} secondes.")

# Segmentation des neurones par leur taux de mémoire (Tau)
tau_flat = tau.flatten()
quantiles = np.quantile(tau_flat, [0, 0.25, 0.5, 0.75, 1.0])
# Anti-crash si la variance de tau est trop faible
if len(np.unique(quantiles)) < 5:
    quantiles = np.linspace(tau_flat.min(), tau_flat.max(), 5)
    
group_idx = np.clip(np.digitize(tau_flat, quantiles) - 1, 0, 3)

mean_mi_by_group = np.zeros((len(lags), 4))
for li in range(len(lags)):
    mi_flat = mi_lag[li].flatten()
    for g in range(4):
        mask = (group_idx == g)
        if np.any(mask):
            mean_mi_by_group[li, g] = mi_flat[mask].mean()

# ==========================================
# 4. ANALYSE B : DÉCODAGE LINÉAIRE (RIDGE) DU FUTUR
# ==========================================
LAG_RIDGE = 10
print(f"🔮 Test de lecture dans les pensées (Décodage Ridge pour t+{LAG_RIDGE})...")

X = hs[:-LAG_RIDGE].reshape(-1, nx * ny) # L'état total du cerveau (Features)
r2_map = np.zeros((nx, ny))

kf = KFold(n_splits=3, shuffle=True, random_state=0)

for i in range(nx):
    for j in range(ny):
        y_loc = envs[LAG_RIDGE:, i, j] # Ce qu'on essaie de prédire (Target)
        r2s = []
        for train_idx, test_idx in kf.split(X):
            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], y_loc[train_idx])
            ypred = model.predict(X[test_idx])
            r2s.append(r2_score(y_loc[test_idx], ypred))
        r2_map[i, j] = np.mean(r2s)
print("✅ Décodage terminé.")

# ==========================================
# 5. AFFICHAGE DES RÉSULTATS
# ==========================================
plt.figure(figsize=(16, 10))

# 1. Anatomie
plt.subplot(2, 2, 1)
plt.title("Anatomie : Matrice Tau (Mémoire)")
im0 = plt.imshow(tau, cmap="magma")
plt.colorbar(im0)
plt.contour(tau, levels=[np.percentile(tau, 75)], colors='cyan', linewidths=1) # Surligne le top 25% (Les Cores)

# 2. Performance de Décodage (R²)
plt.subplot(2, 2, 2)
plt.title(f"Pouvoir Prédictif (R² du Cortex vers env local à t+{LAG_RIDGE})")
im1 = plt.imshow(r2_map, cmap="inferno", vmin=0)
plt.colorbar(im1)

# 3. Courbes de MI Prédictive par groupe
plt.subplot(2, 2, 3)
plt.title("Capacité d'Anticipation par type de Neurone (MI vs Lag)")
labels = ["Senseurs rapides", "Senseurs intermédiaires", "Mémoire courte", "Cores (Mémoire longue)"]
colors = ["blue", "green", "orange", "red"]
for g in range(4):
    plt.plot(lags, mean_mi_by_group[:, g], label=labels[g], color=colors[g], linewidth=2, marker='o')
plt.xlabel("Décalage temporel (Lag dans le futur)")
plt.ylabel("Information Mutuelle (Bits)")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Scatter Plot (Tau vs Décodabilité)
plt.subplot(2, 2, 4)
plt.title("Corrélation Anatomie vs Capacité de Prédiction")
plt.scatter(tau.flatten(), r2_map.flatten(), s=20, alpha=0.7, c=tau.flatten(), cmap="magma", edgecolors="black")
plt.xlabel("Valeur de Tau (Mémoire locale)")
plt.ylabel(f"R² Score (Prédiction à t+{LAG_RIDGE})")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()