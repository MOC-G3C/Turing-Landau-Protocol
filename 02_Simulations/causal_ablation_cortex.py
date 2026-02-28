import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import scipy.stats as st
import time

# ==========================================
# 1. PARAMÈTRES ET CHARGEMENT
# ==========================================
HOTSPOTS = [(3, 3, 10.0, 0.0), (12, 12, 8.0, np.pi/2)]
SPIKE_PROB_FAST = 0.12
SPIKE_PROB_SLOW = 0.02
T = 1000
LAG_RIDGE = 10

print("🔬 Chargement du Champion V8.11.1...")
data = np.load("final_population_v8_11.npz", allow_pickle=True)
champ = data["population"][0]
theta_base = champ["theta"]
tau_base = champ["tau_matrix"]
nx, ny = tau_base.shape
XS = np.arange(nx)[:, None]
YS = np.arange(ny)[None, :]

# ==========================================
# 2. FONCTION DE SIMULATION ET DÉCODAGE
# ==========================================
def simulate_and_decode(tau_matrix, description=""):
    print(f"\n🌊 Simulation et Décodage en cours : {description}")
    hs = np.zeros((T, nx, ny))
    envs = np.zeros((T, nx, ny))
    h = np.zeros_like(theta_base)
    
    np.random.seed(42) # Seed fixe pour comparaison stricte
    for t in range(1, T + 1):
        env_field = np.zeros_like(theta_base)
        fast_comp = (1.0 if np.random.rand() < SPIKE_PROB_FAST else 0.1) * np.sin(t / 8.0)
        slow_comp = (1.0 if np.random.rand() < SPIKE_PROB_SLOW else 0.1) * np.sin(t / 200.0)
        for (x, y, amp, phase) in HOTSPOTS:
            dist2 = (XS - x)**2 + (YS - y)**2
            env_field += amp * np.exp(-dist2 / 9.0) * (fast_comp + slow_comp + 0.2 * np.sin(t/10.0 + phase))
        env_field += 0.3 * np.sin(t/100.0)
        
        h = tau_matrix * h + (1.0 - tau_matrix) * env_field
        hs[t-1] = h
        envs[t-1] = env_field

    X = hs[:-LAG_RIDGE].reshape(-1, nx * ny)
    r2_map = np.zeros((nx, ny))
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    
    # Pour inspecter les poids Ridge sur le Hotspot Nord-Ouest (3,3)
    target_weights = None 
    
    for i in range(nx):
        for j in range(ny):
            y_loc = envs[LAG_RIDGE:, i, j]
            r2s = []
            weights_folds = []
            for train_idx, test_idx in kf.split(X):
                model = Ridge(alpha=1.0)
                model.fit(X[train_idx], y_loc[train_idx])
                ypred = model.predict(X[test_idx])
                r2s.append(r2_score(y_loc[test_idx], ypred))
                if i == 3 and j == 3: # Sauvegarde des poids pour le Hotspot
                    weights_folds.append(model.coef_)
            r2_map[i, j] = np.mean(r2s)
            
            if i == 3 and j == 3:
                target_weights = np.mean(weights_folds, axis=0).reshape(nx, ny)
                
    return r2_map, target_weights

# ==========================================
# 3. EXÉCUTION : BASELINE vs ABLATION
# ==========================================
# A. Baseline
r2_baseline, weights_nw = simulate_and_decode(tau_base, "Baseline (Cortex Intact)")

# B. Statistiques (Spearman & Bootstrap)
print("\n📊 STATISTIQUES (Baseline) :")
tau_flat = tau_base.flatten()
r2_flat = r2_baseline.flatten()
rho, pval = st.spearmanr(tau_flat, r2_flat)
print(f"Spearman rho = {rho:.3f}, p = {pval:.2e}")

def boot_rho(tau_f, r2_f, n=2000):
    idx = np.arange(len(tau_f))
    boots = []
    for _ in range(n):
        s = np.random.choice(idx, size=len(idx), replace=True)
        boots.append(st.spearmanr(tau_f[s], r2_f[s]).correlation)
    return np.percentile(boots, [2.5, 97.5])

lo, hi = boot_rho(tau_flat, r2_flat)
print(f"95% CI rho = [{lo:.3f}, {hi:.3f}]")

# C. Ablation (La Lobotomie)
tau_ablate = tau_base.copy()
core_threshold = np.percentile(tau_ablate, 75)
core_mask = tau_ablate >= core_threshold
tau_ablate[core_mask] = 0.2 # Forcer l'oubli rapide sur les Cores

r2_ablated, _ = simulate_and_decode(tau_ablate, "Ablation (Cores détruits -> Tau=0.2)")

# Comparaison Causale
delta_map = r2_baseline - r2_ablated
delta_mean_global = np.mean(delta_map)
delta_mean_cores = np.mean(delta_map[core_mask])

print("\n🔪 RÉSULTAT DE L'ABLATION CAUSALE :")
print(f"Chute moyenne du R² (Global) : {delta_mean_global:.4f}")
print(f"Chute moyenne du R² (Dans les ex-Cores) : {delta_mean_cores:.4f}")

# ==========================================
# 4. VISUALISATION
# ==========================================
plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.title("R² Baseline (Cortex Intact)")
plt.imshow(r2_baseline, cmap="inferno", vmin=0)
plt.colorbar(fraction=0.046, pad=0.04)
plt.contour(tau_base, levels=[core_threshold], colors='cyan', linewidths=1)

plt.subplot(1, 4, 2)
plt.title("R² après Ablation des Cores")
plt.imshow(r2_ablated, cmap="inferno", vmin=0)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 4, 3)
plt.title("Delta R² (Perte de prédiction)")
plt.imshow(delta_map, cmap="Reds")
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 4, 4)
plt.title("Poids Ridge pour prédire le Hotspot(3,3)")
plt.imshow(np.abs(weights_nw), cmap="viridis")
plt.colorbar(fraction=0.046, pad=0.04)
plt.contour(tau_base, levels=[core_threshold], colors='red', linewidths=0.5, linestyles='dashed')

plt.tight_layout()
plt.show()