# analysis_predictive_mi.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

data = np.load("final_population_v8_11.npz", allow_pickle=True)
champ = data["population"][0].item()
tau = champ["tau_matrix"]
# re-generate hs and envs like in your plot script (T long)
# --- reuse your env generation code, produce hs (T,nx,ny) and envs (T,nx,ny) ---
# ... (copy the env/h generation exactly as in plot_champion_mi.py) ...
# assume hs, envs are available here

lags = np.arange(0, 101, 5)   # test 0..100 timesteps ahead
nx, ny = tau.shape
mi_lag = np.zeros((len(lags), nx, ny))

for li, lag in enumerate(lags):
    # target is env shifted backward: env[t+lag] predicted by h[t]
    if lag == 0:
        target = envs
        source = hs
    else:
        target = envs[lag:]
        source = hs[:-lag]
    T_eff = source.shape[0]
    for i in range(nx):
        for j in range(ny):
            x = source[:, i, j].reshape(-1,1)
            y = target[:, i, j].ravel()
            try:
                mi = mutual_info_regression(x, y, random_state=0)
                mi_lag[li, i, j] = mi[0]
            except:
                mi_lag[li, i, j] = 0.0

# aggregate by tau quantiles for plotting
tau_flat = tau.flatten()
bins = np.quantile(tau_flat, [0,0.25,0.5,0.75,1.0])
group_idx = np.digitize(tau_flat, bins) - 1

mean_mi_by_group = []
for li in range(len(lags)):
    mi_flat = mi_lag[li].flatten()
    mean_by_group = [mi_flat[group_idx == g].mean() for g in range(4)]
    mean_mi_by_group.append(mean_by_group)

mean_mi_by_group = np.array(mean_mi_by_group)  # shape (n_lags, 4)
plt.figure(figsize=(6,4))
for g in range(4):
    plt.plot(lags, mean_mi_by_group[:,g], label=f"tau-q{g}")
plt.xlabel("Lag (timesteps ahead)")
plt.ylabel("MI (h(t) -> env(t+lag))")
plt.legend()
plt.grid(True)
plt.show()