import numpy as np
import copy
import json
import zlib
import math
from scipy.stats import entropy

# ============================================================
# 1. CONSTANTES PHYSIQUES (LANDAUER PROXY)
# ============================================================

kB = 1.380649e-23
T_env = 300.0

alpha = 1e-3
Q0 = 1e-12

bits_floor = 10.0
Q_floor = bits_floor * kB * T_env * math.log(2)

Q_max = 1e-1

# ============================================================
# 2. MESURES D'INFORMATION
# ============================================================

def compressibility_bits_1d(series):
    a = series - np.min(series)
    if np.max(a) < 1e-12:
        return 0.0
    b = (a / (np.max(a) + 1e-12) * 255).astype(np.uint8).tobytes()
    c = zlib.compress(b)
    return max(0.0, (1.0 - (len(c) / len(b))) * len(b) * 8.0)

def mutual_info_time_series(x_series, y_series, bins=32):
    c_xy, _, _ = np.histogram2d(x_series, y_series, bins=bins)
    p_xy = c_xy / (np.sum(c_xy) + 1e-12)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    h_x = entropy(p_x + 1e-12)
    h_y = entropy(p_y + 1e-12)
    h_xy = entropy(p_xy.flatten() + 1e-12)
    return max(0.0, (h_x + h_y - h_xy) / math.log(2))

# ============================================================
# 3. GÉNÉTIQUE M.O.C.
# ============================================================

def creer_individu(adn_base, meta_parent=None):
    theta = adn_base + np.random.randn(*adn_base.shape) * 0.05
    theta = (theta - np.mean(theta)) / (np.std(theta) + 1e-8) * 0.25

    if meta_parent is None:
        meta = {
            "kappa_gain": 0.15,
            "cooling_strength": 0.60,
            "mutation_rate": 0.06,
            "target_base": 0.04
        }
    else:
        meta = copy.deepcopy(meta_parent)
        for key in ["kappa_gain", "cooling_strength"]:
            meta[key] *= (1.0 + np.random.randn() * 0.06)

    return {"theta": theta, "meta": meta}

# ============================================================
# 4. ÉVALUATION RÉSILIENTE
# ============================================================

def eval_landauer_resilient(
        ind,
        k_env=8,
        steps=2000,
        low_var_thresh=0.012,
        low_var_grace=400):

    theta0, meta = ind["theta"].copy(), ind["meta"]
    trials_info = []

    for k in range(k_env):
        np.random.seed(100 + k)

        theta = theta0.copy()
        heat_sim = 0.0
        theta_means = []
        env_series = []

        consecutive_low = 0
        survival_time = 0

        for t in range(1, steps + 1):

            env_val = np.sin(t / 50.0) * (
                8.0 if np.random.rand() < 0.02 else 1.0
            )

            if t % 9 == 0:
                theta *= meta["cooling_strength"]

            var = float(np.var(theta))
            heat_sim += max(0.0, (var * 15.0) - 0.3)

            theta_means.append(float(np.mean(theta)))
            env_series.append(float(env_val))

            if var < low_var_thresh:
                consecutive_low += 1
            else:
                consecutive_low = 0

            if consecutive_low >= low_var_grace:
                survival_time = t
                break

            target = meta["target_base"] + 0.02 * abs(env_val - np.mean(theta_means))
            kappa = np.clip(
                1.0 + meta["kappa_gain"] * (target - var),
                0.1,
                4.0
            )

            theta += (
                np.random.randn(*theta.shape)
                * (0.02 * kappa * meta["mutation_rate"])
                + env_val * 0.005
            )

            survival_time = t

        q_joules = alpha * heat_sim

        mi = mutual_info_time_series(
            np.array(theta_means),
            np.array(env_series)
        )

        cr = compressibility_bits_1d(np.array(theta_means))
        i_useful = 0.8 * mi + 0.2 * cr

        eta = i_useful / (q_joules + Q0)
        eta_norm = eta / (1.0 + eta)

        survival_frac = survival_time / steps

        penalty = 1.0
        if q_joules < Q_floor:
            penalty *= 0.5
        if q_joules > Q_max:
            penalty *= 0.5

        fitness = (0.4 * eta_norm + 0.6 * survival_frac) * penalty
        fitness = max(0.0, fitness)

        trials_info.append({
            "fitness": fitness,
            "surv_time": survival_time,
            "q": q_joules,
            "i": i_useful
        })

    return {
        "fitness": float(np.mean([tr["fitness"] for tr in trials_info])),
        "surv_rate": float(np.mean([1.0 if tr["surv_time"] >= steps else 0.0 for tr in trials_info])),
        "q_avg": float(np.mean([tr["q"] for tr in trials_info])),
        "i_avg": float(np.mean([tr["i"] for tr in trials_info])),
        "avg_survtime": float(np.mean([tr["surv_time"] for tr in trials_info]))
    }

# ============================================================
# 5. BOUCLE ÉVOLUTIVE
# ============================================================

alpha_path = "../03_Core/alpha_post_arena.npz"
data = np.load(alpha_path)
adn_base = data[data.files[0]]

nb_gen = 20
pop_size = 30

population = [creer_individu(adn_base) for _ in range(pop_size)]
history = []

print("⚡ Lancement du Protocole Landauer V7.4 (Robustesse Prioritaire)...")

for gen in range(nb_gen):

    evals = [eval_landauer_resilient(ind) for ind in population]
    idx = np.argsort([e["fitness"] for e in evals])[::-1]
    best = evals[idx[0]]

    print(
        f"Gen {gen:02d} | "
        f"Fit: {best['fitness']:.3e} | "
        f"Q: {best['q_avg']:.2e} | "
        f"I: {best['i_avg']:.2f} | "
        f"Surv%: {best['surv_rate']*100:.0f} | "
        f"avgSurv: {best['avg_survtime']:.0f}"
    )

    history.append({
        "fitness": [e["fitness"] for e in evals],
        "surv": [e["avg_survtime"] for e in evals]
    })

    next_pop = [
        population[idx[0]],
        population[idx[1]],
        population[idx[2]]
    ]

    while len(next_pop) < pop_size:
        tournament = np.random.choice(pop_size, 3)
        p_idx = max(tournament, key=lambda i: evals[i]["fitness"])
        next_pop.append(
            creer_individu(
                population[p_idx]["theta"],
                population[p_idx]["meta"]
            )
        )

    population = next_pop

np.savez("final_population_v7_4.npz", population=population)

with open("meta_history_v7_4.json", "w") as f:
    json.dump(history, f)

print("\n✅ Simulation terminée.")