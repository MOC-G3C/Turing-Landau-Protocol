import numpy as np
import copy
import json
import zlib
import math
from scipy.stats import entropy

# ============================================================
# 1. CONSTANTES PHYSIQUES (LANDAUER PROXY)
# ============================================================

np.random.seed(42)  # Reproductibilité globale

kB = 1.380649e-23
T_env = 300.0

alpha = 1e-3
Q0 = 1e-12

bits_floor = 10.0
Q_floor = bits_floor * kB * T_env * math.log(2)
Q_max = 1e-1

# Bornes méta-paramètres
META_BOUNDS = {
    "kappa_gain":       (0.01, 2.0),
    "cooling_strength": (0.30, 0.95),
    "mutation_rate":    (0.01, 0.30),
    "target_base":      (0.005, 0.15),
    "low_var_thresh":   (0.003, 0.05),   # NOUVEAU : seuil de mort évolutif
}

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
# 3. GÉNÉTIQUE M.O.C. (améliorée)
# ============================================================

def clip_meta(meta):
    """Applique les bornes sur tous les méta-paramètres."""
    for key, (lo, hi) in META_BOUNDS.items():
        if key in meta:
            meta[key] = float(np.clip(meta[key], lo, hi))
    return meta

def creer_individu(adn_base, meta_parent=None):
    theta = adn_base + np.random.randn(*adn_base.shape) * 0.05
    theta = (theta - np.mean(theta)) / (np.std(theta) + 1e-8) * 0.25

    if meta_parent is None:
        meta = {
            "kappa_gain":       0.15,
            "cooling_strength": 0.60,
            "mutation_rate":    0.06,
            "target_base":      0.04,
            "low_var_thresh":   0.012,   # NOUVEAU
        }
    else:
        meta = copy.deepcopy(meta_parent)
        # Tous les méta-paramètres mutent maintenant (pas seulement 2)
        for key in META_BOUNDS:
            meta[key] *= (1.0 + np.random.randn() * 0.06)
        clip_meta(meta)

    return {"theta": theta, "meta": meta}

def creer_individu_aleatoire(adn_shape):
    """Immigrант aléatoire : diversité injectée chaque génération."""
    theta = np.random.randn(*adn_shape) * 0.25
    meta = {
        "kappa_gain":       np.random.uniform(*META_BOUNDS["kappa_gain"]),
        "cooling_strength": np.random.uniform(*META_BOUNDS["cooling_strength"]),
        "mutation_rate":    np.random.uniform(*META_BOUNDS["mutation_rate"]),
        "target_base":      np.random.uniform(*META_BOUNDS["target_base"]),
        "low_var_thresh":   np.random.uniform(*META_BOUNDS["low_var_thresh"]),
    }
    return {"theta": theta, "meta": meta}

# ============================================================
# 4. ÉVALUATION RÉSILIENTE (améliorée)
# ============================================================

def eval_landauer_resilient(
        ind,
        k_env=8,
        steps=2000,
        low_var_grace=400):

    theta0, meta = ind["theta"].copy(), ind["meta"]
    low_var_thresh = meta.get("low_var_thresh", 0.012)  # seuil évolutif
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

            # ── COOLING ADAPTATIF (au lieu de toutes les 9 steps) ──
            var = float(np.var(theta))
            target_var = meta["target_base"] + 0.02 * abs(env_val - (np.mean(theta_means) if theta_means else 0.0))
            if var > target_var * 2.0:
                theta *= meta["cooling_strength"]
                var = float(np.var(theta))  # recalcul post-cooling

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

            kappa = np.clip(
                1.0 + meta["kappa_gain"] * (target_var - var),
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

        trials_info.append({
            "fitness":    (eta_norm, survival_frac, penalty),
            "surv_time":  survival_time,
            "q":          q_joules,
            "i":          i_useful
        })

    return {
        # fitness calculée plus bas avec poids dynamiques
        "_raw": trials_info,
        "surv_rate": float(np.mean([1.0 if tr["surv_time"] >= steps else 0.0 for tr in trials_info])),
        "q_avg":     float(np.mean([tr["q"] for tr in trials_info])),
        "i_avg":     float(np.mean([tr["i"] for tr in trials_info])),
        "avg_survtime": float(np.mean([tr["surv_time"] for tr in trials_info]))
    }

def compute_fitness(eval_result, gen, nb_gen):
    """Poids dynamiques : survie prioritaire en début, efficacité en fin."""
    gen_ratio = gen / max(nb_gen - 1, 1)
    w_surv = 0.8 - 0.4 * gen_ratio   # 0.80 → 0.40
    w_eta  = 0.2 + 0.4 * gen_ratio   # 0.20 → 0.60

    fits = []
    for tr in eval_result["_raw"]:
        eta_norm, survival_frac, penalty = tr["fitness"]
        fits.append((w_eta * eta_norm + w_surv * survival_frac) * penalty)
    return float(np.mean(fits))

# ============================================================
# 5. BOUCLE ÉVOLUTIVE (améliorée)
# ============================================================

alpha_path = "../03_Core/alpha_post_arena.npz"
data = np.load(alpha_path)
adn_base = data[data.files[0]]
adn_shape = adn_base.shape

nb_gen   = 20
pop_size = 50     # augmenté de 30 → 50
n_elites = 3
n_immigrants = 3  # individus aléatoires injectés chaque génération

population = [creer_individu(adn_base) for _ in range(pop_size)]
history = []

# Suivi pour arrêt anticipé
best_fit_history = []
PATIENCE = 5
MIN_DELTA = 0.01

print("⚡ Lancement du Protocole Landauer V8.0 (Robustesse++ / Méta-évolution complète)...")
print(f"   pop={pop_size} | élites={n_elites} | immigrants/gen={n_immigrants} | gens={nb_gen}\n")

for gen in range(nb_gen):

    # ── Évaluation ──
    evals = [eval_landauer_resilient(ind) for ind in population]

    # Calcul fitness avec poids dynamiques
    fitnesses = [compute_fitness(e, gen, nb_gen) for e in evals]
    for e, f in zip(evals, fitnesses):
        e["fitness"] = f

    idx = np.argsort(fitnesses)[::-1]
    best = evals[idx[0]]
    best_fit = best["fitness"]
    fit_std = float(np.std(fitnesses))

    print(
        f"Gen {gen:02d} | "
        f"Fit: {best_fit:.3e} ± {fit_std:.2e} | "
        f"Q: {best['q_avg']:.2e} | "
        f"I: {best['i_avg']:.2f} | "
        f"Surv%: {best['surv_rate']*100:.0f} | "
        f"avgSurv: {best['avg_survtime']:.0f}"
    )

    history.append({
        "gen":     gen,
        "fitness": fitnesses,
        "fit_mean": float(np.mean(fitnesses)),
        "fit_std":  fit_std,
        "surv":    [e["avg_survtime"] for e in evals],
        "q_avg":   [e["q_avg"] for e in evals],
    })

    # ── Arrêt anticipé ──
    best_fit_history.append(best_fit)
    if len(best_fit_history) >= PATIENCE:
        recent = best_fit_history[-PATIENCE:]
        improvement = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-12)
        if improvement < MIN_DELTA:
            print(f"\n⏹  Arrêt anticipé à la génération {gen} (stagnation détectée)")
            break

    # ── Sélection ──
    # Élites
    next_pop = [population[idx[i]] for i in range(n_elites)]

    # Immigrants aléatoires
    for _ in range(n_immigrants):
        next_pop.append(creer_individu_aleatoire(adn_shape))

    # Tournoi pour le reste
    while len(next_pop) < pop_size:
        tournament = np.random.choice(pop_size, 3, replace=False)
        p_idx = max(tournament, key=lambda i: fitnesses[i])
        next_pop.append(
            creer_individu(
                population[p_idx]["theta"],
                population[p_idx]["meta"]
            )
        )

    population = next_pop

# ── Sauvegarde ──
np.savez("final_population_v8_0.npz", population=population)

with open("meta_history_v8_0.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n✅ Simulation V8.0 terminée.")
print(f"   Fichiers sauvegardés : final_population_v8_0.npz | meta_history_v8_0.json")
