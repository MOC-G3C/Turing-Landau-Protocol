import numpy as np
import copy
import json
import zlib
import math
from scipy.stats import entropy

# ============================================================
# 1. CONSTANTES PHYSIQUES
# ============================================================

np.random.seed(42)

kB    = 1.380649e-23
T_env = 300.0
alpha = 1e-3
Q0    = 1e-3

bits_floor = 10.0
Q_floor    = bits_floor * kB * T_env * math.log(2)

Q_soft_min = 1e-4
Q_soft_max = 1e-1

META_BOUNDS = {
    "kappa_gain":       (0.01, 2.0),
    "cooling_strength": (0.30, 0.95),
    "mutation_rate":    (0.01, 0.30),
    "target_base":      (0.005, 0.15),
    "low_var_thresh":   (0.003, 0.05),
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
    p_x  = np.sum(p_xy, axis=1)
    p_y  = np.sum(p_xy, axis=0)
    h_x  = entropy(p_x + 1e-12)
    h_y  = entropy(p_y + 1e-12)
    h_xy = entropy(p_xy.flatten() + 1e-12)
    return max(0.0, (h_x + h_y - h_xy) / math.log(2))

# ============================================================
# 3. OPTION B — RÉPONSE DIFFÉRENTIELLE AUX SPIKES
# ============================================================

def differential_spike_response(theta_means, env_series):
    """
    Mesure la capacité d'adaptation réelle :
    compare l'information produite PENDANT les spikes vs hors-spikes.

    Un bon individu doit répondre PLUS aux perturbations fortes
    qu'aux phases calmes. Un survivant passif aura un ratio ≈ 1.

    Retourne un score dans [0, 1] :
      - 0.0 : aucune différenciation (comportement identique spike/calme)
      - 1.0 : forte réponse adaptative aux spikes
    """
    theta_arr = np.array(theta_means)
    env_arr   = np.array(env_series)

    spike_mask = np.abs(env_arr) > 4.0   # spikes = |env| > 4 (amplitude ×8 × sin)
    calm_mask  = ~spike_mask

    if spike_mask.sum() < 5 or calm_mask.sum() < 5:
        return 0.0

    # Variance de theta pendant spikes vs calme
    var_spike = float(np.var(theta_arr[spike_mask]))
    var_calm  = float(np.var(theta_arr[calm_mask]))

    # Ratio de réponse : on veut var_spike > var_calm (réactivité)
    # mais pas une explosion (ratio plafonné)
    if var_calm < 1e-10:
        return 0.0

    ratio = var_spike / (var_calm + 1e-10)
    # Normalisation : ratio=1 → 0.0 (pas de différenciation)
    #                 ratio=5 → 0.67, ratio=10 → 0.82 (bonne adaptation)
    score = math.tanh(max(0.0, ratio - 1.0) / 4.0)
    return float(score)

# ============================================================
# 4. PÉNALITÉ Q CONTINUE
# ============================================================

def q_penalty_continuous(q_joules):
    if q_joules <= 0.0:
        return 0.0
    if Q_soft_min <= q_joules <= Q_soft_max:
        return 1.0
    if q_joules < Q_soft_min:
        ratio = math.log(q_joules / Q_soft_min) / math.log(Q_floor / Q_soft_min + 1e-30)
        return float(np.clip(1.0 + ratio * 0.8, 0.05, 1.0))
    ratio = math.log(q_joules / Q_soft_max) / math.log(10.0)
    return float(np.clip(1.0 - ratio * 0.7, 0.05, 1.0))

# ============================================================
# 5. NORMALISATION Z-SCORE INTER-POPULATION (OPTION A)
# ============================================================

def normalize_population_scores(raw_scores_list):
    """
    Normalise une liste de valeurs brutes par z-score,
    puis remappe dans [0, 1] via sigmoid.
    Évite qu'une métrique à grande échelle (I~900) domine
    une métrique à petite échelle (Q~0.02).
    """
    arr = np.array(raw_scores_list, dtype=float)
    mu  = np.mean(arr)
    std = np.std(arr) + 1e-10
    z   = (arr - mu) / std
    # Sigmoid pour rester dans [0,1]
    return 1.0 / (1.0 + np.exp(-z))

# ============================================================
# 6. GÉNÉTIQUE M.O.C.
# ============================================================

def clip_meta(meta):
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
            "low_var_thresh":   0.012,
        }
    else:
        meta = copy.deepcopy(meta_parent)
        for key in META_BOUNDS:
            meta[key] *= (1.0 + np.random.randn() * 0.06)
        clip_meta(meta)

    return {"theta": theta, "meta": meta}

def creer_individu_aleatoire(adn_shape):
    theta = np.random.randn(*adn_shape) * 0.25
    meta  = {k: np.random.uniform(lo, hi) for k, (lo, hi) in META_BOUNDS.items()}
    return {"theta": theta, "meta": meta}

# ============================================================
# 7. ÉVALUATION — retourne les métriques brutes (pas de fitness ici)
# ============================================================

def eval_individu_raw(ind, k_env=8, steps=2000, low_var_grace=400):
    """
    Retourne les métriques BRUTES pour normalisation post-hoc :
      - i_useful   : information produite (bits)
      - q_joules   : énergie dissipée (J proxy)
      - spike_resp : score de réponse différentielle aux spikes [0,1]
      - surv_frac  : fraction de survie [0,1]
      - q_penalty  : pénalité énergétique continue [0,1]
    """
    theta0, meta   = ind["theta"].copy(), ind["meta"]
    low_var_thresh = meta.get("low_var_thresh", 0.012)
    trials         = []

    for k in range(k_env):
        np.random.seed(100 + k)

        theta       = theta0.copy()
        heat_sim    = 0.0
        theta_means = []
        env_series  = []

        consecutive_low = 0
        survival_time   = 0

        for t in range(1, steps + 1):

            env_val     = np.sin(t / 50.0) * (8.0 if np.random.rand() < 0.02 else 1.0)
            var         = float(np.var(theta))
            mean_so_far = float(np.mean(theta_means)) if theta_means else 0.0
            target_var  = meta["target_base"] + 0.02 * abs(env_val - mean_so_far)

            # Cooling adaptatif
            if var > target_var * 2.0:
                theta *= meta["cooling_strength"]
                var    = float(np.var(theta))

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

            kappa = np.clip(1.0 + meta["kappa_gain"] * (target_var - var), 0.1, 4.0)
            theta += (
                np.random.randn(*theta.shape) * (0.02 * kappa * meta["mutation_rate"])
                + env_val * 0.005
            )
            survival_time = t

        q_joules  = alpha * heat_sim
        mi        = mutual_info_time_series(np.array(theta_means), np.array(env_series))
        cr        = compressibility_bits_1d(np.array(theta_means))
        i_useful  = 0.8 * mi + 0.2 * cr
        spike_rsp = differential_spike_response(theta_means, env_series)

        trials.append({
            "i":          i_useful,
            "q":          q_joules,
            "spike_resp": spike_rsp,
            "surv_frac":  survival_time / steps,
            "q_penalty":  q_penalty_continuous(q_joules),
        })

    return {
        "i_avg":        float(np.mean([t["i"]          for t in trials])),
        "q_avg":        float(np.mean([t["q"]          for t in trials])),
        "spike_avg":    float(np.mean([t["spike_resp"] for t in trials])),
        "surv_avg":     float(np.mean([t["surv_frac"]  for t in trials])),
        "q_pen_avg":    float(np.mean([t["q_penalty"]  for t in trials])),
        "surv_rate":    float(np.mean([1.0 if t["surv_frac"] >= 1.0 else 0.0 for t in trials])),
        "avg_survtime": float(np.mean([t["surv_frac"] * steps for t in trials])),
    }

# ============================================================
# 8. CALCUL DE FITNESS AVEC NORMALISATION Z-SCORE (A+B)
# ============================================================

def compute_population_fitness(evals, gen, nb_gen):
    """
    Option A : normalise I, Q_efficiency (1/Q), spike_resp par z-score + sigmoid.
    Option B : spike_resp est déjà dans [0,1], mais normalisé aussi pour équité.

    Poids dynamiques :
      - survie forte en début, efficacité + adaptation en fin
    """
    gen_ratio = gen / max(nb_gen - 1, 1)

    # Poids dynamiques
    w_surv  = 0.50 - 0.30 * gen_ratio   # 0.50 → 0.20
    w_eta   = 0.25 + 0.20 * gen_ratio   # 0.25 → 0.45
    w_spike = 0.25 + 0.10 * gen_ratio   # 0.25 → 0.35

    # Métriques brutes par individu
    i_raw     = [e["i_avg"]     for e in evals]
    # Efficacité Q : on veut Q dans la fenêtre, pas minimal → on normalise q_pen
    q_pen_raw = [e["q_pen_avg"] for e in evals]
    spike_raw = [e["spike_avg"] for e in evals]
    surv_raw  = [e["surv_avg"]  for e in evals]

    # OPTION A : normalisation z-score → sigmoid pour chaque métrique
    i_norm     = normalize_population_scores(i_raw)
    q_pen_norm = normalize_population_scores(q_pen_raw)
    spike_norm = normalize_population_scores(spike_raw)
    surv_norm  = normalize_population_scores(surv_raw)

    fitnesses = []
    for i in range(len(evals)):
        # Efficacité combinée (I normalisé × qualité Q normalisée)
        eta_combined = 0.5 * i_norm[i] + 0.5 * q_pen_norm[i]

        fit = (
            w_eta   * eta_combined +
            w_spike * spike_norm[i] +
            w_surv  * surv_norm[i]
        )
        # Pénalité absolue si Q=0 (individu complètement inerte)
        if evals[i]["q_avg"] <= 0.0:
            fit *= 0.0
        fitnesses.append(float(fit))

    return fitnesses

# ============================================================
# 9. BOUCLE ÉVOLUTIVE
# ============================================================

alpha_path = "../03_Core/alpha_post_arena.npz"
data       = np.load(alpha_path)
adn_base   = data[data.files[0]]
adn_shape  = adn_base.shape

nb_gen       = 20
pop_size     = 50
n_elites     = 3
n_immigrants = 3

population = [creer_individu(adn_base) for _ in range(pop_size)]
history    = []

# Arrêt sur homogénéité de population
PATIENCE_VAR  = 6
VAR_THRESHOLD = 0.01
low_var_streak = 0

print("⚡ Lancement du Protocole Landauer V8.2 (z-score + réponse différentielle spikes)...")
print(f"   pop={pop_size} | élites={n_elites} | immigrants/gen={n_immigrants} | gens={nb_gen}")
print(f"   Poids initiaux → surv=0.50 | eta=0.25 | spike=0.25")
print(f"   Poids finaux   → surv=0.20 | eta=0.45 | spike=0.35\n")

for gen in range(nb_gen):

    # Évaluation brute
    evals = [eval_individu_raw(ind) for ind in population]

    # Fitness normalisée sur la population entière
    fitnesses = compute_population_fitness(evals, gen, nb_gen)
    for e, f in zip(evals, fitnesses):
        e["fitness"] = f

    idx      = np.argsort(fitnesses)[::-1]
    best     = evals[idx[0]]
    fit_std  = float(np.std(fitnesses))
    fit_mean = float(np.mean(fitnesses))

    print(
        f"Gen {gen:02d} | "
        f"Fit: {best['fitness']:.4f} (μ={fit_mean:.3f} σ={fit_std:.3f}) | "
        f"Q: {best['q_avg']:.3e} | "
        f"I: {best['i_avg']:.1f} | "
        f"Spike: {best['spike_avg']:.3f} | "
        f"Surv%: {best['surv_rate']*100:.0f}"
    )

    history.append({
        "gen":       gen,
        "fitness":   fitnesses,
        "fit_mean":  fit_mean,
        "fit_std":   fit_std,
        "i_avg":     [e["i_avg"]     for e in evals],
        "q_avg":     [e["q_avg"]     for e in evals],
        "spike_avg": [e["spike_avg"] for e in evals],
        "surv_avg":  [e["surv_avg"]  for e in evals],
    })

    # Arrêt anticipé sur variance
    if fit_std < VAR_THRESHOLD:
        low_var_streak += 1
        if low_var_streak >= PATIENCE_VAR:
            print(f"\n⏹  Arrêt : σ fitness = {fit_std:.4f} < {VAR_THRESHOLD} depuis {PATIENCE_VAR} gens")
            break
    else:
        low_var_streak = 0

    # Sélection
    next_pop = [population[idx[i]] for i in range(n_elites)]

    for _ in range(n_immigrants):
        next_pop.append(creer_individu_aleatoire(adn_shape))

    while len(next_pop) < pop_size:
        tournament = np.random.choice(pop_size, 3, replace=False)
        p_idx      = max(tournament, key=lambda i: fitnesses[i])
        next_pop.append(creer_individu(population[p_idx]["theta"], population[p_idx]["meta"]))

    population = next_pop

# Sauvegarde
np.savez("final_population_v8_2.npz", population=population)

with open("meta_history_v8_2.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n✅ Simulation V8.2 terminée.")
print("   Fichiers : final_population_v8_2.npz | meta_history_v8_2.json")
