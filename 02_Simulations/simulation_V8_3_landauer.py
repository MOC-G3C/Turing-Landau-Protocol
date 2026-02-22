import numpy as np
import copy
import json
import zlib
import math
from scipy.stats import entropy
from collections import deque

# ============================================================
# 1. CONSTANTES PHYSIQUES
# ============================================================

np.random.seed(42)

kB    = 1.380649e-23
T_env = 300.0
alpha = 1e-3
Q0    = 1e-3

Q_soft_min = 1e-4
Q_soft_max = 5e-1

META_BOUNDS = {
    "kappa_gain":       (0.01, 2.0),
    "cooling_strength": (0.30, 0.95),
    "mutation_rate":    (0.01, 0.30),
    "target_base":      (0.005, 0.15),
    "low_var_thresh":   (0.003, 0.05),
}

# Historique glissant pour z-score (Option A fix)
ZSCORE_WINDOW = 3   # générations
_metric_history = {
    "i":     deque(maxlen=ZSCORE_WINDOW),
    "q_pen": deque(maxlen=ZSCORE_WINDOW),
    "spike": deque(maxlen=ZSCORE_WINDOW),
    "surv":  deque(maxlen=ZSCORE_WINDOW),
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
# 3. RÉPONSE DIFFÉRENTIELLE AUX SPIKES — FIX V8.3
#    Utilise la DÉRIVÉE de theta (réactivité immédiate)
#    + seuil spike abaissé pour plus d'événements
# ============================================================

SPIKE_PROB = 0.08   # 8% au lieu de 2% → ~160 spikes/2000 steps

def differential_spike_response(theta_means, env_series):
    """
    Compare la DÉRIVÉE ABSOLUE de theta pendant les spikes vs hors-spikes.
    La dérivée capte la réactivité immédiate à chaque perturbation,
    contrairement à la variance globale qui dilue l'effet dans le temps.

    Retourne un score dans [0, 1].
    """
    theta_arr = np.array(theta_means)
    env_arr   = np.array(env_series)

    if len(theta_arr) < 3:
        return 0.0

    # Dérivée absolue de theta (variation step-à-step)
    d_theta = np.abs(np.diff(theta_arr))
    env_mid = env_arr[1:]   # aligner avec diff

    # FIX : seuil plus bas car amplitude = sin * 8 → max ~8
    spike_mask = np.abs(env_mid) > 2.0
    calm_mask  = np.abs(env_mid) <= 1.0   # zone vraiment calme

    n_spike = spike_mask.sum()
    n_calm  = calm_mask.sum()

    if n_spike < 10 or n_calm < 10:
        return 0.0

    mean_d_spike = float(np.mean(d_theta[spike_mask]))
    mean_d_calm  = float(np.mean(d_theta[calm_mask]))

    if mean_d_calm < 1e-12:
        return 0.0

    ratio = mean_d_spike / (mean_d_calm + 1e-12)
    # ratio=1 → 0 (pas de réponse), ratio=5 → 0.58, ratio=20 → 0.88
    score = math.tanh(max(0.0, ratio - 1.0) / 6.0)
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
        ratio = math.log(q_joules / Q_soft_min) / math.log(1e-21 / Q_soft_min + 1e-30)
        return float(np.clip(1.0 + ratio * 0.8, 0.05, 1.0))
    ratio = math.log(q_joules / Q_soft_max) / math.log(10.0)
    return float(np.clip(1.0 - ratio * 0.7, 0.05, 1.0))

# ============================================================
# 5. NORMALISATION Z-SCORE AVEC HISTORIQUE GLISSANT (FIX A)
# ============================================================

def normalize_with_history(values, key):
    """
    Normalise via z-score sur l'historique glissant des ZSCORE_WINDOW
    dernières générations + la génération courante.
    Évite l'amplification de bruit quand la pop converge.
    """
    # Ajouter la génération courante à l'historique
    _metric_history[key].append(list(values))

    # Référence = toutes les valeurs des dernières N générations
    all_vals = []
    for gen_vals in _metric_history[key]:
        all_vals.extend(gen_vals)

    arr = np.array(all_vals, dtype=float)
    mu  = np.mean(arr)
    std = np.std(arr) + 1e-10

    # Normaliser seulement les valeurs courantes
    z = (np.array(values, dtype=float) - mu) / std
    return 1.0 / (1.0 + np.exp(-z))

# ============================================================
# 6. FITNESS SHARING (FIX plateau)
#    Pénalise les individus génétiquement trop proches
# ============================================================

def fitness_sharing(fitnesses, population, sigma_share=0.5, alpha_share=1.0):
    """
    Divise le fitness de chaque individu par la somme des similarités
    avec ses voisins génétiques dans la population.
    Force l'exploration de niches différentes.

    sigma_share : distance génétique seuil (en norme L2 sur theta normalisé)
    """
    n = len(population)

    # Normaliser les theta pour comparaison équitable
    thetas = []
    for ind in population:
        t = ind["theta"].flatten()
        t = t / (np.linalg.norm(t) + 1e-10)
        thetas.append(t)

    shared_fits = []
    for i in range(n):
        niche_count = 0.0
        for j in range(n):
            dist = np.linalg.norm(thetas[i] - thetas[j])
            if dist < sigma_share:
                sh = 1.0 - (dist / sigma_share) ** alpha_share
                niche_count += sh
        # niche_count >= 1 (soi-même)
        shared_fits.append(fitnesses[i] / max(niche_count, 1.0))

    return shared_fits

# ============================================================
# 7. GÉNÉTIQUE M.O.C.
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
# 8. ÉVALUATION — métriques brutes
# ============================================================

def eval_individu_raw(ind, k_env=8, steps=2000, low_var_grace=400):

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

            # FIX : spike_prob augmenté à 8%
            env_val     = np.sin(t / 50.0) * (8.0 if np.random.rand() < SPIKE_PROB else 1.0)
            var         = float(np.var(theta))
            mean_so_far = float(np.mean(theta_means)) if theta_means else 0.0
            target_var  = meta["target_base"] + 0.02 * abs(env_val - mean_so_far)

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
            "i":         i_useful,
            "q":         q_joules,
            "spike":     spike_rsp,
            "surv_frac": survival_time / steps,
            "q_penalty": q_penalty_continuous(q_joules),
        })

    return {
        "i_avg":        float(np.mean([t["i"]         for t in trials])),
        "q_avg":        float(np.mean([t["q"]         for t in trials])),
        "spike_avg":    float(np.mean([t["spike"]     for t in trials])),
        "surv_avg":     float(np.mean([t["surv_frac"] for t in trials])),
        "q_pen_avg":    float(np.mean([t["q_penalty"] for t in trials])),
        "surv_rate":    float(np.mean([1.0 if t["surv_frac"] >= 1.0 else 0.0 for t in trials])),
        "avg_survtime": float(np.mean([t["surv_frac"] * steps for t in trials])),
    }

# ============================================================
# 9. CALCUL FITNESS — z-score glissant + sharing
# ============================================================

def compute_population_fitness(evals, population, gen, nb_gen):
    gen_ratio = gen / max(nb_gen - 1, 1)

    w_surv  = 0.50 - 0.30 * gen_ratio   # 0.50 → 0.20
    w_eta   = 0.25 + 0.20 * gen_ratio   # 0.25 → 0.45
    w_spike = 0.25 + 0.10 * gen_ratio   # 0.25 → 0.35

    i_norm     = normalize_with_history([e["i_avg"]     for e in evals], "i")
    q_pen_norm = normalize_with_history([e["q_pen_avg"] for e in evals], "q_pen")
    spike_norm = normalize_with_history([e["spike_avg"] for e in evals], "spike")
    surv_norm  = normalize_with_history([e["surv_avg"]  for e in evals], "surv")

    raw_fits = []
    for i in range(len(evals)):
        eta_combined = 0.5 * i_norm[i] + 0.5 * q_pen_norm[i]
        fit = w_eta * eta_combined + w_spike * spike_norm[i] + w_surv * surv_norm[i]
        if evals[i]["q_avg"] <= 0.0:
            fit = 0.0
        raw_fits.append(float(fit))

    # Fitness sharing pour forcer la diversité des niches
    shared_fits = fitness_sharing(raw_fits, population, sigma_share=0.5)

    return raw_fits, shared_fits

# ============================================================
# 10. BOUCLE ÉVOLUTIVE
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

PATIENCE_VAR  = 6
VAR_THRESHOLD = 0.008
low_var_streak = 0

print("⚡ Lancement du Protocole Landauer V8.3")
print(f"   Fixes : spike dérivée (p={SPIKE_PROB}) | fitness sharing | z-score glissant ({ZSCORE_WINDOW} gens)")
print(f"   pop={pop_size} | élites={n_elites} | immigrants/gen={n_immigrants} | gens={nb_gen}\n")

for gen in range(nb_gen):

    evals = [eval_individu_raw(ind) for ind in population]

    raw_fits, shared_fits = compute_population_fitness(evals, population, gen, nb_gen)

    # Sélection sur shared_fits (diversité), affichage sur raw_fits (performance réelle)
    idx      = np.argsort(shared_fits)[::-1]
    best_raw = evals[idx[0]]
    fit_std  = float(np.std(raw_fits))
    fit_mean = float(np.mean(raw_fits))

    print(
        f"Gen {gen:02d} | "
        f"Fit: {raw_fits[idx[0]]:.4f} (μ={fit_mean:.3f} σ={fit_std:.3f}) | "
        f"Q: {best_raw['q_avg']:.3e} | "
        f"I: {best_raw['i_avg']:.1f} | "
        f"Spike: {best_raw['spike_avg']:.3f} | "
        f"Surv%: {best_raw['surv_rate']*100:.0f}"
    )

    history.append({
        "gen":        gen,
        "fitness":    raw_fits,
        "shared_fit": shared_fits,
        "fit_mean":   fit_mean,
        "fit_std":    fit_std,
        "i_avg":      [e["i_avg"]     for e in evals],
        "q_avg":      [e["q_avg"]     for e in evals],
        "spike_avg":  [e["spike_avg"] for e in evals],
        "surv_avg":   [e["surv_avg"]  for e in evals],
    })

    if fit_std < VAR_THRESHOLD:
        low_var_streak += 1
        if low_var_streak >= PATIENCE_VAR:
            print(f"\n⏹  Arrêt : σ={fit_std:.4f} < {VAR_THRESHOLD} depuis {PATIENCE_VAR} gens")
            break
    else:
        low_var_streak = 0

    # Sélection : élites selon shared_fits, tournoi aussi sur shared_fits
    next_pop = [population[idx[i]] for i in range(n_elites)]

    for _ in range(n_immigrants):
        next_pop.append(creer_individu_aleatoire(adn_shape))

    while len(next_pop) < pop_size:
        tournament = np.random.choice(pop_size, 3, replace=False)
        p_idx      = max(tournament, key=lambda i: shared_fits[i])
        next_pop.append(creer_individu(population[p_idx]["theta"], population[p_idx]["meta"]))

    population = next_pop

np.savez("final_population_v8_3.npz", population=population)

with open("meta_history_v8_3.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n✅ Simulation V8.3 terminée.")
print("   Fichiers : final_population_v8_3.npz | meta_history_v8_3.json")
