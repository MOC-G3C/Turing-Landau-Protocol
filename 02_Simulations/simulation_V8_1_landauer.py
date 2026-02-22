import numpy as np
import copy
import json
import zlib
import math
from scipy.stats import entropy

# ============================================================
# 1. CONSTANTES PHYSIQUES (LANDAUER PROXY)
# ============================================================

np.random.seed(42)

kB    = 1.380649e-23
T_env = 300.0
alpha = 1e-3

# ── CORRECTIF #1 : Q0 relevé de 1e-12 → 1e-3 ──────────────
# Évite que eta explose quand Q → 0, ce qui saturait eta_norm à 1.0
# dès que les individus apprenaient à ne plus dissiper de chaleur.
Q0 = 1e-3

bits_floor = 10.0
Q_floor    = bits_floor * kB * T_env * math.log(2)  # ~2.9e-21 J (référence physique)

# Plage cible pour Q : on veut que les individus opèrent dans cette fenêtre
Q_soft_min = 1e-4   # en-dessous : pénalité croissante
Q_soft_max = 1e-1   # au-dessus  : pénalité croissante

# Référence eta pour la normalisation tanh
ETA_REF = 500.0     # calibré sur les valeurs I ~900 bits et Q ~1e-2 → 1e-3

# Bornes méta-paramètres
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
# 3. PÉNALITÉ Q CONTINUE (CORRECTIF #2)
# ============================================================

def q_penalty_continuous(q_joules):
    """
    Remplace la pénalité binaire ×0.5.
    Retourne 1.0 dans la fenêtre cible [Q_soft_min, Q_soft_max],
    décroît progressivement hors fenêtre.
    Q=0 → pénalité 0 (individu complètement inerte, éliminé).
    """
    if q_joules <= 0.0:
        return 0.0  # individu inerte : fitness nulle

    if Q_soft_min <= q_joules <= Q_soft_max:
        return 1.0

    if q_joules < Q_soft_min:
        # Pénalité log : douce mais qui monte vite vers 0
        ratio = math.log(q_joules / Q_soft_min) / math.log(Q_floor / Q_soft_min + 1e-30)
        return float(np.clip(1.0 + ratio * 0.8, 0.05, 1.0))

    # q_joules > Q_soft_max
    ratio = math.log(q_joules / Q_soft_max) / math.log(10.0)  # décroît sur 1 décade
    return float(np.clip(1.0 - ratio * 0.7, 0.05, 1.0))

# ============================================================
# 4. GÉNÉTIQUE M.O.C.
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
# 5. ÉVALUATION RÉSILIENTE
# ============================================================

def eval_landauer_resilient(ind, k_env=8, steps=2000, low_var_grace=400):

    theta0, meta   = ind["theta"].copy(), ind["meta"]
    low_var_thresh = meta.get("low_var_thresh", 0.012)
    trials_info    = []

    for k in range(k_env):
        np.random.seed(100 + k)

        theta       = theta0.copy()
        heat_sim    = 0.0
        theta_means = []
        env_series  = []

        consecutive_low = 0
        survival_time   = 0

        for t in range(1, steps + 1):

            env_val = np.sin(t / 50.0) * (
                8.0 if np.random.rand() < 0.02 else 1.0
            )

            var        = float(np.var(theta))
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

            kappa = np.clip(
                1.0 + meta["kappa_gain"] * (target_var - var),
                0.1, 4.0
            )

            theta += (
                np.random.randn(*theta.shape) * (0.02 * kappa * meta["mutation_rate"])
                + env_val * 0.005
            )

            survival_time = t

        q_joules = alpha * heat_sim

        mi       = mutual_info_time_series(np.array(theta_means), np.array(env_series))
        cr       = compressibility_bits_1d(np.array(theta_means))
        i_useful = 0.8 * mi + 0.2 * cr

        # ── CORRECTIF #1 appliqué : Q0=1e-3, tanh pour eta_norm ──
        eta      = i_useful / (q_joules + Q0)
        eta_norm = math.tanh(eta / ETA_REF)   # borné [0,1] sans saturation brutale

        survival_frac = survival_time / steps

        # ── CORRECTIF #2 : pénalité continue ──
        penalty = q_penalty_continuous(q_joules)

        trials_info.append({
            "fitness":     (eta_norm, survival_frac, penalty),
            "surv_time":   survival_time,
            "q":           q_joules,
            "i":           i_useful,
        })

    return {
        "_raw":        trials_info,
        "surv_rate":   float(np.mean([1.0 if tr["surv_time"] >= steps else 0.0 for tr in trials_info])),
        "q_avg":       float(np.mean([tr["q"] for tr in trials_info])),
        "i_avg":       float(np.mean([tr["i"] for tr in trials_info])),
        "avg_survtime": float(np.mean([tr["surv_time"] for tr in trials_info])),
    }

# ============================================================
# 6. FITNESS DYNAMIQUE
# ============================================================

def compute_fitness(eval_result, gen, nb_gen):
    gen_ratio = gen / max(nb_gen - 1, 1)
    w_surv    = 0.8 - 0.4 * gen_ratio   # 0.80 → 0.40
    w_eta     = 0.2 + 0.4 * gen_ratio   # 0.20 → 0.60

    fits = []
    for tr in eval_result["_raw"]:
        eta_norm, survival_frac, penalty = tr["fitness"]
        fits.append((w_eta * eta_norm + w_surv * survival_frac) * penalty)
    return float(np.mean(fits))

# ============================================================
# 7. BOUCLE ÉVOLUTIVE
# ============================================================

alpha_path = "../03_Core/alpha_post_arena.npz"
data       = np.load(alpha_path)
adn_base   = data[data.files[0]]
adn_shape  = adn_base.shape

nb_gen      = 20
pop_size    = 50
n_elites    = 3
n_immigrants = 3

population = [creer_individu(adn_base) for _ in range(pop_size)]
history    = []

# ── CORRECTIF #3 : arrêt anticipé sur variance de population ──
# (au lieu du meilleur fitness, qui atteint vite un faux plafond)
PATIENCE_VAR  = 5
VAR_THRESHOLD = 0.005   # std de fitness < ce seuil → population homogène

print("⚡ Lancement du Protocole Landauer V8.1 (Correctifs Q0 / pénalité continue / arrêt variance)...")
print(f"   Q0={Q0} | ETA_REF={ETA_REF} | Q_fenêtre=[{Q_soft_min}, {Q_soft_max}]")
print(f"   pop={pop_size} | élites={n_elites} | immigrants/gen={n_immigrants} | gens={nb_gen}\n")

low_var_streak = 0

for gen in range(nb_gen):

    evals     = [eval_landauer_resilient(ind) for ind in population]
    fitnesses = [compute_fitness(e, gen, nb_gen) for e in evals]
    for e, f in zip(evals, fitnesses):
        e["fitness"] = f

    idx      = np.argsort(fitnesses)[::-1]
    best     = evals[idx[0]]
    best_fit = best["fitness"]
    fit_std  = float(np.std(fitnesses))
    fit_mean = float(np.mean(fitnesses))

    print(
        f"Gen {gen:02d} | "
        f"Fit: {best_fit:.4f} (μ={fit_mean:.3f} σ={fit_std:.3f}) | "
        f"Q: {best['q_avg']:.3e} | "
        f"I: {best['i_avg']:.1f} | "
        f"Surv%: {best['surv_rate']*100:.0f} | "
        f"avgSurv: {best['avg_survtime']:.0f}"
    )

    history.append({
        "gen":      gen,
        "fitness":  fitnesses,
        "fit_mean": fit_mean,
        "fit_std":  fit_std,
        "surv":     [e["avg_survtime"] for e in evals],
        "q_avg":    [e["q_avg"] for e in evals],
        "i_avg":    [e["i_avg"] for e in evals],
    })

    # ── Arrêt anticipé sur homogénéité de la population ──
    if fit_std < VAR_THRESHOLD:
        low_var_streak += 1
        if low_var_streak >= PATIENCE_VAR:
            print(f"\n⏹  Arrêt anticipé à la génération {gen} "
                  f"(σ fitness = {fit_std:.4f} < {VAR_THRESHOLD} depuis {PATIENCE_VAR} gens)")
            break
    else:
        low_var_streak = 0

    # ── Sélection ──
    next_pop = [population[idx[i]] for i in range(n_elites)]

    for _ in range(n_immigrants):
        next_pop.append(creer_individu_aleatoire(adn_shape))

    while len(next_pop) < pop_size:
        tournament = np.random.choice(pop_size, 3, replace=False)
        p_idx      = max(tournament, key=lambda i: fitnesses[i])
        next_pop.append(creer_individu(population[p_idx]["theta"], population[p_idx]["meta"]))

    population = next_pop

# ── Sauvegarde ──
np.savez("final_population_v8_1.npz", population=population)

with open("meta_history_v8_1.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n✅ Simulation V8.1 terminée.")
print("   Fichiers : final_population_v8_1.npz | meta_history_v8_1.json")
