import numpy as np
import copy
import json
import zlib
import math
from scipy.stats import entropy
from collections import deque

# ============================================================\n# 1. CONSTANTES PHYSIQUES ET HYPERPARAMÈTRES V8.9
# ============================================================\n
np.random.seed(42)

kB    = 1.380649e-23
T_env = 300.0
alpha = 1e-3
Q0    = 1e-3

Q_soft_min = 1e-4
Q_soft_max = 5e-1
SPIKE_PROB = 0.08

# --- NOUVELLES CONSTANTES LANDAUER / LUCIDITÉ ---
INFO_TO_BITS = 0.5         # Échelle : unité d'état -> bits (départ conservatif)
LN2 = math.log(2.0)
Q_MAX_LANDAUER = Q_soft_max  
SEUIL_REACTION = 0.05      # Magnitude minimale moyenne de theta pour considérer une réaction
MAX_INACTION = 3           # Nombre de spikes ignorés tolérés avant mort cognitive

META_BOUNDS = {
    "kappa_gain":       (0.01, 2.0),
    "cooling_strength": (0.30, 0.95),
    "mutation_rate":    (0.01, 0.30),
    "target_base":      (0.005, 0.15),
    "low_var_thresh":   (0.003, 0.05),
    "tau":              (0.0,  0.95),
    "env_gain_spike":   (0.0005, 0.15),
    "env_gain_calm":    (0.0001, 0.02),
}

ZSCORE_WINDOW = 3
_metric_history = {
    "i":     deque(maxlen=ZSCORE_WINDOW),
    "q_pen": deque(maxlen=ZSCORE_WINDOW),
    "luc":   deque(maxlen=ZSCORE_WINDOW) # Ajout de l'historique de lucidité
}

# ============================================================\n# 2. FONCTIONS DE BASE ET MÉTRIQUES
# ============================================================\n
def creer_individu_aleatoire(shape=(16, 16)):
    return {
        "theta": np.random.randn(*shape) * 0.1,
        "meta": {k: np.random.uniform(v[0], v[1]) for k, v in META_BOUNDS.items()}
    }

def mutual_info_time_series(x, y, bins=10):
    if len(x) < 2: return 0.0
    x_norm = (x - np.min(x)) / (np.ptp(x) + 1e-9)
    y_norm = (y - np.min(y)) / (np.ptp(y) + 1e-9)
    hist_2d, _, _ = np.histogram2d(x_norm, y_norm, bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return float(np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs])))

def compressibility_bits_1d(arr):
    if len(arr) == 0: return 0.0
    quantized = np.round(arr * 100).astype(np.int32).tobytes()
    comp = zlib.compress(quantized)
    return float(len(comp)) * 8.0

def differential_spike_response(theta_means, env_vals, threshold=2.0):
    t_m = np.array(theta_means)
    e_v = np.array(env_vals)
    spikes = np.where(np.abs(e_v) > threshold)[0]
    calms  = np.where(np.abs(e_v) <= threshold)[0]
    if len(spikes) == 0 or len(calms) == 0: return 0.0
    v_spikes = np.var(t_m[spikes])
    v_calms  = np.var(t_m[calms])
    return float(v_spikes / (v_calms + 1e-9))

def q_survival_penalty(q_joules, surv_frac):
    if surv_frac < 1.0: return surv_frac
    if q_joules <= Q_soft_min: return 1.0
    if q_joules >= Q_MAX_LANDAUER: return 0.01
    ratio = (q_joules - Q_soft_min) / (Q_MAX_LANDAUER - Q_soft_min)
    return 1.0 - ratio

def q_penalty_continuous(q_joules):
    if q_joules <= Q_soft_min: return 0.0
    if q_joules >= Q_MAX_LANDAUER: return 1.0
    return (q_joules - Q_soft_min) / (Q_MAX_LANDAUER - Q_soft_min)

def normalize_with_history(values, key):
    arr = np.array(values, dtype=float)
    if len(arr) == 0: return arr
    if len(_metric_history[key]) > 0:
        past = np.concatenate(_metric_history[key])
        global_mean = np.mean(past)
        global_std  = np.std(past) + 1e-9
    else:
        global_mean = np.mean(arr)
        global_std  = np.std(arr) + 1e-9
    _metric_history[key].append(arr)
    z = (arr - global_mean) / global_std
    z_min, z_max = np.min(z), np.max(z)
    if z_max > z_min:
        norm = (z - z_min) / (z_max - z_min)
    else:
        norm = np.ones_like(z)
    return norm

# ============================================================\n# 3. MOTEUR DE SIMULATION V8.9 (L'Arène)
# ============================================================\n
def eval_individu_raw(adn, steps=200):
    trials = []
    
    for _ in range(3):
        theta = adn["theta"].copy()
        meta = adn["meta"]
        
        tau = meta["tau"]
        low_var_thresh = meta["low_var_thresh"]
        env_gain_spike = meta["env_gain_spike"]
        env_gain_calm  = meta["env_gain_calm"]
        
        theta_means = []
        env_series = []
        
        survival_time = 0
        consecutive_low = 0
        low_var_grace = 15
        h = 0.0
        
        # --- V8.9 : Initialisation Landauer / Lucidité ---
        Q_heat = 0.0
        lucidity_score = 0.0
        inaction_penalty = 0.0
        consecutive_spike_ignores = 0
        heat_sim_old = 0.0 # Gardé pour compatibilité/comparaison
        
        for t in range(1, steps + 1):
            env_val = np.sin(t / 50.0) * (8.0 if np.random.rand() < SPIKE_PROB else 1.0)
            h       = tau * h + (1.0 - tau) * env_val

            var         = float(np.var(theta))
            mean_so_far = float(np.mean(theta_means)) if theta_means else 0.0
            target_var  = meta["target_base"] + 0.02 * abs(h - mean_so_far)

            if var > target_var * 2.0:
                theta *= meta["cooling_strength"]
                var    = float(np.var(theta))

            # === V8.9: Reaction / Plasticity step ===
            kappa    = np.clip(1.0 + meta["kappa_gain"] * (target_var - var), 0.1, 4.0)
            is_spike = abs(h) > 2.0
            gain_now = env_gain_spike if is_spike else env_gain_calm

            theta += (np.random.randn(*theta.shape) * (0.02 * kappa * meta["mutation_rate"]) + h * gain_now)

            # === V8.9: Forgetting / Relaxation (Taxe de Landauer) ===
            theta_after_relax = theta * (1.0 - (1.0 - tau) * 0.5) 
            erased_state = theta - theta_after_relax

            erased_bits = np.sum(np.abs(erased_state)) * INFO_TO_BITS
            dQ = kB * T_env * LN2 * float(abs(erased_bits))  
            Q_heat += dQ

            theta = theta_after_relax

            # Book-keeping
            heat_sim_old += max(0.0, (var * 15.0) - 0.3)  
            theta_means.append(float(np.mean(theta)))
            env_series.append(float(env_val))

            # === V8.9: Lucidity & inaction checks ===
            mean_theta_now = float(np.mean(theta))
            if is_spike:
                if abs(mean_theta_now) < SEUIL_REACTION:
                    inaction_penalty += 1.0
                    consecutive_spike_ignores += 1
                else:
                    lucidity_score += 1.0
                    consecutive_spike_ignores = 0

            # Early-stop variance trop faible
            if var < low_var_thresh:
                consecutive_low += 1
            else:
                consecutive_low = 0

            if consecutive_low >= low_var_grace:
                survival_time = t
                break

            survival_time = t

        # --- Fin du trial ---
        surv_frac = survival_time / steps
        surv_pen  = q_survival_penalty(Q_heat, surv_frac)

        mi        = mutual_info_time_series(np.array(theta_means), np.array(env_series))
        cr        = compressibility_bits_1d(np.array(theta_means))
        i_useful  = 0.8 * mi + 0.2 * cr
        spike_rsp = differential_spike_response(theta_means, env_series)

        trials.append({
            "i":         i_useful,
            "q":         Q_heat,          # Dissipation Landauer
            "q_old":     alpha * heat_sim_old, 
            "spike":     spike_rsp,
            "surv_frac": surv_pen,
            "surv_raw":  surv_frac,
            "q_penalty": q_penalty_continuous(Q_heat),
            "lucidity":  lucidity_score,
            "inaction":  inaction_penalty,
        })
        
    return {
        "i_avg":         np.mean([t["i"] for t in trials]),
        "q_avg":         np.mean([t["q"] for t in trials]),
        "q_old_avg":     np.mean([t["q_old"] for t in trials]),
        "spike_avg":     np.mean([t["spike"] for t in trials]),
        "surv_avg":      np.mean([t["surv_frac"] for t in trials]),
        "q_pen_avg":     np.mean([t["q_penalty"] for t in trials]),
        "lucidity_avg":  np.mean([t["lucidity"] for t in trials]),
        "inaction_avg":  np.mean([t["inaction"] for t in trials]),
    }

def compute_population_fitness(evals):
    n = len(evals)
    raw_fits = np.zeros(n)
    
    i_norm    = normalize_with_history([e["i_avg"] for e in evals], "i")
    q_pen_n   = normalize_with_history([e["q_pen_avg"] for e in evals], "q_pen")
    luc_n     = normalize_with_history([e["lucidity_avg"] for e in evals], "luc")
    surv_norm = np.array([e["surv_avg"] for e in evals])

    # Poids ajustés pour V8.9
    w_lucidity = 0.40  # Récompense forte pour l'encodage
    w_eta      = 0.30  # Efficacité (faible pénalité thermodynamique)
    w_surv     = 0.30  # Survie basique

    for idx, e in enumerate(evals):
        # Le couperet cognitif : mort si trop d'inaction
        if e["inaction_avg"] > MAX_INACTION:
            raw_fits[idx] = 0.01
        else:
            raw_fits[idx] = (w_lucidity * luc_n[idx]) + (w_eta * (1.0 - q_pen_n[idx])) + (w_surv * surv_norm[idx])
            
    if np.sum(raw_fits) == 0:
        return np.ones(n) / n
    return raw_fits / np.sum(raw_fits)

# ============================================================\n# 4. ALGORITHME GÉNÉTIQUE
# ============================================================\n
def croiser_adn(adn1, adn2):
    mask = np.random.rand(16, 16) > 0.5
    new_theta = np.where(mask, adn1["theta"], adn2["theta"])
    new_meta = {}
    for k in META_BOUNDS.keys():
        new_meta[k] = adn1["meta"][k] if np.random.rand() > 0.5 else adn2["meta"][k]
    return {"theta": new_theta, "meta": new_meta}

def muter_adn(adn, mutation_rate=0.1):
    new_theta = adn["theta"] + np.random.randn(16, 16) * mutation_rate
    new_meta = copy.deepcopy(adn["meta"])
    for k, (b_min, b_max) in META_BOUNDS.items():
        if np.random.rand() < 0.2:
            delta = np.random.randn() * 0.1 * (b_max - b_min)
            new_meta[k] = np.clip(new_meta[k] + delta, b_min, b_max)
    return {"theta": new_theta, "meta": new_meta}

def run_evolution(pop_size=40, generations=20):
    adn_shape = (16, 16)
    population = [creer_individu_aleatoire(adn_shape) for _ in range(pop_size)]
    
    history_val = []
    
    n_elites = max(1, int(0.1 * pop_size))
    n_immigrants = int(0.1 * pop_size)
    
    PATIENCE_VAR = 5
    VAR_THRESHOLD = 1e-4
    low_var_streak = 0

    print(f"🔬 Début Évolution V8.9 (Landauer Strict) - {generations} Gen | Pop: {pop_size}")

    for gen in range(generations):
        evals = [eval_individu_raw(ind) for ind in population]
        probs = compute_population_fitness(evals)
        
        # Tracking métriques V8.9
        luc_mean = np.mean([e["lucidity_avg"] for e in evals])
        ina_mean = np.mean([e["inaction_avg"] for e in evals])
        q_landauer_mean = np.mean([e["q_avg"] for e in evals])
        surv_pct = np.mean([e["surv_avg"] for e in evals]) * 100
        
        idx = np.argsort(probs)[::-1]
        best_idx = idx[0]
        fit_std = float(np.std(probs))

        print(f"Gen {gen:03d} | Max Fit: {probs[best_idx]:.4f} | Surv: {surv_pct:5.1f}% | Luc: {luc_mean:.2f} | Inact: {ina_mean:.2f} | Q(Landauer): {q_landauer_mean:.2e} | Std: {fit_std:.4f}")

        # Validation sur le champion (à chaque génération comme demandé)
        val_eval = eval_individu_raw(population[best_idx], steps=300)
        history_val.append(val_eval["spike_avg"])

        if fit_std < VAR_THRESHOLD:
            low_var_streak += 1
            if low_var_streak >= PATIENCE_VAR:
                print(f"\\n⏹  Arrêt : σ={fit_std:.4f} < {VAR_THRESHOLD} depuis {PATIENCE_VAR} gens")
                break
        else:
            low_var_streak = 0

        next_pop = [population[idx[i]] for i in range(n_elites)]
        for _ in range(n_immigrants):
            next_pop.append(creer_individu_aleatoire(adn_shape))
            
        while len(next_pop) < pop_size:
            tournament = np.random.choice(pop_size, 3, replace=False)
            p_idx = max(tournament, key=lambda i: probs[i])
            parent1 = population[p_idx]
            parent2 = population[np.random.choice(pop_size)]
            enfant = croiser_adn(parent1, parent2)
            enfant = muter_adn(enfant)
            next_pop.append(enfant)
            
        population = next_pop

    print("\\n✅ Évolution terminée.")
    print("Spike Response du Champion sur la validation :", history_val)
    
    # Export du Master DNA
    evals_final = [eval_individu_raw(ind) for ind in population]
    probs_final = compute_population_fitness(evals_final)
    idx_final = np.argsort(probs_final)[::-1]
    population_triee = [population[i] for i in idx_final]
    
    np.savez_compressed("final_population_v8_9.npz", population=np.array(population_triee, dtype=object))
    print("💾 Population finale sauvegardée dans 'final_population_v8_9.npz'")

if __name__ == "__main__":
    run_evolution(pop_size=40, generations=20)