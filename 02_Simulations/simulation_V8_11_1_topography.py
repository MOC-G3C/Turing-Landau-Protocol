import numpy as np
import copy
import json
import zlib
import math
from scipy.stats import entropy
from collections import deque

# ============================================================
# 1. CONSTANTES V8.11.1 (CORRECTION DE LA DILUTION SPATIALE)
# ============================================================
np.random.seed(42)

kB    = 1.380649e-23
T_env = 300.0
alpha = 1e-3

SPIKE_PROB_FAST = 0.12  
SPIKE_PROB_SLOW = 0.02

INFO_TO_BITS = 0.2         
LN2 = math.log(2.0)
Q_soft_min = 1e-20         
Q_MAX_LANDAUER = 5e-19     

SEUIL_REACTION = 0.08      
MAX_INACTION = 3           

META_BOUNDS = {
    "kappa_gain":       (0.01, 2.0),
    "cooling_strength": (0.30, 0.95),
    "mutation_rate":    (0.01, 0.30),
    "target_base":      (0.005, 0.15),
    "low_var_thresh":   (0.003, 0.05),
    "env_gain_spike":   (0.0005, 0.15),
    "env_gain_calm":    (0.0001, 0.02),
}

ZSCORE_WINDOW = 3
_metric_history = {
    "i": deque(maxlen=ZSCORE_WINDOW), "q_pen": deque(maxlen=ZSCORE_WINDOW)
}

# --- Définition des Hotspots Spatiaux Amplifiés ---
HOTSPOTS = [
    (3, 3, 10.0, 0.0),            
    (12, 12, 8.0, np.pi/2)        
]
XS = np.arange(16)[:, None]
YS = np.arange(16)[None, :]

# ============================================================
# 2. FONCTIONS DE BASE
# ============================================================
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
    return float(len(zlib.compress(quantized))) * 8.0

def differential_spike_response(theta_means, env_vals, threshold=2.0):
    t_m = np.array(theta_means)
    e_v = np.array(env_vals)
    spikes = np.where(np.abs(e_v) > threshold)[0]
    calms  = np.where(np.abs(e_v) <= threshold)[0]
    if len(spikes) == 0 or len(calms) == 0: return 0.0
    return float(np.var(t_m[spikes]) / (np.var(t_m[calms]) + 1e-9))

def q_survival_penalty(q_joules, surv_frac):
    if surv_frac < 1.0: return surv_frac
    if q_joules <= Q_soft_min: return 1.0
    if q_joules >= Q_MAX_LANDAUER: return 0.01
    return 1.0 - ((q_joules - Q_soft_min) / (Q_MAX_LANDAUER - Q_soft_min))

def q_penalty_continuous(q_joules):
    if q_joules <= Q_soft_min: return 0.0
    if q_joules >= Q_MAX_LANDAUER: return 1.0
    return (q_joules - Q_soft_min) / (Q_MAX_LANDAUER - Q_soft_min)

def normalize_with_history(values, key):
    arr = np.array(values, dtype=float)
    if len(arr) == 0: return arr
    if len(_metric_history[key]) > 0:
        past = np.concatenate(_metric_history[key])
        g_mean, g_std = np.mean(past), np.std(past) + 1e-9
    else:
        g_mean, g_std = np.mean(arr), np.std(arr) + 1e-9
    _metric_history[key].append(arr)
    z = (arr - g_mean) / g_std
    z_min, z_max = np.min(z), np.max(z)
    return (z - z_min) / (z_max - z_min) if z_max > z_min else np.ones_like(z)

def creer_individu_aleatoire(shape=(16, 16)):
    return {
        "theta": np.random.randn(*shape) * 0.1,
        "tau_matrix": np.random.uniform(0.4, 0.6, shape),
        "meta": {k: np.random.uniform(v[0], v[1]) for k, v in META_BOUNDS.items()}
    }

# ============================================================
# 3. MOTEUR DE SIMULATION V8.11.1
# ============================================================
def eval_individu_raw(adn, steps=200):
    trials = []
    
    for _ in range(3):
        theta = adn["theta"].copy()
        tau_mat = adn["tau_matrix"].copy() 
        meta = adn["meta"]
        
        low_var_thresh = meta["low_var_thresh"]
        env_gain_spike = meta["env_gain_spike"]
        env_gain_calm  = meta["env_gain_calm"]
        
        theta_means = []
        env_series = [] 
        
        survival_time = 0
        consecutive_low = 0
        low_var_grace = 150 
        
        h = np.zeros_like(theta) 
        
        Q_heat = 0.0
        lucidity_score = 0.0
        inaction_penalty = 0.0
        consecutive_spike_ignores = 0
        
        for t in range(1, steps + 1):
            
            env_field = np.zeros_like(theta)
            
            fast_comp = (1.0 if np.random.rand() < SPIKE_PROB_FAST else 0.1) * np.sin(t / 8.0)
            slow_comp = (1.0 if np.random.rand() < SPIKE_PROB_SLOW else 0.1) * np.sin(t / 200.0)
            
            for (x, y, amp, phase) in HOTSPOTS:
                dist2 = (XS - x)**2 + (YS - y)**2
                env_field += amp * np.exp(-dist2 / 9.0) * (fast_comp + slow_comp + np.sin(t/10.0 + phase)*0.2)
            
            env_field += 0.3 * np.sin(t / 100.0)
            
            max_env_val = float(np.max(np.abs(env_field)))

            h = tau_mat * h + (1.0 - tau_mat) * env_field

            var         = float(np.var(theta))
            mean_so_far = float(np.mean(theta_means)) if theta_means else 0.0
            
            target_var  = meta["target_base"] + 0.02 * abs(float(np.max(np.abs(h))) - mean_so_far)

            if var > target_var * 2.0:
                theta *= meta["cooling_strength"]
                var    = float(np.var(theta))

            kappa    = np.clip(1.0 + meta["kappa_gain"] * (target_var - var), 0.1, 4.0)
            
            is_spike = max_env_val > 4.0 
            gain_now = env_gain_spike if is_spike else env_gain_calm

            theta += (np.random.randn(*theta.shape) * (0.02 * kappa * meta["mutation_rate"]) + h * gain_now)

            theta_after_relax = theta * (1.0 - (1.0 - tau_mat) * 0.5) 
            erased_state = theta - theta_after_relax
            erased_bits = np.sum(np.abs(erased_state)) * INFO_TO_BITS
            Q_heat += kB * T_env * LN2 * float(abs(erased_bits))  
            theta = theta_after_relax

            theta_means.append(float(np.mean(theta)))
            env_series.append(max_env_val)

            max_theta_now = float(np.max(np.abs(theta)))
            if is_spike:
                if max_theta_now < SEUIL_REACTION:
                    inaction_penalty += 1.0
                    consecutive_spike_ignores += 1
                else:
                    lucidity_score += 1.0
                    consecutive_spike_ignores = 0

            if var < low_var_thresh: consecutive_low += 1
            else: consecutive_low = 0

            if consecutive_low >= low_var_grace:
                survival_time = t
                break

            survival_time = t

        surv_frac = survival_time / steps
        surv_pen  = q_survival_penalty(Q_heat, surv_frac)

        trials.append({
            "i":         0.8 * mutual_info_time_series(np.array(theta_means), np.array(env_series)) + 0.2 * compressibility_bits_1d(np.array(theta_means)),
            "q":         Q_heat,
            "spike":     differential_spike_response(theta_means, env_series),
            "surv_frac": surv_pen,
            "q_penalty": q_penalty_continuous(Q_heat),
            "lucidity":  lucidity_score,
            "inaction":  inaction_penalty,
        })
        
    return {
        "i_avg":         np.mean([t["i"] for t in trials]),
        "q_avg":         np.mean([t["q"] for t in trials]),
        "spike_avg":     np.mean([t["spike"] for t in trials]),
        "surv_avg":      np.mean([t["surv_frac"] for t in trials]),
        "q_pen_avg":     np.mean([t["q_penalty"] for t in trials]),
        "lucidity_avg":  np.mean([t["lucidity"] for t in trials]),
        "inaction_avg":  np.mean([t["inaction"] for t in trials]),
        "tau_mean":      float(np.mean(adn["tau_matrix"])),
        "tau_std":       float(np.std(adn["tau_matrix"]))
    }

def compute_population_fitness(evals):
    n = len(evals)
    raw_fits = np.zeros(n)
    
    q_pen_n   = normalize_with_history([e["q_pen_avg"] for e in evals], "q_pen")
    surv_norm = np.array([e["surv_avg"] for e in evals])

    luc_raw = np.array([e["lucidity_avg"] for e in evals])
    if np.ptp(luc_raw) == 0:
        luc_n = np.zeros_like(luc_raw)
    else:
        luc_n = (luc_raw - np.min(luc_raw)) / (np.ptp(luc_raw) + 1e-9)

    w_lucidity = 0.50  
    w_eta      = 0.30  
    w_surv     = 0.20  

    for idx, e in enumerate(evals):
        if e["inaction_avg"] > MAX_INACTION:
            raw_fits[idx] = 0.01 
        else:
            bonus_std = min(e["tau_std"], 0.25) * 0.12 
            raw_fits[idx] = (w_lucidity * luc_n[idx]) + (w_eta * (1.0 - q_pen_n[idx])) + (w_surv * surv_norm[idx]) + bonus_std
            
    if np.sum(raw_fits) == 0: return np.ones(n) / n
    return raw_fits / np.sum(raw_fits)

# ============================================================
# 4. ALGORITHME GÉNÉTIQUE
# ============================================================
def croiser_adn(adn1, adn2):
    mask = np.random.rand(16, 16) > 0.5
    new_theta = np.where(mask, adn1["theta"], adn2["theta"])
    new_tau   = np.where(mask, adn1["tau_matrix"], adn2["tau_matrix"])
    # CORRECTION ICI : .keys() au lieu de .items()
    new_meta = {k: adn1["meta"][k] if np.random.rand() > 0.5 else adn2["meta"][k] for k in META_BOUNDS.keys()}
    return {"theta": new_theta, "tau_matrix": new_tau, "meta": new_meta}

def muter_adn(adn, mutation_rate=0.1):
    new_theta = adn["theta"] + np.random.randn(16, 16) * mutation_rate
    new_tau = np.clip(adn["tau_matrix"] + np.random.randn(16, 16) * (mutation_rate * 0.5), 0.01, 0.99)
    new_meta = copy.deepcopy(adn["meta"])
    for k, (b_min, b_max) in META_BOUNDS.items():
        if np.random.rand() < 0.2:
            new_meta[k] = np.clip(new_meta[k] + np.random.randn() * 0.1 * (b_max - b_min), b_min, b_max)
    return {"theta": new_theta, "tau_matrix": new_tau, "meta": new_meta}

def run_evolution(pop_size=40, generations=15):
    adn_shape = (16, 16)
    population = [creer_individu_aleatoire(adn_shape) for _ in range(pop_size)]
    
    print(f"🔬 Début Évolution V8.11.1 (Correction Dilution Spatiale & NumPy 2.0) - {generations} Gen")

    for gen in range(generations):
        evals = [eval_individu_raw(ind) for ind in population]
        probs = compute_population_fitness(evals)
        
        idx = np.argsort(probs)[::-1]
        
        print(f"Gen {gen:03d} | Surv: {np.mean([e['surv_avg'] for e in evals])*100:5.1f}% | Luc: {np.mean([e['lucidity_avg'] for e in evals]):.2f} | Q: {np.mean([e['q_avg'] for e in evals]):.2e} | Tau_Std: {np.mean([e['tau_std'] for e in evals]):.3f}")

        next_pop = [population[idx[i]] for i in range(max(1, int(0.1 * pop_size)))]
        for _ in range(int(0.1 * pop_size)): next_pop.append(creer_individu_aleatoire(adn_shape))
            
        while len(next_pop) < pop_size:
            p_idx = max(np.random.choice(pop_size, 3, replace=False), key=lambda i: probs[i])
            enfant = muter_adn(croiser_adn(population[p_idx], population[np.random.choice(pop_size)]))
            next_pop.append(enfant)
            
        population = next_pop

    print("\n✅ Évolution terminée.")
    
    evals_final = [eval_individu_raw(ind) for ind in population]
    champion_tau = population[np.argsort(compute_population_fitness(evals_final))[::-1][0]]["tau_matrix"]
    
    print("\n🧠 CARTOGRAPHIE DU CORTEX V8.11.1 :")
    print("Légende : ░░ Senseur (Oubli rapide) | ▒▒ Relais | ██ Core (Mémoire longue)")
    for row in champion_tau:
        print("".join(["░░" if val < 0.4 else "▒▒" if val < 0.7 else "██" for val in row]))

    np.savez_compressed("final_population_v8_11.npz", population=np.array([population[i] for i in np.argsort(compute_population_fitness(evals_final))[::-1]], dtype=object))

if __name__ == "__main__":
    run_evolution(pop_size=40, generations=15)