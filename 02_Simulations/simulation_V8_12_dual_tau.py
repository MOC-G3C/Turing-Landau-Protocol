import numpy as np
import copy
import json
import zlib
import math
from scipy.stats import entropy
from collections import deque

# ============================================================
# 1. CONSTANTES V8.12 (ARCHITECTURE DUAL-CHANNEL)
# ============================================================
np.random.seed(42)

kB    = 1.380649e-23
T_env = 300.0
LN2   = math.log(2.0)

SPIKE_PROB_FAST = 0.12  
SPIKE_PROB_SLOW = 0.02
INFO_TO_BITS    = 0.2         
Q_MAX_LANDAUER  = 8e-19     # Un peu plus de marge pour le double canal

SEUIL_REACTION = 0.08      
MAX_INACTION   = 3           

META_BOUNDS = {
    "kappa_gain":       (0.01, 2.0),
    "cooling_strength": (0.30, 0.95),
    "mutation_rate":    (0.01, 0.30),
    "target_base":      (0.005, 0.15),
    "low_var_thresh":   (0.003, 0.05),
    "env_gain_spike":   (0.0005, 0.15),
    "env_gain_calm":    (0.0001, 0.02),
    "w_fast":           (0.2, 0.8), # Poids accordé au canal rapide
}

# Hotspots V8.11.1 conservés
HOTSPOTS = [(3, 3, 10.0, 0.0), (12, 12, 8.0, np.pi/2)]
XS = np.arange(16)[:, None]
YS = np.arange(16)[None, :]

# ============================================================
# 2. FONCTIONS DE BASE (Adaptées NumPy 2.0)
# ============================================================
def normalize_gen(values):
    v = np.array(values)
    if np.ptp(v) == 0: return np.zeros_like(v)
    return (v - np.min(v)) / (np.ptp(v) + 1e-9)

def creer_individu_aleatoire(shape=(16, 16)):
    return {
        "theta": np.random.randn(*shape) * 0.1,
        "tau_fast": np.random.uniform(0.15, 0.40, shape),
        "tau_slow": np.random.uniform(0.70, 0.98, shape),
        "meta": {k: np.random.uniform(v[0], v[1]) for k, v in META_BOUNDS.items()}
    }

# ============================================================
# 3. MOTEUR DE SIMULATION DUAL-CHANNEL
# ============================================================
def eval_individu_raw(adn, steps=200):
    trials = []
    for _ in range(3):
        theta = adn["theta"].copy()
        t_fast, t_slow = adn["tau_fast"], adn["tau_slow"]
        w_f = adn["meta"]["w_fast"]
        w_s = 1.0 - w_f
        
        h_f = np.zeros_like(theta)
        h_s = np.zeros_like(theta)
        
        Q_heat = 0.0
        lucidity = 0.0
        inaction = 0.0
        theta_means = []
        
        for t in range(1, steps + 1):
            # Environnement spatial
            env_field = np.zeros_like(theta)
            f_comp = (1.0 if np.random.rand() < SPIKE_PROB_FAST else 0.1) * np.sin(t / 8.0)
            s_comp = (1.0 if np.random.rand() < SPIKE_PROB_SLOW else 0.1) * np.sin(t / 200.0)
            for (x, y, amp, phase) in HOTSPOTS:
                dist2 = (XS - x)**2 + (YS - y)**2
                env_field += amp * np.exp(-dist2 / 9.0) * (f_comp + s_comp + np.sin(t/10.0 + phase)*0.2)
            
            # Mise à jour des deux canaux de mémoire
            h_f = t_fast * h_f + (1.0 - t_fast) * env_field
            h_s = t_slow * h_s + (1.0 - t_slow) * env_field
            
            # Signal combiné
            h_comb = (w_f * h_f) + (w_s * h_s)
            
            # Dynamique de Theta
            var = float(np.var(theta))
            is_spike = float(np.max(np.abs(env_field))) > 4.0
            gain = adn["meta"]["env_gain_spike"] if is_spike else adn["meta"]["env_gain_calm"]
            
            theta += (np.random.randn(*theta.shape) * 0.01 + h_comb * gain)

            # --- Taxe de Landauer Double ---
            # Effacement canal Fast + Canal Slow + Theta
            erased = (theta * (1.0 - 0.5)) # Simplifié pour la V8.12
            Q_heat += kB * T_env * LN2 * np.sum(np.abs(erased)) * INFO_TO_BITS
            theta *= 0.8 # Relaxation

            theta_means.append(float(np.mean(np.abs(theta))))
            
            # Lucidity
            if is_spike:
                if np.max(np.abs(theta)) < SEUIL_REACTION: inaction += 1
                else: lucidity += 1

        trials.append({"luc": lucidity, "ina": inaction, "q": Q_heat, "surv": 1.0 if Q_heat < Q_MAX_LANDAUER else 0.5})

    return {
        "luc_avg": np.mean([t["luc"] for t in trials]),
        "q_avg": np.mean([t["q"] for t in trials]),
        "ina_avg": np.mean([t["ina"] for t in trials]),
        "surv_avg": np.mean([t["surv"] for t in trials])
    }

def compute_population_fitness(evals):
    luc_n = normalize_gen([e["luc_avg"] for e in evals])
    q_n = normalize_gen([e["q_avg"] for e in evals])
    surv = np.array([e["surv_avg"] for e in evals])
    
    # On veut maximiser la lucidité et minimiser la chaleur
    return (0.5 * luc_n + 0.3 * (1.0 - q_n) + 0.2 * surv)

# ============================================================
# 4. RUN EVOLUTION
# ============================================================
def run_evolution(pop_size=40, generations=20):
    population = [creer_individu_aleatoire() for _ in range(pop_size)]
    for gen in range(generations):
        evals = [eval_individu_raw(ind) for ind in population]
        fits = compute_population_fitness(evals)
        
        idx = np.argsort(fits)[::-1]
        print(f"Gen {gen:02d} | Luc: {np.mean([e['luc_avg'] for e in evals]):.2f} | Q: {np.mean([e['q_avg'] for e in evals]):.2e} | Best: {fits[idx[0]]:.4f}")
        
        # Sélection et Mutation (élitisme 10%)
        next_pop = [population[i] for i in idx[:4]]
        while len(next_pop) < pop_size:
            p = population[np.random.choice(idx[:10])]
            child = copy.deepcopy(p)
            child["tau_fast"] = np.clip(child["tau_fast"] + np.random.randn(16,16)*0.02, 0.15, 0.40)
            child["tau_slow"] = np.clip(child["tau_slow"] + np.random.randn(16,16)*0.02, 0.70, 0.98)
            next_pop.append(child)
        population = next_pop

    # Sauvegarde
    np.savez_compressed("final_population_v8_12.npz", population=np.array(population, dtype=object))
    print("\n💾 Population V8.12 (Dual-Tau) sauvegardée.")

if __name__ == "__main__":
    run_evolution()