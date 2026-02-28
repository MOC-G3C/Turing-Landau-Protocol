#!/usr/bin/env python3

"""
V8.21 — Predictive Fitness Evolution (Hexa Memory + Reinforced Exploration)
Fitness = 13% Luc + 87% R² + Diversity + Strong Exploration Reward
"""

import numpy as np
import copy
import warnings
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import simulation_V8_13_spatial_mix as engine

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===============================
# PARAMÈTRES V8.21
# ===============================

GENS = 180
POP = 40

PRED_LAGS = [90, 70, 50, 30, 15, 5]   # HEXA memory
BASE_ADAPT_NOISE = 0.05
NOISE_INCREASE = 0.0008
VALID_STEPS = 500

W_LUC = 0.13
W_PRED = 0.87
Q_SCALE = 1e19
COMPLEXITY_PENALTY = 0.15
DIVERSITY_BONUS = 0.10
EXPLORATION_REWARD = 0.12

BASE_MUT_RATE = 0.05
MUT_DECAY = 0.0005

# META_BOUNDS (copié de V8.13)
META_BOUNDS = {
    "kappa_gain":     (0.01, 2.0),
    "mutation_rate":  (0.01, 0.30),
    "env_gain_spike": (0.005, 0.20),
    "env_gain_calm":  (0.0001, 0.02),
}

np.random.seed(42)


# ===============================
# FONCTIONS UTILITAIRES
# ===============================

def load_previous_population(file="final_population_v8_20.npz"):
    data = np.load(file, allow_pickle=True)
    return list(data['population'])


def create_population(pop_size):
    seed_pop = load_previous_population()
    print(f"   ✅ Seed chargé : final_population_v8_20.npz ({len(seed_pop)} individus)")
    return seed_pop[:pop_size]


def normalize(v):
    v = np.array(v)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def complexity_penalty(ind):
    tau_var = np.var(ind["tau_fast"]) + np.var(ind["tau_slow"])
    w_var = np.var(ind["w_fast_mat"])
    return (tau_var + w_var) * COMPLEXITY_PENALTY


def individual_distance(ind1, ind2):
    dist = 0.0
    dist += np.linalg.norm(ind1["tau_fast"] - ind2["tau_fast"])
    dist += np.linalg.norm(ind1["tau_slow"] - ind2["tau_slow"])
    dist += np.linalg.norm(ind1["w_fast_mat"] - ind2["w_fast_mat"])
    for k in ind1["meta"]:
        dist += abs(ind1["meta"][k] - ind2["meta"][k])
    return dist


def diversity_bonus(pop):
    dists = []
    for i in range(len(pop)):
        for j in range(i+1, len(pop)):
            dists.append(individual_distance(pop[i], pop[j]))
    return np.mean(dists) * DIVERSITY_BONUS if dists else 0.0


def crossover_meta(p1, p2):
    child = copy.deepcopy(p1)
    for k in p1["meta"]:
        alpha = np.random.uniform(0.3, 0.7)
        child["meta"][k] = alpha * p1["meta"][k] + (1 - alpha) * p2["meta"][k]
    return child


def adaptive_mutate(child, gen):
    mut_rate = BASE_MUT_RATE * (1 - MUT_DECAY * gen)
    child["w_fast_mat"] = np.clip(child["w_fast_mat"] + np.random.randn(16,16)*mut_rate, 0.0, 1.0)
    child["tau_fast"]   = np.clip(child["tau_fast"]   + np.random.randn(16,16)*mut_rate*0.4, 0.15, 0.40)
    child["tau_slow"]   = np.clip(child["tau_slow"]   + np.random.randn(16,16)*mut_rate*0.4, 0.70, 0.98)
    for k, (lo, hi) in META_BOUNDS.items():
        child["meta"][k] = float(np.clip(child["meta"][k] + np.random.randn()*mut_rate, lo, hi))
    return child


# ===============================
# CALCUL R2 (Hexa + Adaptivity + Exploration Reward)
# ===============================

def compute_predictive_r2(ind, gen):
    try:
        res = engine.eval_individu_raw(ind, steps=VALID_STEPS, return_traces=True)
        if "hs" not in res or "envs" not in res:
            return 0.0

        hs = np.clip(np.array(res["hs"]), -100, 100)
        envs = np.clip(np.array(res["envs"]), -100, 100)

        T = hs.shape[0]
        if T < max(PRED_LAGS) + 80:
            return 0.0

        r2s_all = []
        adapt_noise = BASE_ADAPT_NOISE + NOISE_INCREASE * gen

        for lag in PRED_LAGS:
            X = hs[:-lag].reshape(T - lag, -1)
            Y = envs[lag:].reshape(T - lag, -1)

            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            Xn = scaler_x.fit_transform(X)
            Yn = scaler_y.fit_transform(Y)

            Ya = Y + np.random.normal(0, adapt_noise, Y.shape)
            Ya = scaler_y.fit_transform(Ya)

            for Yv in [Yn, Ya]:
                Xtr, Xte, Ytr, Yte = train_test_split(Xn, Yv, test_size=0.3, random_state=0)
                nloc = min(100, Ytr.shape[1])
                idx = np.random.choice(Ytr.shape[1], nloc, replace=False)

                r2_local = []
                for k in idx:
                    ytr = Ytr[:, k]
                    yte = Yte[:, k]
                    if np.var(ytr) < 1e-6:
                        continue
                    try:
                        m = Ridge(alpha=120.0, fit_intercept=True)
                        m.fit(Xtr, ytr)
                        yp = m.predict(Xte)
                        r2_local.append(max(-1.0, r2_score(yte, yp)))
                    except:
                        continue
                r2s_all.append(float(np.mean(r2_local)) if r2_local else 0.0)

        mean_r2 = float(np.mean(r2s_all))

        # Exploration reward renforcé
        long_r2 = r2s_all[0]                    # lag=90
        short_avg = np.mean(r2s_all[-3:])       # 3 lags les plus courts
        exp_bonus = EXPLORATION_REWARD if long_r2 > short_avg * 1.05 else 0.0

        return mean_r2 + exp_bonus

    except:
        return 0.0


# ===============================
# EVOLUTION V8.21
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.21 — HEXA MEMORY + REINFORCED EXPLORATION")
    print(f"   PRED_LAGS={PRED_LAGS} | ADAPT_NOISE croissant | Seed V8.20\n")

    pop = create_population(POP)
    history = []

    for gen in range(GENS):
        evals = []
        r2_list = []
        success_r2 = 0

        print(f"--- Gen {gen:02d} -----------------------------------")

        for ind in pop:
            res = engine.eval_individu_raw(ind)
            luc = res["luc_avg"]
            q = res["q_avg"]
            r2 = compute_predictive_r2(ind, gen)
            comp_pen = complexity_penalty(ind)

            evals.append((ind, luc, q, r2, comp_pen))
            r2_list.append(r2)
            if r2 > 0.01:
                success_r2 += 1

        lucs = np.array([e[1] for e in evals])
        qs = np.array([e[2] for e in evals])
        r2s = np.array(r2_list)
        comps = np.array([e[4] for e in evals])

        luc_n = normalize(lucs)
        r2_n = normalize(r2s)
        q_scaled = qs * Q_SCALE

        div_bonus = diversity_bonus([e[0] for e in evals])

        fitness = (W_LUC * (luc_n / (1 + q_scaled))) + (W_PRED * r2_n) - comps + div_bonus

        order = np.argsort(fitness)[::-1]
        best_fit = fitness[order[0]]

        print(f"BestFit {best_fit:.4f} | Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"Success R2>0.01: {success_r2}/{POP} | "
              f"DivBonus {div_bonus:.4f} | CompPen {comps.mean():.4f} | Q {qs.mean():.2e}")

        # Sauvegarde Top-5 tous les 15 gens
        if gen % 15 == 0 and gen > 0:
            top5 = [evals[i][0] for i in order[:5]]
            np.savez(f"top5_gen_{gen:02d}.npz", top5=top5)
            print(f"   💾 Top-5 sauvegardés : top5_gen_{gen:02d}.npz")

        # Sélection + Elite 5 + Crossover
        survivors = [copy.deepcopy(evals[i][0]) for i in order[:POP//2]]
        newpop = [copy.deepcopy(evals[i][0]) for i in order[:5]]

        while len(newpop) < POP:
            if np.random.rand() < 0.35:
                p1 = survivors[np.random.randint(len(survivors))]
                p2 = survivors[np.random.randint(len(survivors))]
                child = copy.deepcopy(p1)
                alpha = np.random.uniform(0.3, 0.7)
                child["tau_fast"] = alpha * p1["tau_fast"] + (1-alpha) * p2["tau_fast"]
                child["tau_slow"] = alpha * p1["tau_slow"] + (1-alpha) * p2["tau_slow"]
                child["w_fast_mat"] = alpha * p1["w_fast_mat"] + (1-alpha) * p2["w_fast_mat"]
                child = crossover_meta(child, p2)
            else:
                child = copy.deepcopy(survivors[np.random.randint(len(survivors))])

            child = adaptive_mutate(child, gen)
            newpop.append(child)

        pop = newpop

        history.append({
            "gen": gen,
            "luc": float(lucs.mean()),
            "r2": float(r2s.mean()),
            "r2_max": float(r2s.max()),
            "div_bonus": div_bonus,
            "comp": float(comps.mean()),
            "best_fit": float(best_fit)
        })

    np.savez("final_population_v8_21.npz", population=pop, history=history)
    print("\n✅ V8.21 TERMINÉ")
    print("   Fichier final : final_population_v8_21.npz")
    print(f"   Meilleur R2 final = {r2s.max():.4f}")


if __name__ == "__main__":
    run_evolution()