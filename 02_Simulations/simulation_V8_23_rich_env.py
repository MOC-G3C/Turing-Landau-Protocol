#!/usr/bin/env python3

"""
V8.23 — Direction A : Environnement Riche (V8.13b)
Moteur : simulation_V8_13b_rich_env.py
  - Hotspots mobiles (3 sources orbitales)
  - Triple canal mémoire (fast / slow / ultra-slow)
  - Couplage non-linéaire tanh(h_f × h_s)
  - Turbulence spatiale AR(1)
  - Cross-talk global (événements rares)

Objectif : R2_avg > 0.40 (vs 0.33 plateau V8.22)
Fitness : 13% Luc + 87% R² + DivBonus − CompPen
"""

import numpy as np
import copy
import warnings
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import simulation_V8_13b_rich_env as engine

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===============================
# PARAMÈTRES V8.23
# ===============================

GENS       = 200
POP        = 40

PRED_LAGS  = [90, 70, 50, 30, 15, 5]
VALID_STEPS = 500

W_LUC             = 0.13
W_PRED            = 0.87
Q_SCALE           = 1e19
COMPLEXITY_PENALTY = 0.15
DIVERSITY_BONUS    = 0.12
EXPLORATION_REWARD = 0.14

BASE_MUT_RATE = 0.05
MUT_DECAY     = 0.0004
BASE_ADAPT_NOISE = 0.05
NOISE_INCREASE   = 0.0008

META_BOUNDS = engine.META_BOUNDS   # synchronisé avec V8.13b

np.random.seed(42)


# ===============================
# MIGRATION SEED V8.22 → V8.23
# Ajoute les champs manquants aux anciens individus
# ===============================

def _upgrade_individual(ind):
    """Ajoute les champs V8.13b manquants à un individu V8.22."""
    ind = copy.deepcopy(ind)

    # Nouveaux champs spatiaux
    if "tau_ultra" not in ind:
        ind["tau_ultra"]   = np.random.uniform(0.990, 0.999, (16, 16))
    if "w_ultra_mat" not in ind:
        ind["w_ultra_mat"] = np.random.uniform(0.0, 0.05, (16, 16))  # départ conservateur

    # Nouveaux méta-paramètres
    if "nl_gain" not in ind["meta"]:
        ind["meta"]["nl_gain"]          = np.random.uniform(0.0, 0.1)
    if "cross_sensitivity" not in ind["meta"]:
        ind["meta"]["cross_sensitivity"] = np.random.uniform(0.0, 0.3)

    return ind


def load_population():
    """Charge V8.22 avec fallback, puis upgrade tous les individus."""
    for fname in [
        "final_population_v8_22.npz",
        "final_population_v8_21.npz",
        "final_population_v8_20.npz",
    ]:
        try:
            data = np.load(fname, allow_pickle=True)
            raw  = list(data["population"])
            pop  = [_upgrade_individual(ind) for ind in raw]
            print(f"   ✅ Seed chargé + migré : {fname} ({len(pop)} individus)")
            return pop
        except FileNotFoundError:
            continue

    print("   ⚠️  Aucun seed trouvé → nouvelle population V8.13b")
    return [engine.creer_individu_aleatoire() for _ in range(POP)]


def create_population(pop_size):
    pop = load_population()
    # Compléter si besoin avec de nouveaux individus
    while len(pop) < pop_size:
        pop.append(engine.creer_individu_aleatoire())
    return pop[:pop_size]


# ===============================
# UTILITAIRES
# ===============================

def normalize(v):
    v = np.array(v, dtype=float)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def complexity_penalty(ind):
    tau_var = np.var(ind["tau_fast"]) + np.var(ind["tau_slow"]) + np.var(ind["tau_ultra"])
    w_var   = np.var(ind["w_fast_mat"]) + np.var(ind["w_ultra_mat"])
    return (tau_var + w_var) * COMPLEXITY_PENALTY


def individual_distance(a, b):
    d  = np.linalg.norm(a["tau_fast"]   - b["tau_fast"])
    d += np.linalg.norm(a["tau_slow"]   - b["tau_slow"])
    d += np.linalg.norm(a["tau_ultra"]  - b["tau_ultra"])
    d += np.linalg.norm(a["w_fast_mat"] - b["w_fast_mat"])
    for k in a["meta"]:
        if k in b["meta"]:
            d += abs(a["meta"][k] - b["meta"][k])
    return d


def diversity_bonus(pop):
    dists = [
        individual_distance(pop[i], pop[j])
        for i in range(len(pop))
        for j in range(i + 1, len(pop))
    ]
    return np.mean(dists) * DIVERSITY_BONUS if dists else 0.0


def crossover(p1, p2):
    child = copy.deepcopy(p1)
    alpha = np.random.uniform(0.3, 0.7)
    for k in ["tau_fast", "tau_slow", "tau_ultra", "w_fast_mat", "w_ultra_mat"]:
        child[k] = alpha * p1[k] + (1 - alpha) * p2[k]
    for k in p1["meta"]:
        a = np.random.uniform(0.3, 0.7)
        if k in p2["meta"]:
            child["meta"][k] = a * p1["meta"][k] + (1 - a) * p2["meta"][k]
    return child


def adaptive_mutate(child, gen):
    rate = max(0.01, BASE_MUT_RATE * (1 - MUT_DECAY * gen))
    child["w_fast_mat"]  = np.clip(child["w_fast_mat"]  + np.random.randn(16,16)*rate,      0.0,  1.0)
    child["w_ultra_mat"] = np.clip(child["w_ultra_mat"] + np.random.randn(16,16)*rate*0.5,  0.0,  0.5)
    child["tau_fast"]    = np.clip(child["tau_fast"]    + np.random.randn(16,16)*rate*0.4,  0.15, 0.40)
    child["tau_slow"]    = np.clip(child["tau_slow"]    + np.random.randn(16,16)*rate*0.4,  0.70, 0.98)
    child["tau_ultra"]   = np.clip(child["tau_ultra"]   + np.random.randn(16,16)*rate*0.05, 0.990, 0.999)
    for k, (lo, hi) in META_BOUNDS.items():
        child["meta"][k] = float(np.clip(
            child["meta"][k] * (1.0 + np.random.randn() * 0.07), lo, hi
        ))
    return child


# ===============================
# FITNESS PRÉDICTIVE (R² Ridge)
# ===============================

def compute_predictive_r2(ind, gen):
    try:
        res = engine.eval_individu_raw(ind, steps=VALID_STEPS, return_traces=True)
        if "hs" not in res or "envs" not in res:
            return 0.0

        hs   = np.clip(np.array(res["hs"]),   -100, 100)
        envs = np.clip(np.array(res["envs"]), -100, 100)

        T = hs.shape[0]
        if T < max(PRED_LAGS) + 80:
            return 0.0

        adapt_noise = BASE_ADAPT_NOISE + NOISE_INCREASE * gen
        r2s_all = []

        for lag in PRED_LAGS:
            X = hs[:-lag].reshape(T - lag, -1)
            Y = envs[lag:].reshape(T - lag, -1)

            sx, sy = StandardScaler(), StandardScaler()
            Xn = sx.fit_transform(X)
            Yn = sy.fit_transform(Y)

            Ya = sy.fit_transform(Y + np.random.normal(0, adapt_noise, Y.shape))

            for Yv in [Yn, Ya]:
                Xtr, Xte, Ytr, Yte = train_test_split(
                    Xn, Yv, test_size=0.3, random_state=0
                )
                nloc = min(100, Ytr.shape[1])
                idx  = np.random.choice(Ytr.shape[1], nloc, replace=False)

                r2_local = []
                for k in idx:
                    ytr, yte = Ytr[:, k], Yte[:, k]
                    if np.var(ytr) < 1e-6:
                        continue
                    try:
                        m  = Ridge(alpha=120.0)
                        m.fit(Xtr, ytr)
                        yp = m.predict(Xte)
                        r2_local.append(max(-1.0, r2_score(yte, yp)))
                    except Exception:
                        continue
                r2s_all.append(float(np.mean(r2_local)) if r2_local else 0.0)

        if not r2s_all:
            return 0.0

        mean_r2  = float(np.mean(r2s_all))
        long_r2  = r2s_all[0]                    # lag=90
        short_r2 = float(np.mean(r2s_all[-3:]))  # lags 15,5
        exp_bonus = EXPLORATION_REWARD if long_r2 > short_r2 * 1.08 else 0.0

        return mean_r2 + exp_bonus

    except Exception:
        return 0.0


# ===============================
# BOUCLE ÉVOLUTIVE
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.23 — DIRECTION A : ENVIRONNEMENT RICHE")
    print(f"   Moteur    : V8.13b (hotspots mobiles, triple mémoire, NL, turbulence)")
    print(f"   PRED_LAGS : {PRED_LAGS} | VALID_STEPS={VALID_STEPS}")
    print(f"   GENS={GENS} | POP={POP}\n")

    pop     = create_population(POP)
    history = []

    for gen in range(GENS):
        evals    = []
        r2_list  = []
        success  = 0

        print(f"--- Gen {gen:03d} -----------------------------------")

        for ind in pop:
            res  = engine.eval_individu_raw(ind)
            luc  = res["luc_avg"]
            q    = res["q_avg"]
            r2   = compute_predictive_r2(ind, gen)
            comp = complexity_penalty(ind)

            evals.append((ind, luc, q, r2, comp))
            r2_list.append(r2)
            if r2 > 0.01:
                success += 1

        lucs  = np.array([e[1] for e in evals])
        qs    = np.array([e[2] for e in evals])
        r2s   = np.array(r2_list)
        comps = np.array([e[4] for e in evals])

        luc_n    = normalize(lucs)
        r2_n     = normalize(r2s)
        q_scaled = qs * Q_SCALE
        div_b    = diversity_bonus([e[0] for e in evals])

        fitness  = (W_LUC * (luc_n / (1 + q_scaled))
                    + W_PRED * r2_n
                    - comps
                    + div_b)

        order    = np.argsort(fitness)[::-1]
        best_fit = fitness[order[0]]

        print(f"BestFit {best_fit:.4f} | Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"OK {success}/{POP} | Div {div_b:.4f} | Q {qs.mean():.2e}")

        # Sauvegarde Top-5 tous les 10 gens
        if gen % 10 == 0 and gen > 0:
            top5 = [evals[i][0] for i in order[:5]]
            np.savez(f"top5_v8_23_gen_{gen:03d}.npz", top5=top5)
            print(f"   💾 Top-5 : top5_v8_23_gen_{gen:03d}.npz")

        # Immigration tous les 30 gens
        if gen % 30 == 0 and gen > 0:
            immigrants = [engine.creer_individu_aleatoire() for _ in range(4)]
            for i in range(4):
                pop[order[-1 - i]] = immigrants[i]
            print(f"   🌍 Immigration : 4 nouveaux individus V8.13b")

        # Sélection + élitisme 3 + crossover 22%
        survivors = [copy.deepcopy(evals[i][0]) for i in order[:POP // 2]]
        newpop    = [copy.deepcopy(evals[i][0]) for i in order[:3]]

        while len(newpop) < POP:
            if np.random.rand() < 0.22:
                p1    = survivors[np.random.randint(len(survivors))]
                p2    = survivors[np.random.randint(len(survivors))]
                child = crossover(p1, p2)
            else:
                child = copy.deepcopy(survivors[np.random.randint(len(survivors))])
            child = adaptive_mutate(child, gen)
            newpop.append(child)

        pop = newpop

        history.append({
            "gen":      gen,
            "luc":      float(lucs.mean()),
            "r2":       float(r2s.mean()),
            "r2_max":   float(r2s.max()),
            "div":      div_b,
            "comp":     float(comps.mean()),
            "best_fit": float(best_fit),
        })

    np.savez("final_population_v8_23.npz", population=pop, history=history)
    print("\n✅ V8.23 TERMINÉ")
    print("   Fichier : final_population_v8_23.npz")
    r2_final = np.mean([h["r2"] for h in history[-10:]])
    r2_debut = np.mean([h["r2"] for h in history[:10]])
    print(f"   R2_avg gen 0-9   : {r2_debut:.4f}")
    print(f"   R2_avg gen 190+  : {r2_final:.4f}")
    print(f"   Delta R2         : {r2_final - r2_debut:+.4f}")


if __name__ == "__main__":
    run_evolution()
