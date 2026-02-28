#!/usr/bin/env python3

"""
V8.15 — Predictive Fitness Evolution (Boost Prédiction + Complexité)
Fitness = 30% Luc + 70% R² + Pénalité complexité
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
# PARAMÈTRES V8.15
# ===============================

GENS = 80          # un peu plus long pour voir l’effet du PRED_LAG=20
POP = 40

PRED_LAG = 20
VALID_STEPS = 250

W_LUC = 0.30
W_PRED = 0.70
Q_SCALE = 1e19
COMPLEXITY_PENALTY = 0.08   # poids de la pénalité sur variance tau/w

np.random.seed(42)


# ===============================
# FONCTIONS UTILITAIRES
# ===============================

def create_population(pop_size):
    if hasattr(engine, "creer_population"):
        return engine.creer_population(pop_size)
    if hasattr(engine, "creer_individu"):
        return [engine.creer_individu() for _ in range(pop_size)]
    if hasattr(engine, "creer_individu_aleatoire"):
        return [engine.creer_individu_aleatoire() for _ in range(pop_size)]
    raise RuntimeError("Ajoute creer_* dans V8.13")


def normalize(v):
    v = np.array(v)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def complexity_penalty(ind):
    """Pénalité sur la variance des paramètres (évite sur-apprentissage)"""
    tau_var = np.var(ind["tau_fast"]) + np.var(ind["tau_slow"])
    w_var = np.var(ind["w_fast_mat"])
    return (tau_var + w_var) * COMPLEXITY_PENALTY


# ===============================
# CALCUL R2 PRÉDICTIF (robuste)
# ===============================

def compute_predictive_r2(ind):
    try:
        res = engine.eval_individu_raw(ind, steps=VALID_STEPS, return_traces=True)
        if "hs" not in res or "envs" not in res:
            return 0.0

        hs = np.clip(np.array(res["hs"]), -100, 100)
        envs = np.clip(np.array(res["envs"]), -100, 100)

        T = hs.shape[0]
        if T < PRED_LAG + 40:
            return 0.0

        X = hs[:-PRED_LAG].reshape(T - PRED_LAG, -1)
        Y = envs[PRED_LAG:].reshape(T - PRED_LAG, -1)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_x.fit_transform(X)
        Y = scaler_y.fit_transform(Y)

        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)

        nloc = min(120, Ytr.shape[1])
        idx = np.random.choice(Ytr.shape[1], nloc, replace=False)

        r2s = []
        for k in idx:
            ytr = Ytr[:, k]
            yte = Yte[:, k]
            if np.var(ytr) < 1e-6:
                continue
            try:
                m = Ridge(alpha=80.0, fit_intercept=True)
                m.fit(Xtr, ytr)
                yp = m.predict(Xte)
                r2s.append(max(-1.0, r2_score(yte, yp)))
            except:
                continue

        return float(np.mean(r2s)) if r2s else 0.0

    except:
        return 0.0


# ===============================
# EVOLUTION V8.15
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.15 — Predictive Fitness BOOSTÉ")
    print(f"   PRED_LAG={PRED_LAG} | VALID_STEPS={VALID_STEPS} | Elite5 + Crossover\n")

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
            r2 = compute_predictive_r2(ind)
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

        fitness = (W_LUC * (luc_n / (1 + q_scaled))) + \
                  (W_PRED * r2_n) - \
                  comps

        order = np.argsort(fitness)[::-1]
        best_fit = fitness[order[0]]

        print(f"BestFit {best_fit:.4f} | Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"Success R2>0.01: {success_r2}/{POP} | "
              f"CompPen {comps.mean():.4f} | Q {qs.mean():.2e}")

        # === Sélection + Elite 5 + Crossover ===
        survivors = [copy.deepcopy(evals[i][0]) for i in order[:POP//2]]

        newpop = [copy.deepcopy(evals[i][0]) for i in order[:5]]  # Elite 5 intacts

        while len(newpop) < POP:
            if np.random.rand() < 0.30:  # 30% chance de crossover
                p1 = survivors[np.random.randint(len(survivors))]
                p2 = survivors[np.random.randint(len(survivors))]
                child = copy.deepcopy(p1)
                # Crossover léger sur les matrices
                alpha = np.random.uniform(0.3, 0.7)
                child["tau_fast"] = alpha * p1["tau_fast"] + (1-alpha) * p2["tau_fast"]
                child["tau_slow"] = alpha * p1["tau_slow"] + (1-alpha) * p2["tau_slow"]
                child["w_fast_mat"] = alpha * p1["w_fast_mat"] + (1-alpha) * p2["w_fast_mat"]
            else:
                child = copy.deepcopy(survivors[np.random.randint(len(survivors))])

            # Mutation normale
            if hasattr(engine, "mutate_individu"):
                child = engine.mutate_individu(child)
            elif hasattr(engine, "mutate"):
                child = engine.mutate(child)

            newpop.append(child)

        pop = newpop

        history.append({
            "gen": gen,
            "luc": float(lucs.mean()),
            "r2": float(r2s.mean()),
            "r2_max": float(r2s.max()),
            "comp": float(comps.mean()),
            "best_fit": float(best_fit)
        })

    np.savez("final_population_v8_15.npz", population=pop, history=history)
    print("\n✅ V8.15 TERMINÉ")
    print("   Fichier : final_population_v8_15.npz")
    print(f"   Meilleur R2 final = {r2s.max():.4f}")


if __name__ == "__main__":
    run_evolution()