#!/usr/bin/env python3

"""
V8.14 — Predictive Fitness Evolution (version ROBUSTE)
Plus de warnings sklearn + R² stable et rapide
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
# PARAMÈTRES
# ===============================

GENS = 60
POP = 40

PRED_LAG = 12
VALID_STEPS = 200

W_LUC = 0.40
W_PRED = 0.60
Q_SCALE = 1e19

np.random.seed(42)


# ===============================
# CRÉATION + NORMALISATION
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


# ===============================
# CALCUL R2 PRÉDICTIF — VERSION ROBUSTE
# ===============================

def compute_predictive_r2(ind):
    try:
        res = engine.eval_individu_raw(
            ind,
            steps=VALID_STEPS,
            return_traces=True
        )

        if "hs" not in res or "envs" not in res:
            return 0.0

        hs = np.clip(np.array(res["hs"]), -100, 100)
        envs = np.clip(np.array(res["envs"]), -100, 100)

        T, nx, ny = hs.shape
        if T < PRED_LAG + 30:
            return 0.0

        X = hs[:-PRED_LAG].reshape(T - PRED_LAG, -1)
        Y = envs[PRED_LAG:].reshape(T - PRED_LAG, -1)

        # Standardisation forte pour éviter tout overflow
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X = scaler_x.fit_transform(X)
        Y = scaler_y.fit_transform(Y)

        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)

        nloc = min(100, Ytr.shape[1])
        idx = np.random.choice(Ytr.shape[1], nloc, replace=False)

        r2s = []
        for k in idx:
            ytr = Ytr[:, k]
            yte = Yte[:, k]

            if np.var(ytr) < 1e-6:          # seuil plus strict
                continue

            try:
                m = Ridge(alpha=50.0, fit_intercept=True)   # alpha élevé = très stable
                m.fit(Xtr, ytr)
                yp = m.predict(Xte)
                r2 = r2_score(yte, yp)
                r2s.append(max(-1.0, r2))
            except:
                continue

        mean_r2 = float(np.mean(r2s)) if r2s else 0.0
        return np.nan_to_num(mean_r2, nan=0.0, posinf=0.0, neginf=-1.0)

    except:
        return 0.0


# ===============================
# EVOLUTION
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.14 — Predictive Fitness (ROBUSTE)")
    print(f"   PRED_LAG={PRED_LAG} | VALID_STEPS={VALID_STEPS} | POP={POP}\n")

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

            evals.append((ind, luc, q, r2))
            r2_list.append(r2)
            if r2 > 0.01:
                success_r2 += 1

        lucs = np.array([e[1] for e in evals])
        qs = np.array([e[2] for e in evals])
        r2s = np.array(r2_list)

        luc_n = normalize(lucs)
        r2_n = normalize(r2s)
        q_scaled = qs * Q_SCALE

        fitness = W_LUC * (luc_n / (1 + q_scaled)) + W_PRED * r2_n

        order = np.argsort(fitness)[::-1]
        best_fit = fitness[order[0]]

        print(f"BestFit {best_fit:.4f} | "
              f"Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"Success R2>0.01: {success_r2}/{POP} | "
              f"Q {qs.mean():.2e}")

        # Sélection + mutation
        survivors = [copy.deepcopy(evals[i][0]) for i in order[:POP//2]]
        newpop = survivors.copy()

        while len(newpop) < POP:
            p = copy.deepcopy(survivors[np.random.randint(len(survivors))])
            if hasattr(engine, "mutate_individu"):
                p = engine.mutate_individu(p)
            elif hasattr(engine, "mutate"):
                p = engine.mutate(p)
            newpop.append(p)

        pop = newpop

        history.append({
            "gen": gen,
            "luc": float(lucs.mean()),
            "r2": float(r2s.mean()),
            "r2_max": float(r2s.max()),
            "q": float(qs.mean()),
            "best_fit": float(best_fit)
        })

    np.savez("final_population_v8_14.npz", population=pop, history=history)
    print("\n✅ V8.14 TERMINÉ (sans aucun warning)")
    print("   Fichier : final_population_v8_14.npz")


if __name__ == "__main__":
    run_evolution()