import numpy as np

POP_FILE = "final_population_v8_15.npz"

MAX_LAG = 40
VALID_STEPS = 400


# -------------------------
# ENVIRONNEMENT STRUCTURÉ STABLE

def generate_environment(T):

    E = np.random.normal(0,0.05,T)

    for t in range(20,T,40):

        width = np.random.randint(3,8)
        amp = np.random.uniform(0.8,1.5)

        shape = np.exp(-np.linspace(0,2,width))

        end = min(T,t+width)

        E[t:end] += amp*shape[:end-t]

    return E


# -------------------------
# R² ROBUSTE

def compute_R2(a,b):

    a = np.nan_to_num(a)
    b = np.nan_to_num(b)

    var = np.var(a)

    if var < 1e-12:
        return 0.0

    mse = np.mean((a-b)**2)

    if not np.isfinite(mse):
        return 0.0

    return 1 - mse/var


# -------------------------
# SIMULATION AGENT V8.15 STABLE

def simulate_agent(agent):

    E = generate_environment(VALID_STEPS)

    tau_fast = agent["tau_fast"]
    tau_slow = agent["tau_slow"]

    state_fast = np.zeros_like(tau_fast)
    state_slow = np.zeros_like(tau_slow)

    outputs = []

    for t in range(VALID_STEPS):

        inp = E[t]

        state_fast += (inp - state_fast) / tau_fast
        state_slow += (inp - state_slow) / tau_slow

        # Stabilisation numérique
        state_fast = np.clip(state_fast,-5,5)
        state_slow = np.clip(state_slow,-5,5)

        cortex = state_fast + state_slow

        outputs.append(np.mean(cortex))

    return np.array(outputs),E


# -------------------------
# COURBE R² vs LAG

def lag_curve(agent):

    outputs,E = simulate_agent(agent)

    R2s = []

    for lag in range(1,MAX_LAG):

        r2 = compute_R2(
            outputs[:-lag],
            E[lag:]
        )

        R2s.append(r2)

    return np.array(R2s)


# -------------------------
# CHARGEMENT CHAMPION

data = np.load(POP_FILE,allow_pickle=True)

champion = data["population"][0]


# -------------------------
# BASELINE

print("\nBaseline cortex...")

baseline = lag_curve(champion)


# -------------------------
# FAST OFF (ABLATION GLOBALE)

print("FAST OFF...")

fast_ab = champion.copy()

fast_ab["tau_fast"] = np.ones_like(
    champion["tau_fast"]
)*0.95

fast_curve = lag_curve(fast_ab)


# -------------------------
# SLOW OFF (ABLATION GLOBALE)

print("SLOW OFF...")

slow_ab = champion.copy()

slow_ab["tau_slow"] = np.ones_like(
    champion["tau_slow"]
)*0.2

slow_curve = lag_curve(slow_ab)


# -------------------------
# AFFICHAGE

print("\nLag    Base    FastOff    SlowOff\n")

for i in range(1,MAX_LAG):

    print(
        f"{i:2d}   "
        f"{baseline[i-1]:.3f}   "
        f"{fast_curve[i-1]:.3f}   "
        f"{slow_curve[i-1]:.3f}"
    )