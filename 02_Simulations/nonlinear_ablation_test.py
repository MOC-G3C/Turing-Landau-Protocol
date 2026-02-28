import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ========= PARAMÈTRES =========

GRID = 16
N = GRID*GRID

STEPS = 700
PRED_LAG = 20

CLIP_STATE = 50.0


# ========= CHARGEMENT =========

pop = np.load("final_population_v8_15.npz",allow_pickle=True)

champion = pop["population"][0]


# ========= ENVIRONNEMENT =========

def generate_environment(steps):

    E = np.zeros(steps)

    val = 0

    for t in range(steps):

        val += np.random.normal(0,0.15)

        if np.random.rand() < 0.02:
            val += np.random.normal(2,0.5)

        val *= 0.95

        E[t] = val

    return E


# ========= PARAMÈTRES AGENT =========

def flatten_tau(x):

    x = np.array(x)

    if x.ndim == 2:
        x = x.reshape(-1)

    return np.clip(x,0.05,5.0)


def get_tau_fast(agent):

    if "tau_fast" in agent:
        return flatten_tau(agent["tau_fast"])

    return np.ones(N)*0.3


def get_tau_slow(agent):

    if "tau_slow" in agent:
        return flatten_tau(agent["tau_slow"])

    return np.ones(N)*0.8


def get_gain(agent):

    if "gain" in agent:
        return float(agent["gain"])

    if "w_fast" in agent:
        return float(np.mean(agent["w_fast"]))

    return 1.0


# ========= SIMULATION =========

def simulate(agent,fast_off=False,slow_off=False):

    tau_fast = get_tau_fast(agent)
    tau_slow = get_tau_slow(agent)

    gain = get_gain(agent)

    E = generate_environment(STEPS)


    state_fast = np.zeros(N)
    state_slow = np.zeros(N)

    cortex = np.zeros((STEPS,N))


    for t in range(STEPS):

        stim = gain * E[t]


        if not fast_off:

            state_fast += stim
            state_fast -= state_fast/tau_fast


        if not slow_off:

            state_slow += stim
            state_slow -= state_slow/tau_slow


        state_fast = np.clip(state_fast,-CLIP_STATE,CLIP_STATE)
        state_slow = np.clip(state_slow,-CLIP_STATE,CLIP_STATE)


        cortex[t] = state_fast + state_slow


    return cortex,E


# ========= R2 NON LINÉAIRE =========

def predictive_R2(agent,fast_off=False,slow_off=False):

    cortex,E = simulate(agent,fast_off,slow_off)


    X = cortex[:-PRED_LAG]
    Y = E[PRED_LAG:]


    split = int(0.7*len(X))


    Xtrain = X[:split]
    Ytrain = Y[:split]

    Xtest = X[split:]
    Ytest = Y[split:]


    Xtrain = np.nan_to_num(Xtrain)
    Xtest = np.nan_to_num(Xtest)


    model = RandomForestRegressor(

        n_estimators=150,
        max_depth=12,
        n_jobs=-1
    )


    model.fit(Xtrain,Ytrain)

    pred = model.predict(Xtest)


    var = np.var(Ytest)

    if var < 1e-8:
        return 0


    return 1 - np.mean((pred-Ytest)**2)/var


# ========= TEST =========

print("\nBaseline RF R2")

baseline = predictive_R2(champion)

print("R2 =",baseline)



print("\nFAST OFF")

fastoff = predictive_R2(champion,fast_off=True)

print("R2 =",fastoff)



print("\nSLOW OFF")

slowoff = predictive_R2(champion,slow_off=True)

print("R2 =",slowoff)



print("\nSUMMARY\n")

print("Baseline :",baseline)
print("FastOff  :",fastoff)
print("SlowOff  :",slowoff)

print()

print("Loss Fast =",baseline-fastoff)
print("Loss Slow =",baseline-slowoff)