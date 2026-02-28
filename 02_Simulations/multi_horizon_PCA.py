import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# ==============================
# PARAMÈTRES
# ==============================

GRID = 16
N = GRID*GRID

STEPS = 900

LAGS = [5,10,20,40,80]

PCA_DIM = 20

CLIP_STATE = 50.0


# ==============================
# CHARGEMENT POPULATION
# ==============================

pop = np.load("final_population_v8_15.npz",allow_pickle=True)

champion = pop["population"][0]


# ==============================
# ENVIRONNEMENT
# ==============================

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


# ==============================
# EXTRACTION PARAMÈTRES AGENT
# ==============================

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


# ==============================
# SIMULATION CORTEX
# ==============================

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


        # stabilisation physique

        state_fast = np.clip(state_fast,-CLIP_STATE,CLIP_STATE)
        state_slow = np.clip(state_slow,-CLIP_STATE,CLIP_STATE)


        cortex[t] = state_fast + state_slow


    return cortex,E


# ==============================
# R2 AVEC PCA STABLE
# ==============================

def predictive_R2(agent,lag,fast_off=False,slow_off=False):

    cortex,E = simulate(agent,fast_off,slow_off)

    X = cortex[:-lag]
    Y = E[lag:]


    split = int(0.7*len(X))


    Xtrain = X[:split]
    Ytrain = Y[:split]

    Xtest = X[split:]
    Ytest = Y[split:]


    # ==========================
    # NORMALISATION STABLE
    # ==========================

    mean = np.mean(Xtrain,axis=0)
    std = np.std(Xtrain,axis=0)

    mask = std > 1e-4

    Xtrain = (Xtrain[:,mask]-mean[mask])/std[mask]
    Xtest = (Xtest[:,mask]-mean[mask])/std[mask]


    Xtrain = np.nan_to_num(Xtrain)
    Xtest = np.nan_to_num(Xtest)


    if Xtrain.shape[1] == 0:
        return 0


    # ==========================
    # PCA STABLE
    # ==========================

    dim = min(PCA_DIM,Xtrain.shape[1])

    pca = PCA(n_components=dim)

    Xtrain = pca.fit_transform(Xtrain)
    Xtest = pca.transform(Xtest)


    # ==========================
    # RANDOM FOREST
    # ==========================

    model = RandomForestRegressor(

        n_estimators=200,
        max_depth=10,
        n_jobs=-1

    )


    model.fit(Xtrain,Ytrain)

    pred = model.predict(Xtest)


    var = np.var(Ytest)

    if var < 1e-8:
        return 0


    return 1 - np.mean((pred-Ytest)**2)/var


# ==============================
# TEST MULTI-HORIZON
# ==============================

print("\nMULTI-HORIZON PCA TEST\n")

print("Lag    Base      FastOff     SlowOff")

for lag in LAGS:

    base = predictive_R2(champion,lag)

    fast = predictive_R2(champion,lag,fast_off=True)

    slow = predictive_R2(champion,lag,slow_off=True)


    print(
        f"{lag:3d}   "
        f"{base: .3f}   "
        f"{fast: .3f}   "
        f"{slow: .3f}"
    )