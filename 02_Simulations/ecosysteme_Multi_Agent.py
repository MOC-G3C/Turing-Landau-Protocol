import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from scipy.io import wavfile

# -----------------------------
# 1. CONFIGURATION : L'AXE HYBRIDE (BIO-RYTHME)
# -----------------------------
num_agents = 6
L = 16
steps = 10000
dt = 0.01
thermal_capacity = 5000       
cooling_rate = 400            
CORE_DIR = "../03_Core"
DNA_FILE = "champion_dna.npz"

def cognitive_load(step):
    return 0.4 + 1.2 * np.sin(2 * np.pi * step / 2000)**2

def get_tesla_bonus(variance):
    v_scaled = variance * 100
    b3 = np.exp(-(v_scaled - 3)**2 / 0.5)
    b6 = np.exp(-(v_scaled - 6)**2 / 0.5)
    b9 = np.exp(-(v_scaled - 9)**2 / 0.5)
    return b3, b6, b9

# -----------------------------
# 2. INITIALISATION ET INJECTION (NAISSANCES)
# -----------------------------
agents = []
DNA_PATH = os.path.join(CORE_DIR, DNA_FILE)

# Création du dossier si absent
if not os.path.exists(CORE_DIR): os.makedirs(CORE_DIR)

print("🐣 Initialisation des entités numériques...")

# Injection du Champion (Naissance spéciale)
if os.path.exists(DNA_PATH):
    dna = np.load(DNA_PATH)
    agents.append({
        "id": 0, "Theta": dna['Theta'] + np.random.randn(L, L) * 0.01,
        "kappa": float(dna['kappa']), "heat": 0.0, "adapt": 0.0001,
        "color": "gold", "name": "CHAMPION_ELITE", "alive": True,
        "kappa_history": [], "heat_history": [], "var_history": [], 
        "resets": 0, "sound_data": [], "birth_step": 0, "death_step": None
    })

# Création des recrues (Naissances standard)
for i in range(len(agents), num_agents):
    agents.append({
        "id": i, "Theta": 0.001 * np.random.randn(L, L), "kappa": 1.0,
        "heat": 0.0, "adapt": 0.002, "color": "purple",
        "name": f"Recrue_{i}", "alive": True,
        "kappa_history": [], "heat_history": [], "var_history": [], 
        "resets": 0, "sound_data": [], "birth_step": 0, "death_step": None
    })

# -----------------------------
# 3. SIMULATION ET CAPTURE DU DESTIN
# -----------------------------
for step in range(steps):
    active = [a for a in agents if a["alive"]]
    if not active: break
    
    for a in active:
        Theta, kappa = a["Theta"], a["kappa"]
        Omega = cognitive_load(step)
        D = 0.05 + 0.1 * Omega
        
        lap = (np.roll(Theta,1,0) + np.roll(Theta,-1,0) + np.roll(Theta,1,1) + np.roll(Theta,-1,1) - 4*Theta)
        Theta -= 1.0 * (Theta**3 - kappa*lap - Omega) * dt + np.sqrt(2*D*dt) * np.random.randn(L,L)
        
        # Résonance Tesla 3-6-9
        b3, b6, b9 = get_tesla_bonus(np.var(Theta))
        t_total = max(b3, b6, b9)
        effective_cooling = cooling_rate * (1.0 + t_total)
        
        a["heat"] = max(0, a["heat"] + ((D + np.var(lap)) * (L**2 * 4.0) - effective_cooling) * dt)
        
        # Landauer Reset
        if a["heat"] > 4200:
            Theta *= 0.7 
            a["heat"] *= 0.5
            a["resets"] += 1
        
        # Mort Thermique
        if a["heat"] > thermal_capacity:
            a["alive"] = False
            a["death_step"] = step
            continue

        kappa = np.clip(kappa + a["adapt"] * (0.2 - np.var(Theta)), 0.5, 2.5)
        a.update({"Theta": Theta, "kappa": kappa})
        a["kappa_history"].append(kappa); a["heat_history"].append(a["heat"])
        a["var_history"].append(np.var(Theta))
        a["sound_data"].append([b3, b6, b9, (a["heat"] / thermal_capacity)**2])

# -----------------------------
# 4. GÉNÉRATION AUDIO (VIE, NAISSANCE, MORT)
# -----------------------------
print("🎵 Composition de la symphonie du cycle de vie...")
fs = 44100
step_dur = 0.001 
t_step = np.linspace(0, step_dur, int(fs * step_dur))
champion = agents[0]
audio_buffer = np.zeros(len(champion["sound_data"]) * len(t_step))

# Couche 1 : Drone de Vie du Champion
for s_idx, s_data in enumerate(champion["sound_data"]):
    b3, b6, b9, vol = s_data
    wave = vol * (b3 * np.sin(2 * np.pi * 396 * t_step) + 
                  b6 * np.sin(2 * np.pi * 639 * t_step) + 
                  b9 * np.sin(2 * np.pi * 963 * t_step))
    audio_buffer[s_idx*len(t_step) : (s_idx+1)*len(t_step)] = wave

# Couche 2 : Cris et Naissances
event_dur = 0.3 # 300ms
t_event = np.linspace(0, event_dur, int(fs * event_dur))

for a in agents:
    # Son de Naissance (Ascendant)
    if a["birth_step"] is not None:
        freq_up = np.linspace(100, 1200, len(t_event))
        birth_sound = 0.5 * np.sin(2 * np.pi * freq_up * t_event) * np.exp(-t_event * 5)
        start = int(a["birth_step"] * len(t_step))
        if start + len(t_event) < len(audio_buffer):
            audio_buffer[start : start + len(t_event)] += birth_sound

    # Son de Mort (Descendant)
    if a["death_step"] is not None:
        freq_down = np.linspace(800, 100, len(t_event))
        death_sound = 0.8 * np.sin(2 * np.pi * freq_down * t_event) * np.exp(-t_event * 10)
        start = int(a["death_step"] * len(t_step))
        if start + len(t_event) < len(audio_buffer):
            audio_buffer[start : start + len(t_event)] += death_sound

# Normalisation et export
audio_final = (audio_buffer / np.max(np.abs(audio_buffer)) * 32767).astype(np.int16)
wavfile.write("cycle_de_vie.wav", fs, audio_final)

# -----------------------------
# 5. SYNC GITHUB & DASHBOARD
# -----------------------------
np.savez(DNA_PATH, Theta=champion["Theta"], kappa=champion["kappa"], name=champion["name"])

try: 
    subprocess.run(["git", "add", DNA_PATH], check=False)
    subprocess.run(["git", "commit", "-m", f"CYCLE_COMPLETE: {champion['name']}"], check=False)
    subprocess.run(["git", "push", "origin", "main"], check=False)
except: pass

print("🎹 Lancement du monitoring sonore...")
subprocess.Popen(["afplay", "cycle_de_vie.wav"]) 

plt.figure(figsize=(12, 10))
plt.subplot(2,1,1); plt.plot(champion["heat_history"], color="gold")
plt.title(f"Profil Thermique du Champion : {champion['name']}")
plt.subplot(2,1,2); plt.imshow(champion["Theta"], cmap='magma')
plt.title("Signature Mentale : Source de la Vibration")
plt.tight_layout(); plt.show()