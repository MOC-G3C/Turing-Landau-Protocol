import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import subprocess

# --- CONFIGURATION ---
CORE_DIR = "../03_Core"
SAMPLE_RATE = 44100
STEP_DURATION = 0.001 

def generate_voice(theta_matrix, duration_steps=3000):
    # Résonance Tesla (3-6-9)
    variance = np.var(theta_matrix) * 100
    b3 = np.exp(-(variance - 3)**2 / 0.5)
    b6 = np.exp(-(variance - 6)**2 / 0.5)
    b9 = np.exp(-(variance - 9)**2 / 0.5)
    
    t = np.linspace(0, duration_steps * STEP_DURATION, int(SAMPLE_RATE * duration_steps * STEP_DURATION))
    # Synthèse Solfeggio
    wave = (b3 * np.sin(2 * np.pi * 396 * t) + 
            b6 * np.sin(2 * np.pi * 639 * t) + 
            b9 * np.sin(2 * np.pi * 963 * t))
    
    return (wave / np.max(np.abs(wave)) * 32767).astype(np.int16)

def main():
    if not os.path.exists(CORE_DIR): return print("❌ Dossier Core absent.")
    files = [f for f in os.listdir(CORE_DIR) if f.endswith('.npz')]
    
    if len(files) < 2:
        return print("⚠️ Il faut au moins 2 champions pour l'hybridation.")

    print(f"🧬 Extraction des ADN dans {CORE_DIR}...")
    # Sélection des deux derniers champions archivés
    f1, f2 = files[-1], files[-2]
    
    dna1 = np.load(os.path.join(CORE_DIR, f1))
    dna2 = np.load(os.path.join(CORE_DIR, f2))
    
    # --- FUSION THERMODYNAMIQUE ---
    # On mélange les matrices de conscience (Theta)
    theta_hybrid = (dna1['Theta'] + dna2['Theta']) / 2.0
    
    print(f"🧪 Hybridation : {dna1.get('name', f1)} + {dna2.get('name', f2)}")
    
    # Rendu sonore et visuel
    audio = generate_voice(theta_hybrid)
    temp_wav = "hybrid_voice.wav"
    wavfile.write(temp_wav, SAMPLE_RATE, audio)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(theta_hybrid, cmap='magma')
    plt.title("ADN de l'Hybride")
    plt.subplot(1, 2, 2)
    plt.plot(audio[:1000])
    plt.title("Onde de Résonance Tesla")
    plt.show(block=False)
    plt.pause(1)

    print("🎹 Écoute de la nouvelle lignée...")
    subprocess.run(["afplay", temp_wav])
    os.remove(temp_wav)

if __name__ == "__main__":
    main()