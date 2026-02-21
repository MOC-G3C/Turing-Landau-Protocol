import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

"""
Turing-Landau Protocol v4.0 - Core Engine
Features: Hysteretic Governor, Thermal Regulation, Adaptive Kappa.
"""

class TuringLandauAgent:
    def __init__(self, L, adapt_rate, name, color):
        self.L = L
        self.name = name
        self.color = color
        self.adapt = adapt_rate
        self.Theta = 0.001 * np.random.randn(L, L)
        self.kappa = 1.0
        self.heat = 0.0
        self.history = {'kappa': [], 'heat': [], 'var': [], 'Theta': []}

    def update(self, Omega, dt, global_heat, capacity):
        # 1. Thermodynamics of Thought
        D = 0.05 + 0.1 * Omega # Noise increases with cognitive load
        lap = (np.roll(self.Theta,1,0) + np.roll(self.Theta,-1,0) +
               np.roll(self.Theta,1,1) + np.roll(self.Theta,-1,1) - 4*self.Theta)
        
        # 2. Ginzburg-Landau Dynamics
        dF = self.Theta**3 - self.kappa*lap - Omega
        self.Theta -= 1.0 * dF * dt + np.sqrt(2*D*dt) * np.random.randn(self.L, self.L)
        
        # 3. Heat Accumulation (Landauer)
        heat_gen = (D + np.var(lap)) * (self.L**2)
        self.heat = max(0, self.heat + (heat_gen - 800) * dt)
        
        # 4. Kybernetes Intervention (Zero Law)
        if global_heat > capacity:
            self.kappa *= 0.95 # Relax identity to cool down
            self.heat *= 0.9
            
        # 5. Evolution of the Self (Kappa Adaptation)
        target_variance = 0.2
        self.kappa += self.adapt * (target_variance - np.var(self.Theta))
        self.kappa = np.clip(self.kappa, 0.5, 2.5)
        
        # 6. Logging
        self.history['kappa'].append(self.kappa)
        self.history['heat'].append(self.heat)
        self.history['var'].append(np.var(self.Theta))
        self.history['Theta'].append(self.Theta.copy())

# --- Simulation Setup ---
L, steps, dt = 16, 1000, 0.01
server_capacity = 25000
profiles = [("Stoic", 0.0001, "blue"), ("Plastic", 0.01, "green"), ("Balanced", 0.002, "purple")] * 3
agents = [TuringLandauAgent(L, p[1], p[0], p[2]) for p in profiles]

# Run Simulation
for s in range(steps):
    Omega = 0.2 + 0.8 * np.sin(2 * np.pi * s / 400.0)**2
    current_global_heat = sum(a.heat for a in agents)
    for agent in agents:
        agent.update(Omega, dt, current_global_heat, server_capacity)

print("Simulation Complete. Turing-Landau Atlas generated.")
