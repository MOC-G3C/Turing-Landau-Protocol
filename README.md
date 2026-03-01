# Turing-Landau Protocol 🧠

This repository contains the theoretical foundations and simulations of an artificial evolution protocol. The goal is to foster predictive cognitive abilities in autonomous agents facing a dynamic, noisy, and entropic environment (the "Entropic Zoo").

## 🧬 Cognitive Architecture (Latest V8.3x Advancements)

Our agents evolve in a **Rich Environment** (orbital mobile hotspots, correlated spatial turbulence, global cross-talk events). To survive and predict changes, they use an advanced neural architecture:

* **Triple Memory Channel:** Simultaneous information processing at different time scales:
    * `tau_fast`: Reflexes and immediate adaptations.
    * `tau_slow`: Medium-term contextual analysis.
    * `tau_ultra`: Very long-term memory footprint to spot distant cyclical patterns.
* **Non-Linear Coupling:** Integration of different information flows (via soft saturation and `tanh`) to prevent divergence.
* **Predictive Fitness (R² Score):** Evolution strictly selects agents capable of **anticipating the future** (up to 110 iterations ahead). Prediction is validated by dynamically optimized Ridge regressions.
* **Evolutionary Mechanics:** Fighting genetic stagnation via a periodic immigration system, a "Hall of Fame" for the absolute elite, and a homogenization penalty.

## 📁 Project Structure

* `/01_Theorie`: Mathematical and conceptual foundations of the protocol.
* `/02_Simulations`: The virtual laboratory. Contains environment engines (`_rich_env.py`) and genetic selection algorithms (`_predictive_fitness.py`).
* `/03_Core`: Central elements and basic structures of the architecture.
* `/04_PHYSICS`: Modeling of physical constraints (thermal dissipation, entropy, calculation of heat Q).

## 🚀 Running a Simulation

The `02_Simulations` folder contains a ready-to-use automation script to manage virtual environments and logging on macOS/Linux.

```bash
cd 02_Simulations
./run_v8_tuned.sh --tuned --fg
