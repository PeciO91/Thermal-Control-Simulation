# Dynamic Thermal Control Simulation for Edge AI Processor

This project simulates the dynamic thermal control of an Edge AI processor using LQR and MPC controllers. The simulation generates PDF plots for a university semester paper on control theory.

## Overview

The system models the thermal behavior of an Edge AI processor with a nonlinear difference equation. Two control strategies are implemented and compared:

- **LQR (Linear Quadratic Regulator)**: Optimal feedback control with input saturation
- **MPC (Model Predictive Control)**: Constrained optimization over a prediction horizon

## Mathematical Model

The thermal dynamics are described by the nonlinear difference equation:

```
T_{k+1} = T_k * (1 - dt/(R_th * C_th)) + (dt * kappa / C_th) * f_k^3 + (dt / (R_th * C_th)) * T_amb + (dt / C_th) * P_ext
```

### Physical Parameters

- `R_th = 5.0` - Thermal resistance [K/W]
- `C_th = 2.0` - Thermal capacitance [J/K]
- `dt = 0.1` - Sampling time [s]
- `kappa = 0.8` - Power/frequency constant
- `T_amb_nom = 25.0` - Nominal ambient temperature [°C]
- `P_ext_nom = 0.0` - Nominal external power [W]

### Operating Point

- `T_target = 60.0` - Target temperature [°C]
- `f_target = 2.06` - Equilibrium frequency [GHz]

### Linearized Model

Deviations from the operating point: `dx = T - T_target`, `du = f - f_target`

```
A = 0.99
B = 0.51
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation script:

```bash
python simulate_control.py
```

This will generate two PDF plots:
- `simulace_scenar_a.pdf` - Stabilization without disturbances
- `simulace_scenar_b.pdf` - Stabilization with disturbances

## Simulation Scenarios

### Scenario A: Stabilization without Disturbances
- Initial temperature: 75°C (overheated state)
- Constant ambient temperature: 25°C
- No external power disturbances
- Compares: No control, LQR, and MPC

### Scenario B: Stabilization with Disturbances
- Initial temperature: 60°C (steady state)
- Time-varying disturbances:
  - t = 0-15s: T_amb = 25°C, P_ext = 0W
  - t = 15s+: P_ext steps to 3W (increased computational load)
  - t = 35s+: T_amb steps to 35°C (hotter environment)
- Compares: No control, LQR, and MPC

## Controller Design

### LQR Controller
- Weights: Q = 1, R = 10
- Control law: `f_k = f_target - K * (T_k - T_target)`
- Input saturation: [0.5, 2.4] GHz

### MPC Controller
- Prediction horizon: N_p = 10
- Weights: Q = 1, R = 10
- Terminal cost: Solution of Riccati equation from LQR
- Constraints:
  - Input: 0.5 ≤ f_k ≤ 2.4 GHz
  - State: 25.0 ≤ T_k ≤ 85.0 °C
- Solver: OSQP (via cvxpy)

## Output Plots

Each scenario generates a PDF with two subplots:
1. **Top**: Temperature T(t) over time with target (60°C) and critical limit (85°C)
2. **Bottom**: Frequency f(t) over time with maximum limit (2.4 GHz)

## Dependencies

- numpy==1.26.4
- scipy==1.13.0
- matplotlib==3.8.4
- control==0.10.0
- cvxpy==1.4.2
- osqp==0.6.3

## License

This project is created for educational purposes for a university control theory course.
