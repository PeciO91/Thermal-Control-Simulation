# Dynamic Thermal Control Simulation for Edge AI Processor

This project simulates dynamic thermal control of an Edge AI processor using a 2-state thermal model and compares no control, LQR, and MPC strategies.

## Overview

The model tracks two thermal states:

- **Core temperature** `T_c`
- **Heatsink temperature** `T_h`

The simulation loop uses the nonlinear thermal equations, while LQR and MPC are designed around a linearized deviation model. The script generates vector PDF plots suitable for a university semester paper on control theory.

## Mathematical Model

The state vector is:

```text
x = [T_c, T_h]
```

The nonlinear Euler update equations are:

```text
T_c[k+1] = T_c[k] + (dt / C_c) * (kappa * f[k]^3 + P_ext)
           - (dt / (R_c * C_c)) * (T_c[k] - T_h[k])

T_h[k+1] = T_h[k] + (dt / (R_c * C_h)) * (T_c[k] - T_h[k])
           - (dt / (R_h * C_h)) * (T_h[k] - T_amb)
```

## Parameters

- `dt = 0.1` s
- `C_c = 0.5`
- `C_h = 5.0`
- `R_c = 1.5`
- `R_h = 3.5`
- `kappa = 0.8`
- `T_amb_nom = 25.0` °C
- `P_ext_nom = 0.0` W

## Operating Point

- `T_c_star = 60.0` °C
- `T_h_star = 49.5` °C
- `u_star = 2.0606` GHz
- `x_star = [60.0, 49.5]`

## Linearized Deviation Model

The controllers use deviations:

```text
dx = x - x_star
du = f - u_star
```

```text
A = [[0.8667, 0.1333],
     [0.0133, 0.9810]]

B = [[2.038],
     [0.0]]
```

Disturbance feedforward in the MPC uses:

```text
G_dist = [[0.0, dt / C_c],
          [dt / (R_h * C_h), 0.0]]
```

with disturbance vector:

```text
dv = [T_amb - 25.0, P_ext - 0.0]
```

## Controller Design

### LQR Controller

- `Q = diag([1.0, 0.0])`
- `R = [[100.0]]`
- Control law:

```text
f_k = u_star - K @ (x_k - x_star)
```

- Frequency is clipped to `[0.5, 2.4]` GHz.

### MPC Controller

- Prediction horizon: `N_p = 10`
- Decision variables:
  - `dx = cp.Variable((2, N_p + 1))`
  - `du = cp.Variable((1, N_p))`
- Uses disturbance feedforward:

```text
dx[:, i+1] = A @ dx[:, i] + B @ du[:, i] + G_dist @ dv
```

- Constraints:
  - Input: `0.5 <= u_star + du <= 2.4`
  - Core temperature only: `25.0 <= x_star[0] + dx[0] <= 85.0`
- Solver: OSQP through CVXPY
- If the MPC solve is not optimal, the script prints a warning and falls back to LQR.

## Simulation Scenarios

### Scenario A: Stabilization without Disturbances

- Initial state: `x_0 = [75.0, 60.0]`
- Constant ambient temperature: `25.0` °C
- External power: `0.0` W
- Compares no control, LQR, and MPC.

### Scenario B: Stabilization with Disturbances

- Initial state: `x_0 = [60.0, 49.5]`
- Disturbances:
  - `t = 0-15s`: `T_amb = 25.0` °C, `P_ext = 0.0` W
  - `t = 15s+`: `P_ext = 3.0` W
  - `t = 35s+`: `T_amb = 35.0` °C
- Compares no control, LQR, and MPC.

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python simulate_control.py
```

This generates:

- `simulace_scenar_a.pdf`
- `simulace_scenar_b.pdf`

## Output Plots

Each PDF contains two vertical subplots:

1. **Top**: Core and heatsink temperatures over time
   - Solid lines show core temperature `T_c`
   - Dotted lines show heatsink temperature `T_h`
   - The target core temperature `60°C` and critical core limit `85°C` are shown as horizontal reference lines
2. **Bottom**: Frequency over time with maximum limit `2.4 GHz`