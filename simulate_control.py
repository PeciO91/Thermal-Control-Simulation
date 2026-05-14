"""
Dynamic Thermal Control Simulation for Edge AI Processor
This script simulates thermal control using LQR and MPC controllers,
generating PDF plots for a university control theory paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from control import lqr
import cvxpy as cp

# =============================================================================
# 1. Physical Parameters and Model Constants
# =============================================================================
R_th = 5.0          # Thermal resistance [K/W]
C_th = 2.0          # Thermal capacitance [J/K]
dt = 0.1            # Sampling time [s]
kappa = 0.8         # Power/frequency constant
T_amb_nom = 25.0    # Nominal ambient temperature [°C]
P_ext_nom = 0.0     # Nominal external power [W]

# Equilibrium / Operating point
T_target = 60.0     # Target temperature [°C] (state x_star)
f_target = 2.06     # Equilibrium frequency [GHz] (input u_star)

# Linearized model matrices (deviations: dx = T - T_target, du = f - f_target)
A = 0.99
B = 0.51

# Controller weights
Q = 1.0             # State weight
R = 10.0            # Input weight

# MPC parameters
N_p = 10            # Prediction horizon

# Constraints
f_min = 0.5         # Minimum frequency [GHz]
f_max = 2.4         # Maximum frequency [GHz]
T_min = 25.0        # Minimum temperature [°C]
T_max = 85.0        # Maximum temperature [°C] (critical limit)

# Simulation parameters
t_final = 60.0      # Final simulation time [s]
n_steps = int(t_final / dt)  # Number of simulation steps (600)

# =============================================================================
# 2. Mathematical Model Functions
# =============================================================================
def next_temperature(T_k, f_k, T_amb, P_ext):
    """
    Compute next temperature using the nonlinear difference equation.
    
    Parameters:
    T_k : Current temperature [°C]
    f_k : Current frequency [GHz]
    T_amb : Ambient temperature [°C]
    P_ext : External power [W]
    
    Returns:
    T_{k+1} : Next temperature [°C]
    """
    T_kp1 = (T_k * (1 - dt / (R_th * C_th)) + 
             (dt * kappa / C_th) * f_k**3 + 
             (dt / (R_th * C_th)) * T_amb + 
             (dt / C_th) * P_ext)
    return T_kp1

# =============================================================================
# 3. LQR Controller
# =============================================================================
# Calculate optimal gain K using the control library
K, S, E = lqr(A, B, Q, R)
K = K[0, 0]  # Extract scalar gain

def lqr_controller(T_k):
    """
    LQR control law with input saturation.
    
    Parameters:
    T_k : Current temperature [°C]
    
    Returns:
    f_k : Control input (frequency) [GHz]
    """
    # Control law: f_k = f_target - K * (T_k - T_target)
    f_k = f_target - K * (T_k - T_target)
    
    # Input saturation
    f_k = np.clip(f_k, f_min, f_max)
    
    return f_k

# =============================================================================
# 4. MPC Controller
# =============================================================================
# Calculate terminal cost P_f by solving the Riccati equation (already done in LQR)
P_f = S[0, 0]  # Extract scalar terminal cost

def mpc_controller(T_k):
    """
    MPC controller using cvxpy optimization.
    
    Parameters:
    T_k : Current temperature [°C]
    
    Returns:
    f_k : Control input (frequency) [GHz]
    """
    # Current state deviation
    dx = T_k - T_target
    
    # Optimization variables: input increments du over horizon
    du = cp.Variable(N_p)
    
    # State predictions over horizon (as cvxpy expressions)
    dx_pred = cp.Variable(N_p)
    
    # Build state prediction constraints: dx[i+1] = A*dx[i] + B*du[i]
    constraints = []
    for i in range(N_p):
        if i == 0:
            constraints.append(dx_pred[i] == dx)
        else:
            constraints.append(dx_pred[i] == A * dx_pred[i-1] + B * du[i-1])
    
    # Cost function: sum of (Q*dx^2 + R*du^2) + terminal cost
    cost = cp.sum([Q * cp.square(dx_pred[i]) for i in range(N_p)]) + \
           cp.sum([R * cp.square(du[i]) for i in range(N_p)]) + \
           Q * P_f * cp.square(A * dx_pred[N_p-1] + B * du[N_p-1])
    
    # Input constraints: 0.5 <= f_k <= 2.4
    # f_k = f_target + du, so: f_min <= f_target + du <= f_max
    for i in range(N_p):
        constraints.append(f_target + du[i] >= f_min)
        constraints.append(f_target + du[i] <= f_max)
    
    # State constraints: 25.0 <= T_k <= 85.0
    # T_k = T_target + dx, so: T_min <= T_target + dx <= T_max
    for i in range(N_p):
        constraints.append(T_target + dx_pred[i] >= T_min)
        constraints.append(T_target + dx_pred[i] <= T_max)
    
    # Solve optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)
    
    # Extract first control input
    du_opt = du.value[0]
    f_k = f_target + du_opt
    
    # Clip to ensure constraints are satisfied (numerical safety)
    f_k = np.clip(f_k, f_min, f_max)
    
    return f_k

# =============================================================================
# 5. Simulation Loop Function
# =============================================================================
def simulate(T_0, T_amb_func, P_ext_func, controller_type):
    """
    Run simulation with specified controller.
    
    Parameters:
    T_0 : Initial temperature [°C]
    T_amb_func : Function returning ambient temperature at time t
    P_ext_func : Function returning external power at time t
    controller_type : 'none', 'lqr', or 'mpc'
    
    Returns:
    t : Time array [s]
    T : Temperature array [°C]
    f : Frequency array [GHz]
    """
    # Initialize arrays
    t = np.linspace(0, t_final, n_steps + 1)
    T = np.zeros(n_steps + 1)
    f = np.zeros(n_steps + 1)
    
    # Initial conditions
    T[0] = T_0
    
    if controller_type == 'none':
        f[0] = f_target  # Fixed frequency
    else:
        f[0] = f_target
    
    # Simulation loop
    for k in range(n_steps):
        # Get current disturbances
        T_amb = T_amb_func(t[k])
        P_ext = P_ext_func(t[k])
        
        # Compute control input
        if controller_type == 'none':
            f[k] = f_target
        elif controller_type == 'lqr':
            f[k] = lqr_controller(T[k])
        elif controller_type == 'mpc':
            f[k] = mpc_controller(T[k])
        
        # Compute next temperature using NONLINEAR equation
        T[k+1] = next_temperature(T[k], f[k], T_amb, P_ext)
    
    # Set final frequency
    if controller_type == 'none':
        f[-1] = f_target
    elif controller_type == 'lqr':
        f[-1] = lqr_controller(T[-1])
    elif controller_type == 'mpc':
        f[-1] = mpc_controller(T[-1])
    
    return t, T, f

# =============================================================================
# 6. Scenario A: Stabilization without Disturbances
# =============================================================================
def scenario_a():
    """Simulate stabilization from overheated state without disturbances."""
    print("Running Scenario A: Stabilization without disturbances...")
    
    # Disturbance functions (constant)
    T_amb_func = lambda t: T_amb_nom
    P_ext_func = lambda t: P_ext_nom
    
    # Initial state (overheated)
    T_0 = 75.0
    
    # Run three simulations
    t_none, T_none, f_none = simulate(T_0, T_amb_func, P_ext_func, 'none')
    t_lqr, T_lqr, f_lqr = simulate(T_0, T_amb_func, P_ext_func, 'lqr')
    t_mpc, T_mpc, f_mpc = simulate(T_0, T_amb_func, P_ext_func, 'mpc')
    
    return (t_none, T_none, f_none, t_lqr, T_lqr, f_lqr, t_mpc, T_mpc, f_mpc)

# =============================================================================
# 7. Scenario B: Stabilization with Disturbances
# =============================================================================
def scenario_b():
    """Simulate stabilization with time-varying disturbances."""
    print("Running Scenario B: Stabilization with disturbances...")
    
    # Disturbance functions (time-varying)
    def T_amb_func(t):
        if t < 35.0:
            return 25.0
        else:
            return 35.0
    
    def P_ext_func(t):
        if t < 15.0:
            return 0.0
        else:
            return 3.0
    
    # Initial state (steady state)
    T_0 = 60.0
    
    # Run three simulations
    t_none, T_none, f_none = simulate(T_0, T_amb_func, P_ext_func, 'none')
    t_lqr, T_lqr, f_lqr = simulate(T_0, T_amb_func, P_ext_func, 'lqr')
    t_mpc, T_mpc, f_mpc = simulate(T_0, T_amb_func, P_ext_func, 'mpc')
    
    return (t_none, T_none, f_none, t_lqr, T_lqr, f_lqr, t_mpc, T_mpc, f_mpc)

# =============================================================================
# 8. PDF Plot Generation
# =============================================================================
def plot_scenario(results, filename, title):
    """
    Generate PDF plot for a scenario.
    
    Parameters:
    results : Tuple of (t_none, T_none, f_none, t_lqr, T_lqr, f_lqr, t_mpc, T_mpc, f_mpc)
    filename : Output PDF filename
    title : Figure title
    """
    t_none, T_none, f_none, t_lqr, T_lqr, f_lqr, t_mpc, T_mpc, f_mpc = results
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top subplot: Temperature
    ax1.plot(t_none, T_none, 'k-', linewidth=2, label='No control (f=2.06 GHz)')
    ax1.plot(t_lqr, T_lqr, 'b-', linewidth=2, label='LQR')
    ax1.plot(t_mpc, T_mpc, 'r-', linewidth=2, label='MPC')
    ax1.axhline(y=T_target, color='g', linestyle='--', linewidth=1.5, label='Target (60°C)')
    ax1.axhline(y=T_max, color='r', linestyle='--', linewidth=1.5, label='Critical limit (85°C)')
    ax1.set_ylabel('Temperature [°C]', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([20, 90])
    
    # Bottom subplot: Frequency
    ax2.plot(t_none, f_none, 'k-', linewidth=2, label='No control (f=2.06 GHz)')
    ax2.plot(t_lqr, f_lqr, 'b-', linewidth=2, label='LQR')
    ax2.plot(t_mpc, f_mpc, 'r-', linewidth=2, label='MPC')
    ax2.axhline(y=f_max, color='r', linestyle='--', linewidth=1.5, label='Max limit (2.4 GHz)')
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Frequency [GHz]', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 3])
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {filename}")

# =============================================================================
# 9. Main Execution
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Dynamic Thermal Control Simulation")
    print("=" * 60)
    
    # Run Scenario A
    results_a = scenario_a()
    plot_scenario(results_a, 'simulace_scenar_a.pdf', 
                  'Scenario A: Stabilization without Disturbances (T₀ = 75°C)')
    
    # Run Scenario B
    results_b = scenario_b()
    plot_scenario(results_b, 'simulace_scenar_b.pdf', 
                  'Scenario B: Stabilization with Disturbances (T₀ = 60°C)')
    
    print("=" * 60)
    print("Simulation complete!")
    print("=" * 60)
