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
# 1. Physical Parameters and Model Constants (2nd-Order Model)
# =============================================================================
dt = 0.1            # Sampling time [s]
C_c = 0.5           # Core thermal capacitance [J/K]
C_h = 5.0           # Heatsink thermal capacitance [J/K]
R_c = 1.5           # Core-to-heatsink thermal resistance [K/W]
R_h = 3.5           # Heatsink-to-ambient thermal resistance [K/W]
kappa = 0.8         # Power/frequency constant
T_amb_nom = 25.0    # Nominal ambient temperature [°C]
P_ext_nom = 0.0     # Nominal external power [W]

# Equilibrium / Operating point
T_c_star = 60.0     # Target core temperature [°C]
T_h_star = 49.5     # Equilibrium heatsink temperature [°C]
u_star = 2.0606     # Equilibrium frequency [GHz]
f_star = u_star
x_star = np.array([T_c_star, T_h_star])  # State vector at equilibrium

# Linearized model matrices (deviations: dx = x - x_star, du = f - f_star)
A = np.array([[0.8667, 0.1333],
              [0.0133, 0.9810]])
B = np.array([[2.038],
              [0.0]])
G_dist = np.array([[0.0, dt / C_c],
                   [dt / (R_h * C_h), 0.0]])  # Maps [dv1, dv2] where dv1=T_amb-25, dv2=P_ext-0

# Controller weights
Q = np.diag([1.0, 0.0])  # Penalize core temperature only
R = np.array([[100.0]])   # Stronger input weight

# MPC parameters
N_p = 10            # Prediction horizon

# Constraints
f_min = 0.5         # Minimum frequency [GHz]
f_max = 2.4         # Maximum frequency [GHz]
T_c_min = 25.0      # Minimum core temperature [°C]
T_c_max = 85.0      # Maximum core temperature [°C] (critical limit)

# Simulation parameters
t_final = 60.0      # Final simulation time [s]
n_steps = int(t_final / dt)  # Number of simulation steps (600)

# =============================================================================
# 2. Mathematical Model Functions (2nd-Order)
# =============================================================================
def next_state(T_c_k, T_h_k, f_k, T_amb, P_ext):
    """
    Compute next core and heatsink temperatures using nonlinear difference equations.
    
    Parameters:
    T_c_k : Current core temperature [°C]
    T_h_k : Current heatsink temperature [°C]
    f_k : Current frequency [GHz]
    T_amb : Ambient temperature [°C]
    P_ext : External power [W]
    
    Returns:
    T_c_kp1 : Next core temperature [°C]
    T_h_kp1 : Next heatsink temperature [°C]
    """
    # Core temperature update
    T_c_kp1 = (T_c_k + (dt / C_c) * (kappa * f_k**3 + P_ext) - 
              (dt / (R_c * C_c)) * (T_c_k - T_h_k))
    
    # Heatsink temperature update
    T_h_kp1 = (T_h_k + (dt / (R_c * C_h)) * (T_c_k - T_h_k) - 
              (dt / (R_h * C_h)) * (T_h_k - T_amb))
    
    return T_c_kp1, T_h_kp1

# =============================================================================
# 3. LQR Controller (2nd-Order)
# =============================================================================
# Calculate optimal gain K using the control library
K, S, E = lqr(A, B, Q, R)
K = K  # K is 1x2 matrix

def lqr_controller(x_k):
    """
    LQR control law with input saturation.
    
    Parameters:
    x_k : Current state vector [T_c, T_h] [°C]
    
    Returns:
    f_k : Control input (frequency) [GHz]
    """
    # Control law: f_k = f_star - K @ (x_k - x_star)
    dx = x_k - x_star
    f_k = f_star - K @ dx
    f_k = f_k[0]  # Extract scalar
    
    # Input saturation
    f_k = np.clip(f_k, f_min, f_max)
    
    return f_k

# =============================================================================
# 4. MPC Controller (2nd-Order with Disturbance Feedforward)
# =============================================================================
# Calculate terminal cost P_f by solving the Riccati equation (already done in LQR)
P_f = S  # P_f is 2x2 matrix

def mpc_controller(x_k, T_amb, P_ext):
    """
    MPC controller using cvxpy optimization with disturbance feedforward.
    
    Parameters:
    x_k : Current state vector [T_c, T_h] [°C]
    T_amb : Current ambient temperature [°C]
    P_ext : Current external power [W]
    
    Returns:
    f_k : Control input (frequency) [GHz]
    """
    # Current state deviation
    dx_0 = x_k - x_star
    
    # Disturbance deviations
    dv = np.array([T_amb - T_amb_nom, P_ext - P_ext_nom])
    
    # Optimization variables: state and input deviations over horizon
    dx = cp.Variable((2, N_p + 1))
    du = cp.Variable((1, N_p))
    
    # Initial condition
    constraints = [dx[:, 0] == dx_0]
    
    # Cost function: sum of (dx^T*Q*dx + du^T*R*du)
    cost = 0
    for i in range(N_p):
        constraints.append(dx[:, i+1] == A @ dx[:, i] + B[:, 0] * du[0, i] + G_dist @ dv)
        constraints.append(u_star + du[0, i] >= f_min)
        constraints.append(u_star + du[0, i] <= f_max)
        constraints.append(x_star[0] + dx[0, i+1] >= T_c_min)
        constraints.append(x_star[0] + dx[0, i+1] <= T_c_max)
        cost += cp.quad_form(dx[:, i+1], Q) + R[0, 0] * cp.square(du[0, i])
    
    # Solve optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)
    
    if problem.status != 'optimal':
        print(f"Warning: MPC optimization status is {problem.status}. Falling back to LQR control.")
        return lqr_controller(x_k)
    
    # Extract first control input
    f_applied = u_star + du.value[0, 0]
    
    # Clip to ensure constraints are satisfied (numerical safety)
    f_applied = np.clip(f_applied, f_min, f_max)
    
    return f_applied

# =============================================================================
# 5. Simulation Loop Function (2nd-Order)
# =============================================================================
def simulate(x_0, T_amb_func, P_ext_func, controller_type):
    """
    Run simulation with specified controller.
    
    Parameters:
    x_0 : Initial state vector [T_c, T_h] [°C]
    T_amb_func : Function returning ambient temperature at time t
    P_ext_func : Function returning external power at time t
    controller_type : 'none', 'lqr', or 'mpc'
    
    Returns:
    t : Time array [s]
    T_c : Core temperature array [°C]
    T_h : Heatsink temperature array [°C]
    f : Frequency array [GHz]
    """
    # Initialize arrays
    t = np.linspace(0, t_final, n_steps + 1)
    T_c = np.zeros(n_steps + 1)
    T_h = np.zeros(n_steps + 1)
    f = np.zeros(n_steps + 1)
    
    # Initial conditions
    T_c[0], T_h[0] = x_0
    
    if controller_type == 'none':
        f[0] = f_star  # Fixed frequency
    else:
        f[0] = f_star
    
    # Simulation loop
    for k in range(n_steps):
        # Get current disturbances
        T_amb = T_amb_func(t[k])
        P_ext = P_ext_func(t[k])
        
        # Current state
        x_k = np.array([T_c[k], T_h[k]])
        
        # Compute control input
        if controller_type == 'none':
            f[k] = f_star
        elif controller_type == 'lqr':
            f[k] = lqr_controller(x_k)
        elif controller_type == 'mpc':
            f[k] = mpc_controller(x_k, T_amb, P_ext)
        
        # Compute next state using NONLINEAR equations
        T_c[k+1], T_h[k+1] = next_state(T_c[k], T_h[k], f[k], T_amb, P_ext)
    
    # Set final frequency
    x_final = np.array([T_c[-1], T_h[-1]])
    if controller_type == 'none':
        f[-1] = f_star
    elif controller_type == 'lqr':
        f[-1] = lqr_controller(x_final)
    elif controller_type == 'mpc':
        f[-1] = mpc_controller(x_final, T_amb_func(t[-1]), P_ext_func(t[-1]))
    
    return t, T_c, T_h, f

# =============================================================================
# 6. Scenario A: Stabilization without Disturbances
# =============================================================================
def scenario_a():
    """Simulate stabilization from overheated state without disturbances."""
    print("Running Scenario A: Stabilization without disturbances...")
    
    # Disturbance functions (constant)
    T_amb_func = lambda t: T_amb_nom
    P_ext_func = lambda t: P_ext_nom
    
    # Initial state (overheated core and heatsink)
    x_0 = np.array([75.0, 60.0])
    
    # Run three simulations
    t_none, T_c_none, T_h_none, f_none = simulate(x_0, T_amb_func, P_ext_func, 'none')
    t_lqr, T_c_lqr, T_h_lqr, f_lqr = simulate(x_0, T_amb_func, P_ext_func, 'lqr')
    t_mpc, T_c_mpc, T_h_mpc, f_mpc = simulate(x_0, T_amb_func, P_ext_func, 'mpc')
    
    return (t_none, T_c_none, T_h_none, f_none,
            t_lqr, T_c_lqr, T_h_lqr, f_lqr,
            t_mpc, T_c_mpc, T_h_mpc, f_mpc)

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
    
    # Initial state (steady state at equilibrium)
    x_0 = np.array([60.0, 49.5])
    
    # Run three simulations
    t_none, T_c_none, T_h_none, f_none = simulate(x_0, T_amb_func, P_ext_func, 'none')
    t_lqr, T_c_lqr, T_h_lqr, f_lqr = simulate(x_0, T_amb_func, P_ext_func, 'lqr')
    t_mpc, T_c_mpc, T_h_mpc, f_mpc = simulate(x_0, T_amb_func, P_ext_func, 'mpc')
    
    return (t_none, T_c_none, T_h_none, f_none,
            t_lqr, T_c_lqr, T_h_lqr, f_lqr,
            t_mpc, T_c_mpc, T_h_mpc, f_mpc)

# =============================================================================
# 8. PDF Plot Generation
# =============================================================================
def plot_scenario(results, filename, title):
    """
    Generate PDF plot for a scenario.
    
    Parameters:
    results : Tuple of simulation outputs for no control, LQR, and MPC
    filename : Output PDF filename
    title : Figure title
    """
    (t_none, T_c_none, T_h_none, f_none,
     t_lqr, T_c_lqr, T_h_lqr, f_lqr,
     t_mpc, T_c_mpc, T_h_mpc, f_mpc) = results
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top subplot: Core Temperature
    ax1.plot(t_none, T_c_none, 'k-', linewidth=2, label='No control (f=2.06 GHz)')
    ax1.plot(t_lqr, T_c_lqr, 'b-', linewidth=2, label='LQR')
    ax1.plot(t_mpc, T_c_mpc, 'r-', linewidth=2, label='MPC')
    ax1.plot(t_none, T_h_none, 'k:', linewidth=2, label='No control Heatsink')
    ax1.plot(t_lqr, T_h_lqr, 'b:', linewidth=2, label='LQR Heatsink')
    ax1.plot(t_mpc, T_h_mpc, 'r:', linewidth=2, label='MPC Heatsink')
    ax1.axhline(y=T_c_star, color='g', linestyle='--', linewidth=1.5, label='Target (60°C)')
    ax1.axhline(y=T_c_max, color='r', linestyle='--', linewidth=1.5, label='Critical limit (85°C)')
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
