"""
Assignment part 1 Dynamical Modelling
-figures are saved in a corexy_results folder alongside this script.
-Model assumptions: identical ideal DC motors, massless non-slip belts,
 linear viscous guide damping; nonlinear rolling friction is excluded.
"""
from pathlib import Path
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.integrate import solve_ivp

#Choose Question-------------------------------------------------------------
#Options to run: "all", "q1.1", "q1.2", "q1.3", "q1.4, q1.5"
RUN = "q1.6"
SAVE_FIGURES = True
SHOW_FIGURES = True
N_SAMPLES = 800

#Demo parameters Q1.1
PARAMETERS = {
    "r_m": 0.01, "m": 0.4, "J_m": 5e-5, "K_m": 0.04,
    "L_m": 0.01, "R_m": 2.0, "mu_l_x": 0.7, "mu_l_y": 1.0,
}

#Q1.1: motor-angle samples in radians: (left angle, right angle)
ANGLE_EXAMPLES = [(0, 0), (5, 5), (5, -5), (10, 0)]

#Q1.2: (start time [s], LEFT motor voltage [V], RIGHT voltage [V])
MOTOR_VOLTAGE_STEPS = [(0, 2, 2), (0.35, 1, -1), (0.7, 0, 0)]
MOTOR_DURATION = 1.2

#Q1.3: (start time [s], x VIRTUAL voltage [V], y VIRTUAL voltage [V])
AXIS_VOLTAGE_STEPS = [(0, 2, 0), (0.5, 0, 1.5),
                      (1.0, -2, 0), (1.5, 0, -1.5)]
AXIS_DURATION = 2.0

#Q1.4 options: "ellipse", "flower", "both"
SHAPE = "both"
DURATION = 6.0
SHAPES = {
    "ellipse": {"center": (0.10, 0.10), "a": 0.050, "b": 0.025},
    "flower": {"center": (0.10, 0.10), "radius": 0.033,
               "lobes": 3, "amplitude": 0.23},
}


#Q1.5 Nozzle settings--------------------------------------------------------

NOZZLE_PARAMETERS = {
    "L_n": 0.05,       # Inductance [H]
    "C_n": 0.02,       # Capacitance [F]
    "R_n": 12.0,       # Heater resistance [Ohm]
    "C_t_n": 15.0,     # Thermal capacity [J/K]
    "R_t_n": 3.0,      # Thermal resistance [K/W]
    "T_e": 22.0,       # Environment temperature [°C]
}


#Q1.6 Filament settings------------------------------------------------------

FILAMENT_PARAMETERS = {
    # Geometry
    "ellipse_a": 0.05,       # Semi-major axis [m]
    "ellipse_b": 0.03,       # Semi-minor axis [m]
    "width": 0.0008,         # Filament width [m]
    "height": 0.0008,        # Filament height [m]
    "print_speed": 0.015,    # Nozzle speed [m/s]

    # Material properties (illustrative)
    "rho": 1250.0,           # Density [kg/m^3]
    "cp": 1800.0,            # Specific heat [J/(kg K)]
    "k": 0.18,               # Thermal conductivity [W/(m K)]

    # Heat exchange
    "h_air": 12.0,           # Air convection [W/(m^2 K)]
    "h_bed": 35.0,           # Effective bed heat transfer [W/(m^2 K)]

    # Temperatures
    "T_deposit": 205.0,      # Deposited filament temperature [C]
    "T_bed": 55.0,           # Heated bed temperature [C]
    "T_ambient": 22.0,       # Surrounding air temperature [C]
    "T_threshold": 65.0,     # Assumed acceptable temperature [C]

    # Numerical settings
    "n_segments": 100,       # Spatial discretization
    "cooling_duration": 120.0,  # Extra simulation time [s]
}


#Input (start time [s], volt [V])
NOZZLE_VOLTAGE_STEPS = [
    (0.0, 24.0),
    (90.0, 12.0),
    (130.0, 0.0),
]

NOZZLE_DURATION = 200.0

#Initial conditions: [temperature, current, capacitor voltage]
#None -> ambient temperature, zero current and zero voltage
NOZZLE_INITIAL_STATE = None


#Base Functions--------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "corexy_results"

#Save figures
def save_figure(fig, filename):
    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"  Saved: {path}")

#Check response step input voltage
def step_voltage(t, steps):
    values = steps[0][1:]
    for start, first, second in steps:
        if t >= start:
            values = (first, second)
        else:
            break
    return np.asarray(values, dtype=float)


def validate_settings():
    if RUN not in {"all", "q1.1", "q1.2", "q1.3", "q1.4", "q1.5", "q1.6"}:
        raise ValueError("RUN must be 'all' or 'q1.1' ... 'q1.5'")
    if SHAPE not in {"both", *SHAPES.keys()}:
        raise ValueError("SHAPE must be 'both', 'ellipse', or 'flower'")
    for key in ("r_m", "m", "J_m", "K_m", "L_m", "R_m"):
        if PARAMETERS[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for steps in (MOTOR_VOLTAGE_STEPS, AXIS_VOLTAGE_STEPS):
        if not steps or steps[0][0] != 0 or any(steps[k][0] >= steps[k+1][0]
                    for k in range(len(steps)-1)):
            raise ValueError("Voltage steps must start at t=0 and be sorted")


def kinematic_matrix(p):
    """Q1.1: [x, y]^T = H @ [theta_l, theta_r]^T."""
    r = p["r_m"]
    return (r / 2) * np.array([[1., 1.], [1., -1.]])


def mechanical_matrices(p):
    """Effective motor inertia and motor-space viscous damping."""
    H = kinematic_matrix(p)
    M = (p["J_m"] + p["m"] * p["r_m"]**2 / 2) * np.eye(2)
    D = H.T @ np.diag([p["mu_l_x"], p["mu_l_y"]]) @ H
    return M, D


def state_space_matrices(p):
    """Q1.2 model: z_dot = A @ z + B @ u; z=[angles, rates, currents]."""
    M, D = mechanical_matrices(p)
    K, L, R = p["K_m"], p["L_m"], p["R_m"]
    A, B = np.zeros((6, 6)), np.zeros((6, 2))
    A[0:2, 2:4] = np.eye(2)
    A[2:4, 2:4] = -np.linalg.solve(M, D)
    A[2:4, 4:6] = np.linalg.solve(M, K * np.eye(2))
    A[4:6, 2:4] = -(K / L) * np.eye(2)
    A[4:6, 4:6] = -(R / L) * np.eye(2)
    B[4:6, :] = np.eye(2) / L
    return A, B


def build_shape(shape_name, shape_settings):
    """Symbolically derive position and its first 3 derivatives w.r.t. phase."""
    s = sp.symbols("s", real=True)
    cx, cy = shape_settings["center"]
    if shape_name == "ellipse":
        x = cx + shape_settings["a"] * sp.cos(s)
        y = cy + shape_settings["b"] * sp.sin(s)
    elif shape_name == "flower":
        R = shape_settings["radius"]
        eps = shape_settings["amplitude"]
        n = shape_settings["lobes"]
        rho = R * (1 + eps * sp.cos(n * s))
        x, y = cx + rho * sp.cos(s), cy + rho * sp.sin(s)
    else:
        raise ValueError(f"Unknown shape: {shape_name}")
    derivatives = [sp.lambdify(s, sp.Matrix([sp.diff(x, s, k),
                                              sp.diff(y, s, k)]), "numpy")
                   for k in range(4)]
    return derivatives

#
def phase_and_derivatives(t, duration):
    """Smooth closed-loop timing: phase, phase', phase'', phase'''.

    7th-degree smoothstep stops with zero velocity, acceleration, and jerk
    at BOTH ends, so the initial six-state condition is consistent.
    """
    h = np.clip(np.asarray(t, dtype=float) / duration, 0, 1)
    phase = 35*h**4 - 84*h**5 + 70*h**6 - 20*h**7
    phase_1 = (140*h**3 - 420*h**4 + 420*h**5 - 140*h**6) / duration
    phase_2 = (420*h**2 - 1680*h**3 + 2100*h**4 - 840*h**5) / duration**2
    phase_3 = (840*h - 5040*h**2 + 8400*h**3 - 4200*h**4) / duration**3
    return 2*np.pi*phase, 2*np.pi*phase_1, 2*np.pi*phase_2, 2*np.pi*phase_3


def desired_motion(t, duration, shape_functions):
    """Chain rule: desired XY position, velocity, acceleration and jerk."""
    s, sd, sdd, sddd = phase_and_derivatives(t, duration)
    # Each lambdified function returns a (2,1) column.
    p0, p1, p2, p3 = [np.asarray(fn(float(s)), dtype=float).reshape(2)
                       for fn in shape_functions]
    return (p0,
            p1*sd,
            p2*sd**2 + p1*sdd,
            p3*sd**3 + 3*p2*sd*sdd + p1*sddd)


def calculate_voltage(t, duration, shape_functions, p):
    """Inverse dynamics:
       q = H^-1 p_xy;  i=(M qdd + D qd)/K;
       u=L idot + R i + K qd.
    """
    xy, vxy, axy, jxy = desired_motion(t, duration, shape_functions)
    Hinv = np.linalg.inv(kinematic_matrix(p))
    q, qd, qdd, qddd = (Hinv @ v for v in (xy, vxy, axy, jxy))
    M, D = mechanical_matrices(p)
    K, L, R = p["K_m"], p["L_m"], p["R_m"]
    current = (M @ qdd + D @ qd) / K
    current_rate = (M @ qddd + D @ qdd) / K
    voltage = L * current_rate + R * current + K * qd
    return voltage, np.r_[q, qd, current]


def simulate_shape(shape_name, p, duration, n_samples):
    """Feed calculated voltages to the ORIGINAL DC-motor state-space model."""
    funcs = build_shape(shape_name, SHAPES[shape_name])
    A, B = state_space_matrices(p)
    z0 = calculate_voltage(0.0, duration, funcs, p)[1]

    def dynamics(t, z):
        volts, _ = calculate_voltage(t, duration, funcs, p)
        return A @ z + B @ volts

    t = np.linspace(0, duration, n_samples)
    result = solve_ivp(dynamics, (0, duration), z0, t_eval=t,
                       rtol=1e-8, atol=1e-10, max_step=duration/400)
    if not result.success:
        raise RuntimeError(result.message)
    wanted = np.column_stack([desired_motion(ti, duration, funcs)[0] for ti in t])
    voltages = np.column_stack([calculate_voltage(ti, duration, funcs, p)[0] for ti in t])
    actual = kinematic_matrix(p) @ result.y[0:2]
    return t, wanted, actual, voltages, result.y


def plot_shape(shape_name, data, save=True):
    """Compare desired and actual path, voltages, and position tracking error."""
    t, wanted, actual, volts, states = data
    err = np.linalg.norm(wanted - actual, axis=0) * 1000  # mm
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.plot(wanted[0]*1000, wanted[1]*1000, "--", lw=2, label="Desired")
    ax.plot(actual[0]*1000, actual[1]*1000, lw=1.5, label="Simulated")
    ax.scatter([actual[0,0]*1000], [actual[1,0]*1000], marker="o", label="Start")
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", title=f"{shape_name.title()} — XY path")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[1].plot(t, volts[0], label="Left motor")
    axes[1].plot(t, volts[1], label="Right motor")
    axes[1].set(xlabel="Time [s]", ylabel="Voltage [V]", title="Calculated motor inputs")
    axes[1].grid(alpha=.25); axes[1].legend(fontsize=8)
    axes[2].plot(t, err)
    axes[2].set(xlabel="Time [s]", ylabel="Error [mm]", title="Distance from desired XY")
    axes[2].grid(alpha=.25)
    fig.suptitle("Q1.4 — inverse dynamics + six-state forward simulation", fontsize=13)
    fig.tight_layout()
    if save:
        save_figure(fig, f"q14_{shape_name}.png")
    print(f"{shape_name.title()}: max tracking error {err.max():.5f} mm; "
          f"final closure error {np.linalg.norm(actual[:,-1] - actual[:,0])*1000:.5f} mm; "
          f"maximum voltage {np.max(np.abs(volts)):.2f} V")
    return fig


def draw_workflow(save=True):
    """Diagram showing input shape -> mathematics -> output and verification."""
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set(xlim=(0, 14), ylim=(0, 3.5)); ax.axis("off")
    boxes = [
        (0.25, "INPUT\nShape + size\nDuration"),
        (3.08, "TRAJECTORY\nx, y, derivatives\nInverse kinematics"),
        (5.91, "INVERSE DYNAMICS\nMotor currents\nMotor voltages"),
        (8.74, "FORWARD MODEL\n6-state DC motor\nODE simulation"),
        (11.57, "OUTPUT\nActual XY path\nCompare to target"),
    ]
    for x, label in boxes:
        ax.add_patch(FancyBboxPatch((x, .9), 2.2, 1.8,
                     boxstyle="round,pad=0.12", facecolor="#edf2f8",
                     edgecolor="#43637f", linewidth=1.4))
        ax.text(x+1.1, 1.8, label, ha="center", va="center", fontsize=9.6)
    for x, _ in boxes[:-1]:
        ax.add_patch(FancyArrowPatch((x+2.36, 1.8), (x+2.75, 1.8),
                        arrowstyle="-|>", mutation_scale=15, linewidth=1.6))
    ax.set_title("CoreXY Q1.4: desired path to voltage inputs to actual drawing", fontsize=14)
    fig.tight_layout()
    if save:
        save_figure(fig, "q14_workflow.png")
    return fig



def symbolic_q11():
    """Q1.1: derive positions and velocities symbolically."""
    r, tl, tr, wl, wr = sp.symbols('r_m theta_l theta_r omega_l omega_r')
    H = (r / 2) * sp.Matrix([[1, 1], [1, -1]])
    q = sp.Matrix([tl, tr])
    xy = H * q
    v = H * sp.Matrix([wl, wr])
    print("\nQ1.1 KINEMATICS\n[x, y] = "); sp.pprint(xy)
    print("[x_dot, y_dot] = "); sp.pprint(v)
    print("Kinematic Jacobian H = "); sp.pprint(H)
    return H, xy, v


def run_q11():
    symbolic_q11()
    H = kinematic_matrix(PARAMETERS)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for tl, tr in ANGLE_EXAMPLES:
        x, y = H @ np.array([tl, tr])
        ax.scatter(x * 1000, y * 1000, s=65)
        ax.annotate(f"({tl:g}, {tr:g}) rad", (x*1000, y*1000),
                    xytext=(6, 7), textcoords="offset points")
    ax.set(xlabel="x relative to origin [mm]", ylabel="y relative to origin [mm]",
           title="Q1.1 — motor angles to XY coordinates")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.axvline(0, color="gray", linewidth=.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.25); fig.tight_layout()
    save_figure(fig, "q11_kinematics.png")


def symbolic_q12():
    """Q1.2: derive six-state matrices with symbolic parameters."""
    r, m, J, K, L, R, mux, muy = sp.symbols(
        'r_m m J_m K_m L_m R_m mu_l_x mu_l_y', positive=True)
    H = r/2 * sp.Matrix([[1,1],[1,-1]])
    M = (J + m*r*r/2) * sp.eye(2)
    D = H.T * sp.diag(mux, muy) * H
    A = sp.zeros(6, 6); B = sp.zeros(6, 2)
    A[:2, 2:4] = sp.eye(2)
    A[2:4, 2:4] = -M.inv()*D
    A[2:4, 4:6] = M.inv() * K
    A[4:6, 2:4] = -(K/L) * sp.eye(2)
    A[4:6, 4:6] = -(R/L) * sp.eye(2)
    B[4:6, :] = sp.eye(2)/L
    print("\nQ1.2 DYNAMICS: z=[theta_l, theta_r, omega_l, omega_r, i_l, i_r]")
    print("Mechanical inertia matrix M = "); sp.pprint(M)
    print("Mechanical damping matrix D = "); sp.pprint(D)
    print("A = "); sp.pprint(A)
    print("B = "); sp.pprint(B)
    return A, B


def integrate_steps(A, B, steps, duration, initial=None):
    """Integrate exactly across piecewise-voltage switching boundaries."""
    times = np.linspace(0, duration, N_SAMPLES)
    values = np.zeros((len(A), len(times)))
    z = np.zeros(len(A)) if initial is None else np.asarray(initial, float)
    edges = sorted({0.0, duration, *(s for s, _, _ in steps if 0<s<duration)})
    for a, b in zip(edges[:-1], edges[1:]):
        indices = np.flatnonzero((times >= a) & (times <= b))
        eval_times = np.unique(np.r_[a, times[indices], b])
        volts = step_voltage((a+b)/2, steps)
        sol = solve_ivp(lambda t, z: A @ z + B @ volts, (a,b), z,
                        t_eval=eval_times, rtol=1e-9, atol=1e-11)
        if not sol.success:
            raise RuntimeError(sol.message)
        if indices.size:
            values[:, indices] = np.vstack([
                np.interp(times[indices], sol.t, sol.y[i])
                for i in range(len(A))])
        z = sol.y[:, -1]
    inputs = np.array([step_voltage(t, steps) for t in times]).T
    return times, values, inputs


def plot_time_response(times, xy, velocities, currents, voltages, title, filename):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for signal, label in zip(voltages, ("Input 1 [V]", "Input 2 [V]")):
        ax[0,0].plot(times, signal, label=label)
    for signal, label in zip(currents, ("Current 1 [A]", "Current 2 [A]")):
        ax[0,1].plot(times, signal, label=label)
    for signal, label in zip(xy*1000, ("x [mm]", "y [mm]")):
        ax[1,0].plot(times, signal, label=label)
    for signal, label in zip(velocities*1000, ("vx [mm/s]", "vy [mm/s]")):
        ax[1,1].plot(times, signal, label=label)
    for axis in ax.flat:
        axis.grid(alpha=.25); axis.legend(fontsize=9)
        axis.set_xlabel("Time [s]")
    fig.suptitle(title); fig.tight_layout()
    save_figure(fig, filename)


def plot_xy(xy, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(xy[0]*1000, xy[1]*1000, lw=2)
    ax.scatter([xy[0,0]*1000], [xy[1,0]*1000], label="Start")
    ax.scatter([xy[0,-1]*1000], [xy[1,-1]*1000], marker="x", label="End")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", title=title)
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); save_figure(fig, filename)


def run_q12():
    symbolic_q12()
    A, B = state_space_matrices(PARAMETERS)
    t, z, u = integrate_steps(A, B, MOTOR_VOLTAGE_STEPS, MOTOR_DURATION)
    H = kinematic_matrix(PARAMETERS)
    xy, vel = H@z[:2], H@z[2:4]
    plot_time_response(t, xy, vel, z[4:6], u,
                       "Q1.2 — coupled DC-motor dynamics", "q12_time_response.png")
    plot_xy(xy, "Q1.2 — resulting tool-head path", "q12_xy.png")


def axis_matrices(p, mu_key):
    """Q1.3: matrix for one decoupled axis [position, velocity, current]."""
    r, K, L, R = (p[k] for k in ("r_m", "K_m", "L_m", "R_m"))
    je = p["J_m"] + p["m"]*r*r/2
    mu = p[mu_key]
    A = np.array([[0,1,0], [0,-r*r*mu/(2*je),r*K/je],
                  [0,-K/(r*L),-R/L]], float)
    B = np.array([0,0,1/L],float)
    return A, B

#Functions Subquestions-------------------------------------------------------
def symbolic_q13():
    r,m,J,K,L,R,mu = sp.symbols('r_m m J_m K_m L_m R_m mu_axis', positive=True)
    je = J+m*r*r/2
    A = sp.Matrix([[0,1,0], [0,-r*r*mu/(2*je),r*K/je],
                   [0,-K/(r*L),-R/L]])
    B = sp.Matrix([0,0,1/L])
    print("\nQ1.3 DECOUPLED AXIS [position, velocity, virtual current]")
    print("A_axis (mu_axis=mu_l_x or mu_l_y) = "); sp.pprint(A)
    print("B_axis = "); sp.pprint(B)
    print("Virtual inputs ux=(ul+ur)/2, uy=(ul-ur)/2")
    return A, B


def run_q13():
    symbolic_q13()
    Ax, Bx = axis_matrices(PARAMETERS,"mu_l_x")
    Ay, By = axis_matrices(PARAMETERS,"mu_l_y")
    A = np.zeros((6,6)); B = np.zeros((6,2))
    A[:3,:3], A[3:,3:] = Ax, Ay
    B[:3,0], B[3:,1] = Bx, By
    t,z,u_virtual = integrate_steps(A,B,AXIS_VOLTAGE_STEPS,AXIS_DURATION)
    xy = z[[0,3]]; velocities = z[[1,4]]; currents = z[[2,5]]
    plot_time_response(t,xy,velocities,currents,u_virtual,
                       "Q1.3 — decoupled x and y dynamics (virtual inputs)",
                       "q13_time_response.png")
    motors = np.vstack((u_virtual[0]+u_virtual[1],u_virtual[0]-u_virtual[1]))
    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot(t,motors[0],label="Left motor voltage")
    ax.plot(t,motors[1],label="Right motor voltage")
    ax.set(xlabel="Time [s]",ylabel="Voltage [V]",title="Q1.3 — physical motor voltages")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    save_figure(fig,"q13_physical_motor_voltages.png")
    plot_xy(xy,"Q1.3 — independent axes combined into a path","q13_xy.png")
    # Verify the transformed 3+3 system against the original 6-state model.
    _,z_motor,_=integrate_steps(*state_space_matrices(PARAMETERS),
              [(s,ux+uy,ux-uy) for s,ux,uy in AXIS_VOLTAGE_STEPS], AXIS_DURATION)
    xy_motor=kinematic_matrix(PARAMETERS)@z_motor[:2]
    difference=np.max(np.abs(xy-xy_motor))*1000
    print(f"Q1.3 numerical consistency versus coupled model: {difference:.6g} mm")


def run_q14():
    draw_workflow(SAVE_FIGURES)
    chosen = list(SHAPES) if SHAPE == "both" else [SHAPE]
    for shape_name in chosen:
        data = simulate_shape(shape_name, PARAMETERS, DURATION, N_SAMPLES)
        plot_shape(shape_name, data, SAVE_FIGURES)


#Q1.5 Nozzle Functions-----------------------------------------------------

def symbolic_q15():
    """Derive the nozzle state-space equations symbolically."""

    T, i, v, u = sp.symbols("T_n i_n v_C u_n")

    L, C, R, Ct, Rt, Te = sp.symbols(
        "L_n C_n R_n C_t_n R_t_n T_e"
    )

    f = sp.Matrix([
        (v**2/R - (T-Te)/Rt)/Ct,
        (u-v)/L,
        (i-v/R)/C
    ])

    print("\nQ1.5: Nozzle state-space model")
    print("States: [T_n, i_n, v_C]")
    print("Input: u_n")

    sp.pprint(f)

    return f


def nozzle_voltage(t, steps):
    """Return the applied nozzle voltage at time t."""

    voltage = steps[0][1]

    for start, value in steps:
        if t >= start:
            voltage = value
        else:
            break

    return float(voltage)


def nozzle_derivatives(t, state, params, voltage):
    """Calculate electrical and thermal state derivatives."""

    T, i, v = state

    L = params["L_n"]
    C = params["C_n"]
    R = params["R_n"]
    Ct = params["C_t_n"]
    Rt = params["R_t_n"]
    Te = params["T_e"]

    #Electrical dynamics
    di = (voltage - v) / L
    dv = (i - v/R) / C

    #Thermal dynamics
    heating_power = v**2 / R
    heat_loss = (T - Te) / Rt

    dT = (heating_power - heat_loss) / Ct

    return np.array([dT, di, dv])


def simulate_nozzle(params, steps, duration, initial_state=None):
    """Simulate the nozzle using scipy.solve_ivp."""

    times = np.linspace(0, duration, N_SAMPLES)

    if initial_state is None:
        initial_state = [params["T_e"], 0.0, 0.0]

    states = np.zeros((3, len(times)))
    state = np.array(initial_state, dtype=float)

    #Solve separately between voltage changes
    boundaries = sorted({
        0.0, duration,
        *(t for t, _ in steps if 0 < t < duration)
    })

    for start, end in zip(boundaries[:-1], boundaries[1:]):

        indices = np.where(
            (times >= start) & (times <= end)
        )[0]

        evaluation_times = np.unique(
            np.r_[start, times[indices], end]
        )

        voltage = nozzle_voltage((start + end)/2, steps)

        sol = solve_ivp(
            lambda t, z: nozzle_derivatives(
                t, z, params, voltage
            ),
            (start, end),
            state,
            t_eval=evaluation_times,
            rtol=1e-8,
            atol=1e-10
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        for j in range(3):
            states[j, indices] = np.interp(
                times[indices], sol.t, sol.y[j]
            )

        state = sol.y[:, -1]

    voltages = np.array([
        nozzle_voltage(t, steps) for t in times
    ])

    heating_power = states[2]**2 / params["R_n"]

    return times, states, voltages, heating_power


def plot_nozzle(data, params):
    """Plot voltage, current, heating power and temperature."""

    times, states, voltages, power = data
    T, i, v = states

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    #Voltage
    axes[0, 0].step(
        times, voltages, where="post", label="Input voltage"
    )
    axes[0, 0].plot(
        times, v, label="Capacitor/heater voltage"
    )
    axes[0, 0].set_ylabel("Voltage [V]")
    axes[0, 0].legend()

    # Current
    axes[0, 1].plot(times, i, label="Inductor current")
    axes[0, 1].plot(times, v / params["R_n"],
                    label="Heater current")
    axes[0, 1].set_ylabel("Current [A]")
    axes[0, 1].legend()

    #Power
    axes[1, 0].plot(times, power)
    axes[1, 0].set_ylabel("Heating power [W]")

    #Temperature
    axes[1, 1].plot(times, T, label="Nozzle temperature")
    axes[1, 1].axhline(
        params["T_e"], linestyle="--",
        color="gray", label="Environment"
    )
    axes[1, 1].set_ylabel("Temperature [°C]")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Q1.5 - Nozzle dynamics")
    fig.tight_layout()

    save_figure(fig, "q15_nozzle_response.png")

    return fig


def draw_nozzle_workflow():
    """Draw the input -> equations -> output diagram."""

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    boxes = [
        (0.12, "INPUT\nNozzle voltage\n$u_n(t)$"),
        (0.35, "ELECTRICAL MODEL\nInductor and capacitor\n$i_n(t), v_C(t)$"),
        (0.61, "THERMAL MODEL\nHeating and heat loss\n$P=v_C^2/R_n$"),
        (0.87, "OUTPUT\nTemperature\n$T_n(t)$"),
    ]

    for x, label in boxes:
        ax.text(
            x, 0.5, label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#EAF2FA",
                edgecolor="#507090"
            )
        )

    for x1, x2 in [(0.20, 0.25), (0.45, 0.51), (0.72, 0.78)]:
        ax.annotate(
            "", xy=(x2, 0.5), xytext=(x1, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.5)
        )

    ax.set_title("Q1.5 - Nozzle model workflow")

    save_figure(fig, "q15_nozzle_workflow.png")

    return fig


def run_q15():
    """Run the complete Q1.5 model."""

    symbolic_q15()

    draw_nozzle_workflow()

    data = simulate_nozzle(
        NOZZLE_PARAMETERS,
        NOZZLE_VOLTAGE_STEPS,
        NOZZLE_DURATION,
        NOZZLE_INITIAL_STATE
    )

    plot_nozzle(data, NOZZLE_PARAMETERS)

    print(f"Final nozzle temperature: {data[1][0, -1]:.2f} °C")


# ==================== Q1.6 FILAMENT PDE ====================

def ellipse_length(a, b):
    """Calculate circumference elliptical print path."""
    # Numerical integration of the ellipse arc length
    phi = np.linspace(0, 2*np.pi, 5001)
    ds_dphi = np.sqrt(
        (a*np.sin(phi))**2 + (b*np.cos(phi))**2
    )
    return np.trapz(ds_dphi, phi)


def filament_coefficients(params):
    """Calculate the thermal diffusion and cooling coefficients."""

    w = params["width"]
    h = params["height"]

    area = w * h
    exposed_perimeter = w + 2*h

    rho_cp_A = params["rho"] * params["cp"] * area

    alpha = params["k"] / (params["rho"] * params["cp"])

    air_rate = (
        params["h_air"] * exposed_perimeter / rho_cp_A
    )

    bed_rate = (
        params["h_bed"] * w / rho_cp_A
    )

    return alpha, air_rate, bed_rate


def filament_pde(T, params, ds):
    """Calculate temperature derivatives using finite differences."""

    alpha, air_rate, bed_rate = filament_coefficients(params)

    # Longitudinal heat conduction (insulated ends)
    laplacian = np.zeros_like(T)

    if len(T) > 1:
        laplacian[1:-1] = (
            T[2:] - 2*T[1:-1] + T[:-2]
        ) / ds**2

        laplacian[0] = (T[1] - T[0]) / ds**2
        laplacian[-1] = (T[-2] - T[-1]) / ds**2

    # Heat exchange with air and bed
    cooling_air = air_rate * (T - params["T_ambient"])
    exchange_bed = bed_rate * (T - params["T_bed"])

    return alpha * laplacian - cooling_air - exchange_bed


def advance_filament(T, duration, params, ds):
    """Integrate filament heat eq. for time interval."""

    if duration <= 0:
        return T.copy()

    alpha, air_rate, bed_rate = filament_coefficients(params)

    # Stability condition for explicit Euler integration
    dt_max = 0.35 / (
        2*alpha/ds**2 + air_rate + bed_rate
    )

    n_steps = max(1, int(np.ceil(duration / dt_max)))
    dt = duration / n_steps

    result = T.copy()

    for _ in range(n_steps):
        result += dt * filament_pde(result, params, ds)

    return result


def simulate_filament(params):
    """Simulate filament and cooling."""

    a = params["ellipse_a"]
    b = params["ellipse_b"]
    speed = params["print_speed"]
    n = params["n_segments"]

    length = ellipse_length(a, b)
    ds = length / n

    print_time = length / speed
    dt_deposit = ds / speed

    total_time = print_time + params["cooling_duration"]

    # Temperature history:
    # rows = time, columns = spatial position
    history = []
    time_history = []

    T = np.array([], dtype=float)
    current_time = 0.0

    def record(time, temperature):
        row = np.full(n, np.nan)
        row[:len(temperature)] = temperature
        history.append(row)
        time_history.append(time)

    # PRINTING: continuously extend the computational domain
    for j in range(n):
        if j > 0:
            T = advance_filament(T, dt_deposit, params, ds)
            current_time += dt_deposit

        # New filament enters at deposition temperature
        T = np.append(T, params["T_deposit"])
        record(current_time, T)

    # Finish the final deposition interval
    T = advance_filament(T, dt_deposit, params, ds)
    current_time += dt_deposit
    record(current_time, T)

    # COOLING: fixed domain, no more filament added
    cooling_times = np.linspace(
        current_time, total_time, 301
    )

    for new_time in cooling_times[1:]:
        T = advance_filament(
            T, new_time - current_time, params, ds
        )
        current_time = new_time
        record(current_time, T)

    return {
        "time": np.array(time_history),
        "position": np.arange(n) * ds,
        "temperature": np.array(history),
        "print_time": print_time,
        "length": length,
    }


def filament_cooling_time(result, params):
    """Find when the entire printed layer is below a threshold."""

    threshold = params["T_threshold"]
    times = result["time"]
    temperatures = result["temperature"]
    print_time = result["print_time"]

    # Only assess the completed layer
    for t, row in zip(times, temperatures):
        if t >= print_time and np.nanmax(row) <= threshold:
            return t

    return None


def plot_filament(result, params):
    """Plot temperature field."""

    times = result["time"]
    positions = result["position"]
    temperatures = result["temperature"]
    print_time = result["print_time"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    # Temperature heatmap along the filament
    mesh = axes[0].pcolormesh(
        positions * 1000,
        times,
        np.ma.masked_invalid(temperatures),
        shading="auto",
        cmap="inferno"
    )

    fig.colorbar(
        mesh, ax=axes[0], label="Temperature [C]"
    )

    axes[0].axhline(
        print_time, color="cyan", linestyle="--",
        label="Printing completed"
    )

    axes[0].set_xlabel("Distance along filament [mm]")
    axes[0].set_ylabel("Time [s]")
    axes[0].set_title("Growing filament temperature field")
    axes[0].legend()

    # Temperature of the first, middle and last segments
    for index, label in [
        (0, "First segment"),
        (len(positions)//2, "Middle segment"),
        (len(positions)-1, "Last segment"),
    ]:
        axes[1].plot(
            times, temperatures[:, index], label=label
        )

    axes[1].axhline(
        params["T_threshold"], color="red",
        linestyle=":", label="Cooling threshold"
    )

    axes[1].axvline(
        print_time, color="gray",
        linestyle="--", label="Printing completed"
    )

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Temperature [C]")
    axes[1].set_title("Cooling of selected filament segments")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, "q16_filament_temperature.png")

    return fig


def draw_filament_workflow():
    """Draw inputs, heat eq. and outputs filament model."""

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis("off")

    labels = [
        (0.13, "INPUT\nGeometry, speed\nand temperatures"),
        (0.38, "GROWING DOMAIN\nNew hot segments\nadded over time"),
        (0.64, "1D HEAT PDE\nConduction + air\nand bed exchange"),
        (0.88, "OUTPUT\nTemperature field\nand cooling time"),
    ]

    for x, label in labels:
        ax.text(
            x, 0.5, label,
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#EAF2FA",
                edgecolor="#507090"
            )
        )

    for x1, x2 in [(0.24, 0.27), (0.50, 0.53), (0.76, 0.79)]:
        ax.annotate(
            "", xy=(x2, 0.5), xytext=(x1, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.5)
        )

    ax.set_title("Q1.6 - Growing filament thermal model")
    save_figure(fig, "q16_filament_workflow.png")

    return fig


def run_q16():
    """Run filament PDE simulation."""

    print("\nQ1.6 - Filament cooling PDE")

    draw_filament_workflow()

    result = simulate_filament(FILAMENT_PARAMETERS)

    plot_filament(result, FILAMENT_PARAMETERS)

    cooling_time = filament_cooling_time(
        result, FILAMENT_PARAMETERS
    )

    print(f"Ellipse circumference: {result['length']:.4f} m")
    print(f"Printing time: {result['print_time']:.2f} s")

    if cooling_time is None:
        print("Cooling threshold not reached during simulation.")
    else:
        print(f"Entire layer below threshold at: {cooling_time:.2f} s")
        print(
            "Additional time after printing: "
            f"{cooling_time - result['print_time']:.2f} s"
        )


def draw_overview():
    """Automatically create an overview of all model functions."""

    import matplotlib.pyplot as plt
    import inspect

    # Collect functions defined in this script
    functions = inspect.getmembers(
        inspect.getmodule(draw_overview),
        inspect.isfunction
    )

    # Exclude functions imported from other modules
    functions = [
        (name, func)
        for name, func in functions
        if func.__module__ == __name__
    ]

    # Define groups based on function names
    def get_group(name):
        if name.startswith("run_q") or name in [
            "main", "validate_settings"
        ]:
            return "EXECUTION"

        if name in [
            "kinematic_matrix", "symbolic_q11"
        ]:
            return "Q1.1 - KINEMATICS"

        if name in [
            "mechanical_matrices",
            "state_space_matrices",
            "symbolic_q12"
        ]:
            return "Q1.2 - DYNAMICS"

        if name in [
            "axis_matrices",
            "symbolic_q13",
            "physical_motor_voltages"
        ]:
            return "Q1.3 - DECOUPLING"

        if name in [
            "build_shape",
            "desired_motion",
            "calculate_voltage",
            "simulate_shape",
            "plot_shape"
        ]:
            return "Q1.4 - SHAPE DRAWING"

        if "nozzle" in name or name == "symbolic_q15":
            return "Q1.5 - NOZZLE"
    

        if (
            name in {
            "ellipse_length",
            "filament_coefficients",
            "filament_pde",
            "advance_filament",
            "simulate_filament",
            "filament_cooling_time",
            "plot_filament",
            "draw_filament_workflow",
        }
        or name == "run_q16"
        ):
            return "Q1.6 - FILAMENT PDE"

        else: return "SHARED / UTILITIES"

    # Organize functions into groups
    groups = {}

    for name, func in functions:

        group = get_group(name)

        description = inspect.getdoc(func)

        if description is None:
            description = "No description available"

        # Use only the first line of the docstring
        description = description.split("\n")[0]

        groups.setdefault(group, []).append(
            (name, description)
        )

    # Calculate figure size based on number of functions
    total_rows = sum(
        len(items) + 1 for items in groups.values()
    )

    fig_height = max(8, total_rows * 0.38)

    fig, ax = plt.subplots(
        figsize=(14, fig_height)
    )

    ax.axis("off")

    # Draw function groups
    y = 0.97
    row_height = 0.90 / total_rows

    colors = {
        "EXECUTION": "#DDEBFF",
        "Q1.1 - KINEMATICS": "#E8F4FF",
        "Q1.2 - DYNAMICS": "#E6F5EA",
        "Q1.3 - DECOUPLING": "#FFF1D6",
        "Q1.4 - SHAPE DRAWING": "#F1E8FF",
        "Q1.5 - NOZZLE": "#FFE8E8",
        "SHARED / UTILITIES": "#EEEEEE"
    }

    for group, items in groups.items():

        # Group title
        ax.text(
            0.05, y, group,
            fontsize=13,
            fontweight="bold",
            transform=ax.transAxes
        )

        y -= row_height

        # Functions in this group
        for name, description in items:

            ax.text(
                0.07, y,
                f"{name}()",
                fontsize=10,
                fontweight="bold",
                transform=ax.transAxes,
                va="center"
            )

            ax.text(
                0.40, y,
                description,
                fontsize=10,
                transform=ax.transAxes,
                va="center"
            )

            ax.axhspan(
                y - row_height * 0.45,
                y + row_height * 0.45,
                xmin=0.04,
                xmax=0.96,
                facecolor=colors.get(group, "#EEEEEE"),
                alpha=0.6,
                zorder=-1
            )

            y -= row_height

    ax.set_title(
        "CoreXY Printer - Automatic Function Overview",
        fontsize=18,
        fontweight="bold",
        pad=25
    )

    # Save using the existing saving function
    save_figure(
        fig,
        "overview_input_model_output.png"
    )

    return fig



def main():
    validate_settings()
    print("CoreXY model | selected:", RUN)
    draw_overview()
    selected=["q1.1","q1.2","q1.3","q1.4", "q1.5", "q1.6"] if RUN=="all" else [RUN]
    tasks={"q1.1":run_q11,"q1.2":run_q12,"q1.3":run_q13,"q1.4":run_q14,"q1.5":run_q15, "q1.6":run_q16}
    for question in selected:
        print("\n"+"="*65+"\nRUNNING "+question.upper()+"\n"+"="*65)
        tasks[question]()
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")
    print("\nFinished. Figures directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
