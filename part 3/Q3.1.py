"""
Assignment part 3 Dynamical Modelling
-figures are saved in a corexy_results folder alongside this script.
-Model assumptions: identical ideal DC motors, massless non-slip belts,
 linear viscous guide damping; nonlinear rolling friction is excluded.
"""
from pathlib import Path
from enum import Enum
import tkinter as tk
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.integrate import solve_ivp

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

# Hybrid State Machine for 3D printer control taken from Q2.1----------------
class State(Enum):
    """Enum for the states of the FSM."""
    P = "Printing"
    MT = "Moving Towards Target"
    W = "Wiping"
    WA = "Waiting For Filament"
    MB = "Moving Back"
class Input(Enum):
    """Enum for the inputs of the FSM, also the inputs to the system."""
    C = "Color Change"
    F = "Filament Refill"
    PR = "Position Reached"
    WC = "Wipe Complete"
    L = "Filament Loaded"
class Output(Enum):
    """Enum for the outputs of the FSM."""
    PSta = "Printing Start"
    Psto = "Printing Stop"
    SaveLoc = "Save Location"
    SW = "Start Wiping"
    NoOff = "Nozzle Heating Off"
    NoOn = "Nozzle Heating On"
    MoveToW = "Move to Wiping Area"
    R = "Return to Previous Location"

WIPING_AREA_POSITION = np.array([0.0, 0.0]) 
POSITION_TOLERANCE = 0.005

transitions = {
    (State.P, Input.C): (State.MT, {Output.Psto, Output.SaveLoc, Output.MoveToW}),
    (State.P, Input.F): (State.MT, {Output.Psto, Output.SaveLoc, Output.MoveToW}),
    (State.MT, Input.PR): (State.W, {Output.SW}),
    (State.W, Input.WC): (State.WA, {Output.NoOff}),
    (State.WA, Input.L): (State.MB, {Output.NoOn, Output.R}),
    (State.MB, Input.PR): (State.P, {Output.PSta})
}

#Function to process FSM------------------------------------------------------
def process_events(state, event, transitions):
    key = (state, event)

    if key not in transitions:
        return state, set() 

    next_state, outputs = transitions[key]
    return next_state, outputs

# The physical model----------------------------------------------------------------
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
def get_position(x):
    angles = x[:2]
    H = kinematic_matrix(PARAMETERS)
    position = H @ angles
    return position

# Motion generation -----------------------------------------------------------------
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

# Shape function from Q1.4------------------------------------------------------------
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

def plot_shape(shape_name, data, save=False):
    """Compare desired and actual path, voltages, and position tracking error."""
    t, wanted, actual, volts, states = data

    err = np.linalg.norm(wanted - actual, axis=0) * 1000  # mm

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.plot(wanted[0]*1000, wanted[1]*1000, "--", lw=2, label="Reference Path")
    ax.plot(actual[0]*1000, actual[1]*1000, lw=1.5, label="Simulated Path")
    ax.scatter([actual[0, 0]*1000], [actual[1, 0]*1000],
               marker="o", label="Start")

    ax.set_aspect("equal", adjustable="box")
    ax.set(
        xlabel="x [mm]",
        ylabel="y [mm]",
        title=f"{shape_name.title()} — Print Head Trajectory - {shape_name.title()}"
    )
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)

    axes[1].plot(t, volts[0], label="Left motor")
    axes[1].plot(t, volts[1], label="Right motor")
    axes[1].set(
        xlabel="Time [s]",
        ylabel="Voltage [V]",
        title="Calculated motor inputs"
    )
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(t, err)
    axes[2].set(
        xlabel="Time [s]",
        ylabel="Error [mm]",
        title="Distance from desired XY"
    )
    axes[2].grid(alpha=.25)

    fig.suptitle(
        "CoreXY 3D Printer - Hybrid System (Filament Refill) Simulation",
        fontsize=13,
        fontweight="bold"
    )
    fig.tight_layout()

    print(
        f"{shape_name.title()}: max tracking error {err.max():.5f} mm; "
        f"final closure error "
        f"{np.linalg.norm(actual[:, -1] - actual[:, 0])*1000:.5f} mm; "
        f"maximum voltage {np.max(np.abs(volts)):.2f} V"
    )

    return fig

# New Set of Functions for Q3.1 to generate "Parking" motion--------------------------
def build_motion(start_pos, target_pos):
    s = sp.symbols("s", real=True)
    alpha = s / (2 * sp.pi)

    x = start_pos[0] + alpha*(target_pos[0] - start_pos[0])
    y = start_pos[1] + alpha*(target_pos[1] - start_pos[1])

    derivatives = [
        sp.lambdify(
            s,
            sp.Matrix([
                sp.diff(x, s, k),
                sp.diff(y, s, k)
            ]),
            "numpy"
        )
        for k in range(4)
    ]
    
    return derivatives
def simulate_motion(current_state, target_pos, p, duration, n_samples):
    start_pos = get_position(current_state)
    funcs = build_motion(start_pos, target_pos)
    A, B = state_space_matrices(p)

    def dynamics(t, z):
        volts, _ = calculate_voltage(t, duration, funcs, p)
        return A @ z + B @ volts
    
    t = np.linspace(0, duration, n_samples)
    result = solve_ivp(dynamics, (0, duration), current_state, t_eval=t,
                    rtol=1e-8, atol=1e-10, max_step=duration/400)

    return result

# GUI and Buttons for Q3.1-------------------------------------------------------------
def buttons(window, callbacks):
    button_frame = tk.Frame(window, bg="#101923")
    button_frame.pack(pady=10)
    button_widgets = {}

    for name, call in callbacks.items():
        button = tk.Button(
            button_frame,
            text=name,
            command=call,
            bg="#168B79",
            fg="white",
            width=18
        )

        button.pack(side=tk.LEFT, padx=10)
        button_widgets[name] = button

    return button_widgets
def animate_print(window, canvas, fig, data, get_state, send_event, get_interruption):
    t, wanted, actual, volts, states = data

    ax = fig.axes[0]

    line = ax.lines[1] 
    line.set_data([], [])

    colors = ["orange", "purple"]
    color_index = 0

    line.set_color(colors[color_index])

    current_line = line
    segment_start = 0

    head, = ax.plot([], [], "o", color="#00E5FF", markersize=8)

    index = 0
    parking_index = 0
    parking_xy = None
    return_index = 0
    return_xy = None

    saved_position = None
    parked_state = None

    captured = False

    def update():
        nonlocal index, captured, parking_index, parking_xy
        nonlocal return_index, return_xy, saved_position, parked_state
        nonlocal color_index, segment_start, current_line

        if get_state() == State.P and index < len(t):
            index += 1

            current_line.set_data(
                actual[0, segment_start:index] * 1000, 
                actual[1, segment_start:index] * 1000
            )

            head.set_data(
                [actual[0, index-1] * 1000],
                [actual[1, index-1] * 1000]
            )

        elif get_state() == State.MT and not captured:
            current_state = states[:, index-1].copy()
            saved_position = get_position(current_state).copy()

            result = simulate_motion(
                current_state,
                WIPING_AREA_POSITION,
                PARAMETERS,
                3.0,
                200
            )

            if not result.success:
                raise RuntimeError(result.message)

            parked_state = result.y[:, -1].copy()
            parking_xy = kinematic_matrix(PARAMETERS) @ result.y[:2]
            parking_index = 0
            captured = True

        elif get_state() == State.MT and captured:
            if parking_xy is not None and parking_index < parking_xy.shape[1]:
                head.set_data(
                    [parking_xy[0, parking_index] * 1000],
                    [parking_xy[1, parking_index] * 1000]
                )

                parking_index += 1

            elif parking_xy is not None:
                final_position = parking_xy[:, -1]
                distance = np.linalg.norm(final_position-WIPING_AREA_POSITION)

                if distance <= POSITION_TOLERANCE:
                    print("Parked!")
                    print("Distance:", distance)
                    send_event(Input.PR)

                else:
                    print("Distance: ", distance)
        
        elif get_state() == State.MB and return_xy is None:
            result = simulate_motion(
                parked_state,
                saved_position,
                PARAMETERS,
                3.0,
                200
            )

            return_xy = kinematic_matrix(PARAMETERS) @ result.y[:2]
            return_index = 0

        elif get_state() == State.MB and return_xy is not None:
            if return_index < return_xy.shape[1]:
                head.set_data(
                    [return_xy[0, return_index] * 1000],
                    [return_xy[1, return_index] * 1000]
                )

                return_index += 1

            else:
                final_position = return_xy[:, -1]
                distance = np.linalg.norm(final_position-saved_position)

                if distance <= POSITION_TOLERANCE:
                    print("Returned!")
                    print("Distance:", distance)
                    send_event(Input.PR)

                    captured = False
                    parking_xy = None
                    return_xy = None

                    if get_interruption() == Input.C:
                        color_index = (color_index + 1) % len(colors)
                        segment_start = index - 1

                        current_line, = ax.plot(
                            [], [],
                            color = colors[color_index],
                            linewidth=2
                        )

                else:
                    print("Distance:", distance)

        canvas.draw_idle()

        if index < len(t):
            window.after(20, update)

    window.after(20, update)
def main():
    window = tk.Tk()
    window.title("Simulation")
    window.geometry("1200x650")
    window.configure(bg="#101923")

    data = simulate_shape("ellipse", PARAMETERS, 10, 500)
    fig = plot_shape("ellipse",data ,save=False)

    for ax in fig.axes[1:]:
        fig.delaxes(ax)

    ax = fig.axes[0]
    ax.set_position([0.12, 0.12, 0.76, 0.76])
    ax.set_xlim(min(ax.get_xlim()[0], -10), max(ax.get_xlim()[1], 10))
    ax.set_ylim(min(ax.get_ylim()[0], -10), max(ax.get_ylim()[1], 10))
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    canvas.get_tk_widget().pack(
        fill=tk.BOTH, 
        expand=True,
        padx=15, pady=16
        )

    def start():
        button_widgets["START"].config(state="disabled")
        animate_print(window, canvas, fig, data, lambda: state, on_key_press, lambda:interruption_type)

    callbacks = {
        "START": start,
        "COLOR CHANGE (C)": lambda: on_key_press(Input.C),
        "FILAMENT CHANGE (F)": lambda: on_key_press(Input.F),
        "WIPE COMPLETE (WC)": lambda: on_key_press(Input.WC),
        "FILAMENT LOADED (L)": lambda: on_key_press(Input.L)
        }

    button_widgets = buttons(window, callbacks)

    state = State.P
    interruption_type = None

    def on_key_press(event):
        nonlocal state, interruption_type
        if state == State.P and event in (Input.C, Input.F):
            interruption_type = event # Saves the input C or F so the machine knows to print a new color or remain

        state, outputs = process_events(state, event, transitions)

        print("Current state:", state.value)
        print("Output:", [output.value for output in outputs])

    window.mainloop()

if __name__ == "__main__":
    main()
