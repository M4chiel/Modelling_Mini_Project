"""
Assignment part 2 Finite State Machines
"""

from enum import Enum

class State(Enum):
    """Enum for the states of the FSM."""
    P = "Printing"
    MT = "Moving Towards Target"
    W = "Wiping"
    WA = "Waiting For Filament"
    MB = "Moving Back"

class Input(Enum):
    """Enum for the inputs of the FSM."""
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

transitions = {
    (State.P, Input.C): (State.MT, {Output.Psto, Output.SaveLoc, Output.MoveToW}),
    (State.P, Input.F): (State.MT, {Output.Psto, Output.SaveLoc, Output.MoveToW}),
    (State.MT, Input.PR): (State.W, {Output.SW}),
    (State.W, Input.WC): (State.WA, {Output.NoOff}),
    (State.WA, Input.L): (State.MB, {Output.NoOn, Output.R}),
    (State.MB, Input.PR): (State.P, {Output.PSta})

}

state = State.P

while True:
    print(f"\nCurrent state: {state.value}")

    event = input("Enter event (or Q): ").strip().upper()

    if event == "Q":
        break

    event = Input[event] if event in Input.__members__ else None
    previous = state
    key = (state, event)

    if key in transitions:
        next_state, outputs = transitions[key]
        state = next_state

        print(f"State Before: {previous.value}")
        print(f"Input: {event}")
        print(f"State After: {state.value}")
        print(f"Outputs: {', '.join(outputs.value for outputs in outputs)}")
    else:
        print("Invalid Transition")
