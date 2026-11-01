from enum import Enum

class State(Enum):
    Unparked = "U"
    MovingX = "MX"
    MovingY = "MY"
    BoundaryX = "X"
    BoundaryY = "Y"
    MovingX2 = "MX2"
    MovingY2 = "MY2"
    Parked = "P"

initial_state = State.Unparked
state = initial_state

while True:
    print(f"Current state: {state.value}")
    M, m = input("Enter M and m (e.g. V +): ").upper().split()
    event = (M, m)

    if state == State.Unparked and event == ("V", "+"):
        state = State.MovingY

    elif state == State.Unparked and event == ("H", "+"):
        state = State.MovingX

    elif state == State.Unparked and event == ("V", "0"):
        state = State.BoundaryY

    elif state == State.Unparked and event == ("H", "0"):
        state = State.BoundaryX

    elif state == State.MovingY and event == ("V", "0"):
        state = State.BoundaryY

    elif state == State.MovingX and event == ("H", "0"):
        state = State.BoundaryX

    elif state == State.BoundaryY and event == ("H", "+"):
        state = State.MovingX2

    elif state == State.BoundaryX and event == ("V", "+"):
        state = State.MovingY2

    elif state == State.MovingX2 and event == ("H", "0"):
        state = State.Parked

    elif state == State.MovingY2 and event == ("V", "0"):
        state = State.Parked

    else:
        print("Invalid event for the current state. Please try again.")

    print(f"New state: {state.value}")

