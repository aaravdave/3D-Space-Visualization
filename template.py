from engine import *

# LOCAL CONSTANTS
pass
# ENGINE CONSTANTS
pass


def init():
    """Function main() called to initiate simulation."""

    main('Title', [0, 0, 0, 0.01, 0.01], pre, actions)  # Horizontal and vertical camera angles must be nonzero.


def pre():
    """Run before the start of the simulation."""
    global m

    pass


def actions(elapsed_time):
    """Run every tick during the simulation."""

    pass
