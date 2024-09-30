from engine import *

# LOCAL CONSTANTS
pass
# ENGINE CONSTANTS
MOVEMENT_SPEED = 25


def init():
    """Function main() called to initiate simulation."""

    main('Surface', [-40, 6, 10, 0.01, 0.01], pre, actions)


def pre():
    """Run before the start of the simulation."""
    global m

    m['sls'] = Model('assets/surface/sls.obj', 'assets/surface/sls.png')
    m['sls'].change_position(-25, 0.25, 10, rot_y=45, scale=3)

    m['surface'] = Model('assets/surface/surface0.obj', 'assets/surface/surface.png')
    m['surface'].change_position(scale=0.5)


def actions(elapsed_time):
    """Run every tick during the simulation."""

    m['sls'].change_position(y=elapsed_time * m['sls'].position[1] / 2)
    if m['sls'].position[1] >= 50:
        m['sls'].position[1] = 0.25
