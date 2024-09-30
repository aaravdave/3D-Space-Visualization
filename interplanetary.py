from engine import *

# LOCAL CONSTANTS
EARTH_ROTATION_SPEED = 24 / 93
MOON_ORBITAL_PERIOD = 27.3
# ENGINE CONSTANTS
MOVEMENT_SPEED = 0.5


def init():
    """Function main() called to initiate simulation."""

    main('Interplanetary', [6.8, 1.4, 32, 4, 0], pre, actions, 'assets/interplanetary/skybox/skybox.png')


def pre():
    """Run before the start of the simulation."""
    global m

    m['earth'] = Model('assets/interplanetary/earth/earth.obj', 'assets/interplanetary/earth/earth.jpg')
    m['earth'].change_position(5, 0.5, 30)

    m['moon'] = Model('assets/interplanetary/moon/moon.obj', 'assets/interplanetary/moon/moon.jpg')
    m['moon'].change_position(0.5, 0.5, 30)


def actions(elapsed_time):
    """Run every tick during the simulation."""

    m['earth'].change_position(rot_y=elapsed_time * EARTH_ROTATION_SPEED)
    moon_orbit_angle = m['earth'].position[4] / MOON_ORBITAL_PERIOD
    m['moon'].change_position(m['earth'].position[0] + np.cos(moon_orbit_angle) * EARTH_MOON_DISTANCE,
                         m['earth'].position[1],
                         m['earth'].position[2] + np.sin(moon_orbit_angle) * EARTH_MOON_DISTANCE,
                         rot_y=m['moon'].position[4] + elapsed_time * EARTH_ROTATION_SPEED / MOON_ORBITAL_PERIOD,
                         reset=1)
