import numpy as np
from numba import jit, njit
#%%


@jit(cache=True, parallel=True, forceobj=True)
def clip_magnitude(array, low, high):  # To make everything stay in allowed range
    """
    Initializes position and speed of agents in the allowed range.

    Parameters
    ----------
    array: array_like.
    A two-dimensional array with some kind of data.

    low: float.
    The lowest parameter allowed.

    high: float.
    The highest parameter allowed.

    Returns
    -------
    : array_like.
    Modified input array without zeros and with parameters in the acceptable range.
    """
    magnitudes = np.linalg.norm(array.T, axis=0)
    no_zeros = np.abs(magnitudes) > 0
    clipped = np.clip(magnitudes, low, high)
    return clipped[no_zeros, None] * array[no_zeros] / magnitudes[no_zeros, None]


@jit(cache=True, parallel=True, forceobj=True)
def boids_initialization(boids, aspect, velocity_range):
    """
    Initializes position and speed of agents in the allowed area.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    aspect: float.
    Represents the configuration of walls and video format.

    velocity_range: array_like.
    An array representing lowest and highest speed for each agent.

    Returns
    -------
    None.
    """
    number = boids.shape[0]
    rng = np.random.default_rng(seed=None)
    boids[:, 0] = rng.uniform(0., aspect, size=number)  # Setting the coordinates
    boids[:, 1] = rng.uniform(0., 1., size=number)
    velocity = rng.uniform(velocity_range[0], velocity_range[1], size=number)  # Create random speed for each agent
    angle = rng.uniform(0, 2*np.pi, size=number)   # And random angle for each agent
    boids[:, 2] = velocity * np.cos(angle)  # Setting the projections of speeds
    boids[:, 3] = velocity * np.sin(angle)


@njit(cache=True, parallel=True)
def directions(boids):
    """
    Calculates the directions of moving agents on the screen.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    Returns
    -------
    : array_like.

    """
    return np.hstack((boids[:, :2] - boids[:, 2:4], boids[:, :2]))


@jit(cache=True, parallel=True, forceobj=True)
def movement(boids, dt, velocity_range):
    """
    Computes movement of agents and sets their speeds.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    dt: float.
    Represents time step in evaluation.

    velocity_range: array_like.
    An array representing lowest and highest speed for each agent.

    Returns
    -------
    None.
    """
    boids[:, :2] += dt*boids[:, 2:4] + 0.5 * dt**2 * boids[:, 4:6]  # Coordinates
    boids[:, 2:4] += dt*boids[:, 4:6]  # Speed
    boids[:, 2:4] = clip_magnitude(boids[:, 2:4], velocity_range[0], velocity_range[1])


@jit(cache=True, forceobj=True)
def jail(boids, aspect):
    """
    Makes agents stay between the walls. If agent passes the wall, it appears from the opposite side.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    aspect: float.
    Represents the configuration of walls and video format.

    Returns
    -------
    None.
    """
    boids[:, :2] %= np.array([aspect, 1.])


@jit(cache=True, parallel=True, forceobj=True)
def neighbours_detection(distance_matrix, threshold):
    """
    Makes agents stay between the walls. If agent passes the wall, it appears from the opposite side.

    Parameters
    ----------
    distance_matrix: array_like.
    Array with information about positions of the nearest agents (neighbours) regarding this agent.

    threshold: float.
    Represents the scope of the neighbors by the agent.

    Returns
    -------
    neighbours: array_like.
    An array with information about neighbours in the field of view.
    """
    neighbours = distance_matrix < threshold
    np.fill_diagonal(neighbours, False)
    return neighbours


@jit(cache=True, parallel=True, forceobj=True)
def cohesion(boids, i, indexes):
    """
    Describes behaviour when moving towards the average position of local flock mates.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    i: int.
    Represents index of the actual considered agent.

    indexes: array_like.
    Represent indexes of neighbours of considered agent.

    Returns
    -------
    : array_like.
    The acceleration for i-th agent to move to the flock mates.
    """
    return boids[indexes, :2].mean(axis=0) - boids[i, :2]


@jit(cache=True, parallel=True, forceobj=True)
def separation(boids, i, indexes, distances):
    """
    Describes behaviour when avoiding crowding local flock mates.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    i: int.
    Represents index of the actual considered agent.

    indexes: array_like.
    Represent indexes of neighbours of considered agent.

    distances: array_like.
    Array with information about positions of the neighbours regarding this agent.

    Returns
    -------
    : array_like.
    The acceleration for i-th agent to avoid crowding.
    """
    return np.sum((boids[i, :2] - boids[indexes, :2]) / distances[i, indexes, None], axis=0)


@jit(cache=True, parallel=True, forceobj=True)
def alignment(boids, i, indexes):
    """
    Describes behaviour when steering towards the average heading of local flock mates.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    i: int.
    Represents index of the actual considered agent.

    indexes: array_like.
    Represent indexes of neighbours of considered agent.

    Returns
    -------
    : array_like.
    The acceleration for i-th agent to align with the nearest cluster of neighbours.
    """
    return boids[indexes, 2:4].mean(axis=0) - boids[i, 2:4]


@njit(cache=True, parallel=True)
def avoid_walls(boids, aspect):
    """
    Calculates the power of the influence of walls on agents. Also, I added one more wall in the middle.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    aspect: float.
    Represents the configuration of walls and video format.

    Returns
    -------
    : array_like.
    Accelerations that gave the power of the walls' influence.
    """
    left_wall = np.abs(boids[:, 0])
    right_wall = np.abs(aspect - boids[:, 0])
    middle_wall = 1./2.
    upper_part = boids[:, 1] > middle_wall
    floor = np.zeros(boids.shape[0], dtype=boids.dtype)
    sail = np.zeros(boids.shape[0], dtype=boids.dtype)
    floor[upper_part] = np.abs(boids[upper_part, 1] - middle_wall)
    floor[~upper_part] = np.abs(boids[~upper_part, 1])
    sail[upper_part] = np.abs(1 - boids[upper_part, 1])
    sail[~upper_part] = np.abs(middle_wall - boids[~upper_part, 1])
    horizontal_acceleration = np.subtract(1. / left_wall**3, 1. / right_wall**3)  # I made walls influence a little
    vertical_acceleration = np.subtract(1. / floor**3, 1. / sail**3)  # smaller to furthest and bigger for nearest.
    return np.column_stack((horizontal_acceleration, vertical_acceleration))


@njit(cache=True, fastmath=True, parallel=True)
def noise(boids):
    """
    Creates noise to influence agents.

    Parameters
    ----------
    boids: array_like.
    Array with all information about agents: their coordinates, speeds and accelerations.

    Returns
    -------
    : float.
    Acceleration obtained by noise power.
    """
    return np.random.rand(boids.shape[0], 2) ** 2 - 0.1  # Noise influence might be more random
