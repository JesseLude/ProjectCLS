import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit


@jit(nopython=True)
def generate_random_position(rmin, m):
    """
    Generate a random starting position for a particle on a sphere.
    Ensures the particle starts slightly beyond the growth boundary.
    """
    d = rmin + 2  # Growth boundary buffer
    theta = np.random.uniform(0, 2 * np.pi)
    psi = np.arccos(1 - 2 * np.random.rand())  # Uniform sampling on sphere

    x = m + round(d * np.sin(psi) * np.cos(theta))
    y = m + round(d * np.sin(psi) * np.sin(theta))
    z = max(0, round(d * np.cos(psi)))  # Ensure z ≥ 0

    return x, y, z


@jit(nopython=True)
def get_random_direction():
    """
    Return a random direction for particle movement.
    Directions are slightly biased to promote upward growth.
    """
    directions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])

    # Change weights below to simulate advection
    weights = np.array([100, 100, 100, 100, 103, 100])
    cumulative_weights = np.cumsum(weights)
    total_weight = cumulative_weights[-1]
    random_value = np.random.randint(total_weight)

    for i, cum_weight in enumerate(cumulative_weights):
        if random_value < cum_weight:
            return directions[i]


@jit(nopython=True, parallel=True)
def dla3D(rmin, k_cons):
    """
    Simulate a coral growth using 3D diffusion-limited aggregation.

    Parameters:
        rmin (int): Minimum radius defining the region of growth.
        k_cons (float): Stickiness constant determining growth probability.

    Returns:
        mass (int): Total number of particles in the coral.
        A (np.ndarray): 3D grid representing the coral structure.
    """
    # Compute grid size and initialize
    d = 3
    rmax = round(rmin * (d ** (1 / 3)))
    N = 2 * rmax + 3
    A = np.zeros((N, N, N), dtype=np.int32)
    m = N // 2  # Center of the grid
    A[m, m, 0] = 1  # Starting seed at the bottom-center

    mass = 1
    terminate = False

    for _ in range(20000):  # Max iterations
        x, y, z = generate_random_position(rmin, m)

        # Skip if the particle starts inside the cluster
        if A[x, y, z] == 1:
            continue

        while True:
            # Sum neighbors to check proximity to the cluster
            neighbors = (
                A[x + 1, y, z] + A[x - 1, y, z] +
                A[x, y + 1, z] + A[x, y - 1, z] +
                A[x, y, z + 1] + A[x, y, z - 1]
            )

            # Stick particle if near the cluster and meets probability
            threshold = k_cons * z + 0.2
            if neighbors > 0 and np.random.rand() < threshold:
                if (x - m) ** 2 + (y - m) ** 2 + (z) ** 2 >= rmin ** 2:
                    terminate = True

                A[x, y, z] = 1
                mass += 1

                # Fill adjacent empty cells to make coral continuous
                for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N and A[nx, ny, nz] == 0:
                        A[nx, ny, nz] = 1

                break

            # Random walk to a new position
            direction = get_random_direction()
            nx, ny, nz = x + direction[0], y + direction[1], z + direction[2]

            if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N:
                x, y, z = nx, ny, nz

            # Remove particle if it escapes the growth region
            if (x - m) ** 2 + (y - m) ** 2 + (z) ** 2 > rmax ** 2:
                break

        if terminate:
            break

    return mass, A


def plot_coral(A, r, k_cons):
    """
    Visualize the 3D coral structure.
    """
    X, Y, Z = np.where(A == 1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=10, c=Z, alpha=0.7)
    ax.set_xlim([len(A) // 2 - r - 2, len(A) // 2 + r + 2])
    ax.set_ylim([len(A) // 2 - r - 2, len(A) // 2 + r + 2])
    ax.set_zlim([0, max(Z) * 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(f"Coral Growth (r={r}, k_cons={k_cons})")
    plt.show()


# Simulation parameters
r = 75
k_cons = 0.1

# Run simulation
mass, A = dla3D(r, k_cons)

# Plot results
plot_coral(A, r, k_cons)

