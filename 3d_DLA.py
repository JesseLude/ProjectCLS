import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import porespy as ps
from skimage.draw import disk

@jit(nopython=True)
def generate_random_position(rmin, m):
    """
    Generate a random starting position for a particle on a sphere.
    Ensures the particle starts slightly beyond the growth boundary.
    """
    d = rmin + 2  # Buffer for the boundary
    theta = np.random.uniform(0, 2 * np.pi)
    psi = np.arccos(1 - 2 * np.random.rand())

    x = m + round(d * np.sin(psi) * np.cos(theta))
    y = m + round(d * np.sin(psi) * np.sin(theta))
    z = max(0, round(d * np.cos(psi)))  # Make sure z is larger than 0

    return x, y, z

@jit(nopython=True)
def get_random_direction(weights):
    """
    Return a random direction for particle movement.
    Directions are slightly biased to promote upward growth.
    """
    directions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])

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
    d = 3
    rmax = round(rmin * (d ** (1 / 3)))
    N = 2 * rmax + 3
    A = np.zeros((N, N, N), dtype=np.int32)
    m = N // 2  # Initialize the center of the grid
    A[m, m, 0] = 1  # Location for the seed

    mass = 1
    terminate = False

    for _ in range(20000):
        x, y, z = generate_random_position(rmin, m)

        if A[x, y, z] == 1:
            continue

        while True:
            neighbors = (
                A[x + 1, y, z] + A[x - 1, y, z] +
                A[x, y + 1, z] + A[x, y - 1, z] +
                A[x, y, z + 1] + A[x, y, z - 1]
            )

            # Set threshold for stickiness
            threshold = k_cons * z + 0.2
            if neighbors > 0 and np.random.rand() < threshold:
                if (x - m) ** 2 + (y - m) ** 2 + (z) ** 2 >= rmin ** 2:
                    terminate = True

                A[x, y, z] = 1
                mass += 1

                for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N and A[nx, ny, nz] == 0:
                        A[nx, ny, nz] = 1

                break

            # Get random direction and add it to each side
            direction = get_random_direction(weights=weights)
            nx, ny, nz = x + direction[0], y + direction[1], z + direction[2]

            if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N:
                x, y, z = nx, ny, nz

            if (x - m) ** 2 + (y - m) ** 2 + (z) ** 2 > rmax ** 2:
                break

        if terminate:
            break

    return mass, A

def plot_coral(A, r, k_cons, weights, plot_enabled=True):
    """
    Visualize the 3D coral structure.
    """
    if not plot_enabled:
        return

    # Color in where the particles are
    X, Y, Z = np.where(A == 1)

    # Set dimensions of the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=10, c=Z, alpha=0.7)
    ax.set_xlim([len(A) // 2 - r - 2, len(A) // 2 + r + 2])
    ax.set_ylim([len(A) // 2 - r - 2, len(A) // 2 + r + 2])
    ax.set_zlim([0, max(Z) * 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(f"Coral Growth (number of particles={r}, k_cons={k_cons}, weights={weights})")
    plt.show()

def crop_to_occupied_region(A, margin=2):
    """
    Crop the 3D array to the region containing the coral structure.

    Parameters:
        A (numpy.ndarray): 3D array representing the coral structure.
        margin (int): Margin to include around the occupied region.

    Returns:
        numpy.ndarray: Cropped 3D array.
    """
    coords = np.array(np.nonzero(A))
    x_min, y_min, z_min = coords.min(axis=1) - margin
    x_max, y_max, z_max = coords.max(axis=1) + margin + 1
    x_min, y_min, z_min = max(0, x_min), max(0, y_min), max(0, z_min)
    x_max = min(A.shape[0], x_max)
    y_max = min(A.shape[1], y_max)
    z_max = min(A.shape[2], z_max)

    return A[x_min:x_max, y_min:y_max, z_min:z_max]

def generate_binary_slices_with_circles(A, axis='z', circle_radius=2):
    """
    Generate binary 2D slices with circles representing coral particles.

    Parameters:
        A (numpy.ndarray): 3D array representing the coral structure.
        axis (str): Axis along which to slice ('x', 'y', or 'z').
        circle_radius (int): Radius of the circle to draw for each coral particle.

    Returns:
        List[numpy.ndarray]: List of binary 2D slices with circles.
    """
    # Can slice in all 3 dimensions
    if axis == 'z':
        slices = [A[:, :, z] for z in range(A.shape[2])]
    elif axis == 'y':
        slices = [A[:, y, :] for y in range(A.shape[1])]
    elif axis == 'x':
        slices = [A[x, :, :] for x in range(A.shape[0])]
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    binary_slices = []
    for slice_data in slices:
        slice_with_circles = np.zeros_like(slice_data, dtype=np.uint8)
        occupied_coords = np.array(np.nonzero(slice_data)).T
        for y, x in occupied_coords:
            rr, cc = disk((y, x), circle_radius)
            rr = np.clip(rr, 0, slice_with_circles.shape[0] - 1)
            cc = np.clip(cc, 0, slice_with_circles.shape[1] - 1)
            slice_with_circles[rr, cc] = 1
        binary_slices.append(slice_with_circles)

    return binary_slices

def analyze_slices(A, axis='z', circle_radius=2, num_slices=5, plot_enabled=True):
    """
    Analyze binary slices of the 3D coral structure for fractal dimensions.

    Parameters:
        A (numpy.ndarray): 3D array representing the coral structure.
        axis (str): Axis along which to slice ('x', 'y', or 'z').
        circle_radius (int): Radius of the circle to draw for each coral particle.
        num_slices (int): Number of slices to analyze.
        plot_enabled (bool): Whether to plot the analysis results.
    """
    
    binary_slices_with_circles = generate_binary_slices_with_circles(A, axis, circle_radius)
    for i, binary_slice in enumerate(binary_slices_with_circles[:num_slices]):
        print(f"Analyzing slice {i + 1}")
        data = ps.metrics.boxcount(binary_slice)

        # Only plot if true, to save time
        if plot_enabled:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax1.set_xlabel('Box Edge Length')
            ax1.set_ylabel('Number of Boxes Spanning Phases')
            ax1.plot(data.size, data.count, '-o', label=f'Slice {i + 1}')

            ax2.set_xscale('log')
            ax2.set_xlabel('Box Edge Length')
            ax2.set_ylabel('Slope')
            ax2.plot(data.size, data.slope, '-o', label=f'Slice {i + 1}')

            ax1.legend()
            ax2.legend()
            plt.tight_layout()
            plt.show()

# Simulation parameters
r = 75
k_cons = 0.1
weights = np.array([100, 100, 100, 100, 103, 100])

# Will plot the analysis of different slices if True
slice_analysis_enabled = True

# Will plot and show the coral if True
plot_coral_enabled = True

# Run simulation
mass, A = dla3D(r, k_cons)

# Optionally plot coral
plot_coral(A, r, k_cons, weights, plot_enabled=plot_coral_enabled)

# Optionally analyze slices
if slice_analysis_enabled:
    cropped_A = crop_to_occupied_region(A)
    analyze_slices(cropped_A, axis='z', circle_radius=2, num_slices=5, plot_enabled=True)