import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from numba import jit

@jit(nopython=True, parallel=True)
def dla3D(rmin, k_cons):

    '''
    this function simulates a coral, grown using diffusion limited aggregation
    rmin is the minimum radius that defines the region of growth; i.e. determines the cluster size
    k is the stickiness (determines growth probability)
    '''

    # for a sphere, volume is proportional to r^3, so rmax = rmin * (d ^ 1/3), where d scales the coral growth's volume
    d = 3
    max_iter = 50000
    rmax = round(rmin * (d ** (1 / 3)))

    # create a 3D grid and initialise it w 0s
    N = 2 * rmax + 3
    A = np.zeros((N, N, N), dtype=np.int32)

    # start the coral the the bottom-centre
    m = N // 2
    A[m, m, 0] = 1

    # mass tracks the n/o particals in the coral
    # ^ could be an evaluation criteria(?)
    mass = 1

    # flag to stop the simulation when needed
    terminate = False

    for _ in range(max_iter):

        # random starting position on a sphere around the cluster (so they're all equidistant from the seed)
        # new particle is spawned slightly beyond the growth boundary to give it some buffer zone to random-walk
        d = rmin + 2

        # converting (r, lat, lon) to cartesian (x,y,z)
        # x = r * sin(lon) * cos(lat)
        # y = r * sin(lon) * sin(lat)
        # z = r * cos(lon)

        # random angle for x-y plane
        theta = np.random.uniform(0, 2 * np.pi)
        # random angle for z-axis
        psi = np.arccos(1 - 2 * random.random())
        # convert spherical to cartesian x, y, z (ensure z > or = 0, so above seabed)
        x = m + round(d * np.sin(psi) * np.cos(theta))
        y = m + round(d * np.sin(psi) * np.sin(theta))
        z = max(0, round(d * np.cos(psi)))

        # skip if particle starts already inside the cluster
        if A[x, y, z] == 1:
            continue

        while True:
            # sum the neighbors, if the sum is 0, it means the particle is not near the coral
            neighbors = (
                A[x + 1, y, z] + A[x - 1, y, z] + A[x, y + 1, z] +
                A[x, y - 1, z] + A[x, y, z + 1] + A[x, y, z - 1]
            )

            # determine threshold based on height z to make coral tend upwards
            threshold = k_cons * z + 0.2

            # stick the particle if it's near the cluster and satisfies stickiness probability
            if neighbors > 0 and random.random() < threshold:
                # stop the simulation if the cluster exceeds the limit
                if (x - m) ** 2 + (y - m) ** 2 + (z) ** 2 >= rmin ** 2:
                    terminate = True
                    print(f"x={x - m}, y={y - m}, z={z}")

                # add the particle to the cluster
                A[x, y, z] = 1
                mass += 1

                # THIS BIT MAY NEED TO BE CHANGED
                # I FILLED THE ADJACENT EMPTY CELLS SO THAT THE CORAL WOULD BE ONE CONTINUOUS ENTITY
                # BUT NO IDEA IF THIS IS BIOLOGICALLY ACCURATE. BUT WITHOUT IT THE CORAL LOOKS SO SKINNY
                for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N and A[nx, ny, nz] == 0:
                        A[nx, ny, nz] = 1

                # move on to the next particle
                break

            # remove the particle from its current position...
            A[x, y, z] = 0

            # ... and move it randomly to the new position...
            # LIGHT VARIABLE TO BE ADDED HERE TO BIAS THE DIRECTION OF GROWTH
            directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
            idx = np.random.randint(0, len(directions))
            direction = directions[idx]
            nx, ny, nz = x + direction[0], y + direction[1], z + direction[2]

            # ... if it's uncoccupied
            if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N and A[nx, ny, nz] == 0:
                x, y, z = nx, ny, nz
                A[x, y, z] = 1

            # remove the particle if it escapes the growth region
            if (x - m) ** 2 + (y - m) ** 2 + (z) ** 2 > rmax ** 2:
                A[x, y, z] = 0
                break

        # stop if the coral reaches the computational limit
        if terminate:
            break

    return mass, A

### simulation ###

# cluster size
r = 75
# stickiness constant
k_cons = 0.1

mass, A = dla3D(r, k_cons)

# extract the coordinates of all coral particles
X, Y, Z = [], [], []
N = len(A)
for i in range(N):
    for j in range(N):
        for l in range(N):
            if A[i, j, l] == 1:
                X.append(i)
                Y.append(j)
                Z.append(l)


# interactive backend for rotating the 3d grid
plt.switch_backend('QtAgg')

# plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, s=10, marker='o', c=Z, alpha=0.7)

# m is the midpoint
m = (len(A) + 1) // 2
ax.set_xlim([m - r - 2, m + r + 2])
ax.set_ylim([m - r - 2, m + r + 2])
# set z-axis to max value of z * 1.2
ax.set_zlim([0, max(Z) * 1.2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title(f"r={r}, k constant={k_cons}")
plt.show()


def crop_to_occupied_region(A, margin=2):
    """
    Crop the 3D array to the region containing the coral structure.

    Parameters:
        A (numpy.ndarray): 3D array representing the coral structure.

    Returns:
        numpy.ndarray: Cropped 3D array.
    """
    coords = np.array(np.nonzero(A))  # Find indices of non-zero elements
    x_min, y_min, z_min = coords.min(axis=1) - margin
    x_max, y_max, z_max = coords.max(axis=1) + margin + 1  # Include margin and the last occupied index

    # Ensure indices stay within bounds
    x_min, y_min, z_min = max(0, x_min), max(0, y_min), max(0, z_min)
    x_max = min(A.shape[0], x_max)
    y_max = min(A.shape[1], y_max)
    z_max = min(A.shape[2], z_max)

    return A[x_min:x_max, y_min:y_max, z_min:z_max]

# Crop the array
cropped_A = crop_to_occupied_region(A)


def generate_binary_slices_with_circles(A, axis='z', circle_radius=2):
    """
    Generate binary 2D slices with larger circles representing coral particles.

    Parameters:
        A (numpy.ndarray): 3D array representing the coral structure.
        axis (str): Axis along which to slice ('x', 'y', or 'z').
        circle_radius (int): Radius of the circle to draw for each coral particle.

    Returns:
        List[numpy.ndarray]: List of binary 2D slices with circles.
    """
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
        # Create a blank binary image for this slice
        slice_with_circles = np.zeros_like(slice_data, dtype=np.uint8)

        # Find all occupied pixels
        occupied_coords = np.array(np.nonzero(slice_data)).T

        # Draw a circle for each occupied pixel
        for y, x in occupied_coords:
            rr, cc = draw_circle(y, x, circle_radius, slice_with_circles.shape)
            slice_with_circles[rr, cc] = 1  # Fill the circle in the binary image

        binary_slices.append(slice_with_circles)

    return binary_slices

def draw_circle(y, x, radius, shape):
    """
    Generate the coordinates of a circle centered at (y, x) with the given radius.

    Parameters:
        y (int): Y-coordinate of the circle center.
        x (int): X-coordinate of the circle center.
        radius (int): Radius of the circle.
        shape (tuple): Shape of the 2D array (for boundary checks).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Row and column indices of the circle's pixels.
    """
    from skimage.draw import disk
    rr, cc = disk((y, x), radius, shape=shape)
    return rr, cc


import porespy as ps

# Generate binary slices with circles
binary_slices_with_circles = generate_binary_slices_with_circles(cropped_A, axis='z', circle_radius=2)

# Analyze each slice for fractal dimension
for i, binary_slice in enumerate(binary_slices_with_circles[:5]):  # Analyze the first 5 slices
    print(f"Analyzing slice {i + 1}")

    # Use the boxcount method
    data = ps.metrics.boxcount(binary_slice)

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('box edge length')
    ax1.set_ylabel('number of boxes spanning phases')
    ax2.set_xlabel('box edge length')
    ax2.set_ylabel('slope')
    ax2.set_xscale('log')
    ax1.plot(data.size, data.count, '-o', label=f'Slice {i + 1}')
    ax2.plot(data.size, data.slope, '-o', label=f'Slice {i + 1}')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()
