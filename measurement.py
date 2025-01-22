import numpy as np
from dla_3d import dla3D

def mass(coral):
    """
    calculate the total number of particles (mass) in the coral.
    """
    return np.sum(coral)

def shape_features(coral):
    """
    calculate shape features of the coral: radius and asymmetry.
    
    - radius: maximum distance from the center to any particle, representing the coral's growth extent.
    - asymmetry: distance of the coral's centroid from the center of the grid, indicating uneven growth.
    
    output:
        radius: float, maximum distance from the center.
        asymmetry: float, centroid's deviation from the center.
    """
    N = coral.shape[0]  # grid size
    m = N // 2  # center of the grid

    # get coordinates of coral particles
    coordinates = np.argwhere(coral == 1)

    # radius: maximum distance from the center to any particle
    radius = np.max(np.sqrt((coordinates[:, 0] - m) ** 2 + 
                             (coordinates[:, 1] - m) ** 2 + 
                             (coordinates[:, 2]) ** 2))

    # centroid: average position of all particles
    centroid = np.mean(coordinates, axis=0)

    # asymmetry: distance of centroid from the center
    asymmetry = np.sqrt((centroid[0] - m) ** 2 + (centroid[1] - m) ** 2 + (centroid[2]) ** 2)

    return radius, asymmetry

def growth_rate(coral):
    """
    calculate the vertical growth rate by measuring the number of new particles added at each layer.
    
    output:
        growth_rate: list, number of new particles added at each z-layer.
    """
    z_layers = coral.shape[2]  # number of z-layers
    growth_rate = []

    for z in range(z_layers):
        particles_in_layer = np.sum(coral[:, :, z])
        growth_rate.append(particles_in_layer)

    return growth_rate

def surface_area_to_volume(coral):
    """
    calculate the surface area to volume ratio of the coral in order to test the complexity.
    a higher ratio suggests that the coral has a more branched structure.
    
    - surface area: number of exposed faces of particles.
    - volume: total number of particles in the coral.
    
    output:
        ratio: float, surface area to volume ratio.
    """
    neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                          [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    surface_area = 0

    for x, y, z in np.argwhere(coral == 1):
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < coral.shape[0] and 0 <= ny < coral.shape[1] and 0 <= nz < coral.shape[2]:
                if coral[nx, ny, nz] == 0:
                    surface_area += 1

    volume = mass(coral)
    ratio = surface_area / volume if volume > 0 else 0

    return ratio

def fractal_dimension(coral):
    """
    calculate the fractal dimension using the box-counting method.
    
    - fractal dimension: measures the coral's fractal complexity or spatial filling.
    - method: divide the grid into boxes of decreasing size and count non-empty boxes.
    
    output:
        fractal dimension: float
    """
    N = coral.shape[0]  # grid size

    # list of box sizes to test
    box_sizes = [2 ** i for i in range(int(np.log2(N)) - 1, 0, -1)]
    box_counts = []

    for box_size in box_sizes:
        # number of boxes along each dimension
        num_boxes = N // box_size

        # count non-empty boxes
        count = 0
        for i in range(num_boxes):
            for j in range(num_boxes):
                for k in range(num_boxes):
                    # extract the sub-box
                    sub_box = coral[i * box_size:(i + 1) * box_size,
                                    j * box_size:(j + 1) * box_size,
                                    k * box_size:(k + 1) * box_size]
                    # check if sub-box contains any coral particles
                    if np.any(sub_box):
                        count += 1

        box_counts.append(count)

    # fit a line to log-log plot to calculate dimension
    log_box_sizes = np.log(1 / np.array(box_sizes))  # log of inverse box size
    log_box_counts = np.log(box_counts)  # log of box counts
    slope, _ = np.polyfit(log_box_sizes, log_box_counts, 1)  # slope is the fractal dimension

    return slope


if __name__ == "__main__":
    r = 75  # cluster size
    k_cons = 1/10  # stickiness constant
    total_mass, coral = dla3D(r, k_cons)  # calling the DLA simulation function from main file
    
    # mass
    coral_mass = mass(coral)

    # shape features
    radius, asymmetry = shape_features(coral)

    # fractal dimension
    fractal_dim = fractal_dimension(coral)

    # growth rate
    growth_rate_per_layer = growth_rate(coral)

    # surface area to volume ratio
    sv_ratio = surface_area_to_volume(coral)

    print(f"Mass: {coral_mass}")
    print(f"Radius: {radius:.2f}")
    print(f"Asymmetry: {asymmetry:.2f}")
    print(f"Fractal Dimension: {fractal_dim:.2f}")
    print(f"Growth Rate (per layer): {growth_rate_per_layer}")
    print(f"Surface Area to Volume Ratio: {sv_ratio:.2f}")