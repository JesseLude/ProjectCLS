import matplotlib
# specify backend for the animation
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parameters
grid_size = 200
light_factor = 0.7  # intensity of light (0 to 1)
nutrient_factor = 0.7  # influence of nutrients (0 to 1)

# seed at the bottom-center
seed_positions = [(grid_size - 1, grid_size // 2), (grid_size - 1, grid_size - grid_size // 4)
    , (grid_size - 1, grid_size // 4)]

# initialising the grid with FALSE (empty cells), except the singular seed cell
grid = np.zeros((grid_size, grid_size), dtype=bool)
for seed_position in seed_positions:
    grid[seed_position] = True

# directions for random walk: vertical, horizontal, and diagonal
directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1)]

def random_walk(start_position):

    """
    this function performs a random walk until the particle attaches to the coral aggregate.
    """
    x, y = start_position
    while True:

        # N O T E
        # random movement, with 70% bias towards upward movement
        # (thats how corals look, but look for sources for this + research if this bias is affected by environmental factors)
        # UPDATED now added a light factor that influences vertical growth
        if np.random.rand() < light_factor:
            dx, dy = directions[np.random.choice([4, 5])]
        else:
            dx, dy = directions[np.random.randint(0, 4)]

        # boundary conditions: particle should wrap around the edges of the grid
        x = (x + dx) % grid_size
        y = (y + dy) % grid_size

        # check attachment: if the particle lands next to the coral (i.e. a TRUE cell)...
        for dx, dy in directions:
            if grid[(x + dx) % grid_size, (y + dy) % grid_size] and np.random.rand() < nutrient_factor:
                # ... it attaches and stops moving
                grid[x, y] = True
                return x, y

def generate_particles():

    """
    this function generates particles and grows the cluster.
    """
    # initialise a list to store the final positions of those particles that successfully attached to the coral aggregate
    positions = []

    # simulate one particle at a time
    # N O T E - is this^ how it actually works biologically? perhaps there are many baby-corals/larvae?
    for i in range(num_particles):

        # the particles are spawned at random positions in the top half of the grid
        start_position = (np.random.randint(0, grid_size // 2), np.random.randint(0, grid_size))

        # check that the particle doesn’t start on an already occupied cell
        while grid[start_position]:
            start_position = (np.random.randint(0, grid_size // 2), np.random.randint(0, grid_size))

        # perform the random walk, until attachment. track each coral particle's position
        position = random_walk(start_position)
        positions.append(position)

        # yield the current grid so the animation can update
        yield grid, positions, i

# VISUALISATION

fig, ax = plt.subplots()

# blue sea and yellow? coral
# N O T E : probably we make custom cmaps for different environments later? e.g. high light intensity = light blue background
cmap = plt.get_cmap('cividis')

# display the simulation grid on the axes (ax).
# N O T E : with sharp edges between cells (we may wanna smoothen this so it looks like a biological coral)
img = ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)

# hide axes
plt.axis('off')

# add text to the simulation
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color='white', fontsize=12)
settings_text = ax.text(0.02, 0.8, "", transform=ax.transAxes, color='white', fontsize=12)


def update(frame):
    '''
    this function updates the grid for each frame of the animation.
    'frame' is the current state of the grid and positions, passed by generate_particles()
    '''
    global grid

    # get the grid, skip the positions here
    grid, _, time_step = frame

    # display updated grid and refresh
    img.set_data(grid)

    # update the time step text
    time_text.set_text(f'Time Step: {time_step}/{num_particles}')

    # update the settings text
    settings_text.set_text(f'Particles: {num_particles}\nLight Factor: {light_factor:.2f}\nNutrient Factor: {nutrient_factor:.2f}')

    return [img, time_text, settings_text]

simulation = generate_particles()

# create an animation by repeatedly calling the update function. 50ms per frame. optmisied refresh
ani = FuncAnimation(fig, update, frames=simulation, interval=50, blit=True, save_count=num_particles)

plt.show()