import matplotlib.pyplot as plt
import numpy as np
import random

# make a grid
grid_size = 200
grid = np.zeros((grid_size, grid_size))
grid[grid_size // 2, grid_size // 2] = 1          # set a seed in the middle

nutrient_map = np.outer(np.linspace(1, 0.1, grid_size),np.linspace(1, 0.5, grid_size))
def random_walk(max_steps = 1000):
    x, y = np.random.randint(0, grid_size, size=2)
    steps = 0

    while steps < max_steps:
        x += np.random.choice([-1, 0, 1])
        y += np.random.choice([-1, 0, 1])
        x, y = np.clip(x, 1, grid_size-2), np.clip(y, 1, grid_size-2)

        if grid[x-1:x+2, y-1:y+2].any():
            if np.random.rand() < nutrient_map[x, y]:  # nutrient-based probability
                grid[x, y] = 1
            break

        steps += 1

# simulate growth
for i in range(500):
    random_walk()

plt.imshow(grid, cmap='viridis')
plt.show()
