import matplotlib.pyplot as plt
import numpy as np
import random

# make a grid
grid_size = 200
grid = np.zeros((grid_size, grid_size))
grid[grid_size // 2, grid_size // 2] = 1          # set a seed in the middle

nutrient_map = np.linspace(1, 0.1, grid_size).reshape(-1, 1)

def random_walk():
    x, y = np.random.randint(0, grid_size, size=2)
    while True:
        x += np.random.choice([-1, 0, 1])
        y += np.random.choice([-1, 0, 1])
        x, y = np.clip(x, 1, grid_size-2), np.clip(y, 1, grid_size-2)
        if grid[x-1:x+2, y-1:y+2].any():
            print(nutrient_map[x, y])
            if np.random.rand() < nutrient_map[x, y]:  # Nutrient-based probability
                grid[x, y] = 1
            break

# simulate growth
for i in range(5000):
    random_walk()

plt.imshow(grid, cmap='viridis')
plt.show()
