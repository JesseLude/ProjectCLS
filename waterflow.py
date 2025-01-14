import random

# define parameters
grid_size = 100
flow_velocity = 0.5
diffusion_coefficient = 0.1
flow_update_interval = 50 # time steps
peclet_number = flow_velocity * grid_size / diffusion_coefficient # formula of peclet number
directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

# initialize current direction
current_flow_direction = (0, 1) 

def update_flow_direction(time_step):
    """
    parameter: current time step
    return: the updated flow direction
    """
    global current_flow_direction
    if time_step % flow_update_interval == 0: # change direction every set interval
        current_flow_direction = random.choice(directions) # random new direction
    return current_flow_direction

def flow_diffusion_probability(x, y):
    """
    parameter: x, y: components of the direction vector
    return: the probability of moving in this direction
    """
    if (x, y) == current_flow_direction:
        return peclet_number / (peclet_number + 1)
    else: # other directions depend more on diffusion
        return 1 / (peclet_number + 1)

def random_walk(start_position, time_step):
    """
    simulates the random walk of a particle, incorporating dynamic water flow and peclet number.
    
    Parameters:
        start_position (tuple): Initial position (x, y) of the particle.
        time_step (int): Current time step of the simulation.
    
    Returns:
        tuple: The final position (x, y) where the particle attaches to the grid.
    """
    x, y = start_position
    
    while True:
        # update the flow direction
        update_flow_direction(time_step)
        
        # compute movement probabilities for each direction
        probabilities = [flow_diffusion_probability(dx, dy) for dx, dy in directions]
        probabilities = probabilities / sum(probabilities)  # normalize probabilities
        
        # randomly choose a direction based on probabilities
        chosen_index = np.random.choice(len(directions), p=probabilities)
        dx, dy = directions[chosen_index]
        
        # update particle position
        x = (x + dx) % grid_size
        y = (y + dy) % grid_size
        
        # check if the particle is attaching to the coral structure
        for dx, dy in directions:
            if grid[(x + dx) % grid_size, (y + dy) % grid_size]:  # if adjacent to coral
                grid[x, y] = True  # Mark the position as attached
                return x, y  # Return the final position
