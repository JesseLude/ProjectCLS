import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def dla3D(rmin, k):
    
    '''
    this function simulates a coral, grown using diffusion limited aggregation
    rmin is the minimum radius that defines the region of growth; i.e. determines the cluster size
    k is the stickiness (determines growth probability)
    '''

    # for a sphere, volumne is proportional to r^3, so rmax = rmin * (d ^ 1/3), where d scales the coral growth's volume
    d = 3  
    max_iter = 20000  
    rmax = round(rmin * (d ** (1 / 3)))  

    # create a 3D grid and initialise it w 0s 
    N = 2 * rmax + 3
    A = np.zeros((N, N, N), dtype=int)  
    
    # start the coral the the bottom-centre 
    m = N // 2
    A[m, m, 0] = 1  

    # mass tracks the n/o particals in the coral
    # ^ could be an evaluation criteria(?) 
    mass = 1

    # flag to stop the simulation when needed
    terminate = False  

    for i in range(max_iter):

        # random starting position on a sphere around the cluster (so they're all equidistant from the seed)
        # new particle is spawned slightly beyond the growth boundary to give it some buffer zone to random-walk
        d = rmin + 2
        
        # converting (r, lat, lon) to cartesian (x,y,z) 
        # x = r * sin(lon) * cos(lat)
        # y = r * sin(lon) * sin(lat)
        # z = r * cos(lon)

        # random angle for x-y plane
        theta = random.uniform(0, 2 * np.pi)  
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

            # stick the particle if it's near the cluster and satisfies stickiness probability
            if neighbors > 0 and random.random() < k:
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
            direction = random.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
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
r = 50  
# stickiness 
k = .3

mass, A = dla3D(r, k)

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
ax.scatter(X, Y, Z, s=10, c='blue', alpha=0.7) 
# m is the midpoint 
m = (len(A) + 1) // 2 
ax.set_xlim([m - r - 2, m + r + 2])  
ax.set_ylim([m - r - 2, m + r + 2]) 
ax.set_zlim([0, m + r + 2]) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title(f"r={r}, k={k}")
plt.show()  