import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

# Define system parameters
N = 60  # number of particles
L = 6  # size of the square field
dt = 0.000001  # time step
T_eq = 10000  # equilibrium time

k_damp = 0.02
k_frict = 0.05
R = 0.1

# Initialize positions and velocities randomly
x = np.random.rand(N) * L
y = np.random.rand(N) * L
vx = np.random.randn(N)
vy = np.random.randn(N)

# Define Lennard-Jones potential function
def LJ(r):
    return 4 * (5*r**-12 - 5*r**-6)

# Define function to calculate forces
def calculate_forces():
    global x, y, vx, vy
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r = np.sqrt(dx**2 + dy**2)        
            F = LJ(r)
            Fx[i] += F * dx / r
            Fy[i] += F * dy / r
            Fx[j] -= F * dx / r
            Fy[j] -= F * dy / r

        if x[i] < R:
            Fx[i] += k_damp * (R - x[i]) - k_frict * vx[i]
            x[i] = R
            vx[i] = -vx[i]          
        elif x[i] > L - R:
            Fx[i] -= k_damp * (R - (L - x[i])) - k_frict * vx[i]
            x[i] = L - R
            vx[i] = -vx[i]
        if y[i] < R:
            Fy[i] += k_damp * (R - y[i]) - k_frict * vy[i]
            y[i] = R
            vy[i] = -vy[i]
        elif y[i] > L - R:
            Fy[i] -= k_damp * (R - (L - y[i])) - k_frict * vy[i]
            y[i] = L - R
            vy[i] = -vy[i]
        
        Fx[i] -= k_damp * vx[i]
        Fy[i] -= k_damp * vy[i]
    return Fx, Fy

# Initialize plot
plt.ioff()
fig, ax = plt.subplots(figsize=(L, L))
fig.canvas.toolbar.pack_forget()

plt.xticks([])
plt.yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
scat = ax.scatter(x, y, color='red', s=60)
ax.set_xlim(0, L)
ax.set_ylim(0, L)

v_max = 50000
def update(i):
    global x, y, vx, vy, v_max
    Fx, Fy = calculate_forces()
    vx += Fx * dt
    vy += Fy * dt
    
    # Limit particle speed
    v = np.sqrt(vx**2 + vy**2)
    mask = v > v_max
    vx[mask] *= v_max / v[mask]
    vy[mask] *= v_max / v[mask]

    x += vx * dt
    y += vy * dt
    
    # Handle particle-wall collisions
    for i in range(N):
        if x[i] < R:
            x[i] = 2*R - x[i]
            vx[i] = -vx[i]
        elif x[i] > L - R:
            x[i] = 2*(L - R) - x[i]
            vx[i] = -vx[i]
        if y[i] < R:
            y[i] = 2*R - y[i]
            vy[i] = -vy[i]
        elif y[i] > L - R:
            y[i] = 2*(L - R) - y[i]
            vy[i] = -vy[i]

    # Update scatter plot
    scat.set_offsets(np.column_stack((x, y)))
    return scat,

# Animate the plot
ani = FuncAnimation(fig, update, frames=T_eq, interval=10, blit=True)

# Show final plot
plt.show(block=True)