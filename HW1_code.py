import numpy as np
import matplotlib.pyplot as plt

# define grid size, goal state, obstacles, and parameters
grid_size = (6, 6)
goal_state = (2, 5)
obstacles = [(1, 2), (3, 1), (4, 5)]
gamma = 0.9
threshold = 1e-3

value_function = np.zeros(grid_size)

# Define reward function
def reward(s, a, next_s):
    if next_s == goal_state:
        return 1
    elif next_s in obstacles:
        return -1
    else:
        return 0

def transition(s, a):
    i, j = s
    if a == "up":
        next_s = (max(i - 1, 0), j)
    elif a == "down":
        next_s = (min(i + 1, grid_size[0] - 1), j)
    elif a == "left":
        next_s = (i, max(j - 1, 0))
    elif a == "right":
        next_s = (i, min(j + 1, grid_size[1] - 1))
    return next_s

def value_iteration_visualize(iterations):
    global value_function
    delta = float('inf')
    matrices = []  
    for _ in range(iterations):
        new_value_function = np.copy(value_function)
        delta = 0
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                s = (i, j)
                if s == goal_state or s in obstacles:
                    continue
                v = value_function[s]
                new_value_function[s] = max(
                    reward(s, a, transition(s, a)) + gamma * value_function[transition(s, a)]
                    for a in ["up", "down", "left", "right"]
                )
                delta = max(delta, abs(v - new_value_function[s]))
        value_function[:] = new_value_function 
        matrices.append(np.copy(value_function)) 
    return matrices

matrices = value_iteration_visualize(8)

# Create the plots for the iterations
fig, axes = plt.subplots(8, 1, figsize=(16, 30), constrained_layout=True)
for i, ax in enumerate(axes):
    im = ax.imshow(matrices[i], cmap="GnBu", interpolation='nearest')
    ax.set_title(f"Iteration {i+1}", fontsize=14)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            ax.text(y, x, f"{matrices[i][x, y]:.2f}", ha="center", va="center", color="black")
fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, label='Value Function Intensity')
plt.show()

# policy extraction
policy = np.full(grid_size, None) 

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        s = (i, j)
        if s == goal_state or s in obstacles:
            continue
        # Find the best action(maximum value)
        best_action = max(["up", "down", "left", "right"],
            key=lambda a: reward(s, a, transition(s, a)) + gamma * value_function[transition(s, a)])
        policy[s] = best_action

# Plotting the grid
plt.figure(figsize=(8, 6))

plt.imshow(value_function, cmap='Reds')

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        plt.text(j, i, f"{value_function[i, j]:.2f}",  color="white")
        if policy[i, j] is not None:
            arrow = {"up": "\u2191", "down": "\u2193", "left": "\u2190", "right": "\u2192"}[policy[i, j]]
            plt.text(j, i - 0.25, arrow, fontsize=18, color="white")
        if (i, j) == goal_state:
            plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="green"))
        elif (i, j) in obstacles:
            plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="black"))

plt.xlim(-0.5, grid_size[1] - 0.5)
plt.ylim(-0.5, grid_size[0] - 0.5)
plt.gca().invert_yaxis()
plt.show()