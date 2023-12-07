import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Binary cross-entropy function
def binary_cross_entropy(p: np.darray, q: np.darray) -> np.darray:
    """
    Calculate the binary cross entropy between two distributions.

    Parameters:
        p (np.darray): The first distribution.
        q (np.darray): The second distribution.

    Returns:
        np.darray: The binary cross entropy between the two distributions.
    """
    return -(p * np.log(q) + (1 - p) * np.log(1 - q))

x_values = np.linspace(0.01, 0.99, 100)

# Initialize plot
fig, ax = plt.subplots()
ax.plot(x_values, binary_cross_entropy(0, x_values), label='Set 1', c='darkorange')
ax.plot(x_values, binary_cross_entropy(1, x_values), label='Set 2', c='blue')

ax.set_xlim(0, 1)
ax.set_ylim(0, 5)
ax.set_xlabel('NN Output Score')
ax.set_ylabel('Binary Cross Entropy')
ax.legend(loc='upper center')

fig.savefig('binary_cross_entropy/binary_cross_entropy.png', dpi=300)
plt.clf()

# Initialize plot
fig, ax = plt.subplots()
ax.plot(x_values, binary_cross_entropy(0, x_values), label='Set 1', c='darkorange')
ax.set_xlim(0, 1)
ax.set_ylim(0, 5)
ax.set_xlabel('NN Output Score')
ax.set_ylabel('Binary Cross Entropy')
ax.legend(loc='upper center')

fig.savefig('binary_cross_entropy/binary_cross_entropy_set1.png', dpi=300)
plt.clf()

# Initialize plot
fig, ax = plt.subplots()
ax.plot(x_values, binary_cross_entropy(1, x_values), label='Set 2', c='blue')

ax.set_xlim(0, 1)
ax.set_ylim(0, 5)
ax.set_xlabel('NN Output Score')
ax.set_ylabel('Binary Cross Entropy')
ax.legend(loc='upper center')

fig.savefig('binary_cross_entropy/binary_cross_entropy_set2.png', dpi=300)
plt.clf()


def one_point_cross_entropy(label: int):
    """
    Plot the binary cross entropy and adapt one point on the curve.

    Parameters:
        label (int): The label of the point to adapt. 0 for set 1, 1 for set 2.
    """
    # Initialize plot
    fig, ax = plt.subplots()
    ax.plot(x_values, binary_cross_entropy(0, x_values), label='Set 1', c='darkorange')
    ax.plot(x_values, binary_cross_entropy(1, x_values), label='Set 2', c='blue')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    ax.set_xlabel('NN Output Score')
    ax.set_ylabel('Binary Cross Entropy')

    if label == 0:
        point, = ax.plot([], [], 'o', alpha=0.4, c='darkorange')
    else:
        point, = ax.plot([], [], 'o', alpha=0.4, c='blue')
    ax.legend()
    arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->'))
    text = ax.text(0.5, 4.5, '', ha='center', va='center')

    n_frames = 200

    # Update function for animation
    def update(frame, ax):
        if label == 0:
            point_values = np.linspace(0.8, 0.1, n_frames)
        else:
            point_values = np.linspace(0.2, 0.9, n_frames)
        
        # Update point on the background curve
        point.set_data(point_values[frame], binary_cross_entropy(label, point_values[frame]))
        text.set_text(f'loss = {binary_cross_entropy(label, point_values[frame]):.3f}')
        text.set_position((point_values[frame], binary_cross_entropy(label, point_values[frame]) + 0.2))
        ax.legend()

        # Update arrow
        if label == 0:
            arrow.set_position((point_values[frame] - 0.02, binary_cross_entropy(label, point_values[frame])))
            arrow.xy = (point_values[frame] - 0.15, binary_cross_entropy(label, point_values[frame]))
        else:
            arrow.set_position((point_values[frame] + 0.02, binary_cross_entropy(label, point_values[frame])))
            arrow.xy = (point_values[frame] + 0.15, binary_cross_entropy(label, point_values[frame]))

        return point, arrow,

    # Create animation
    animation = FuncAnimation(fig, update, frames=n_frames, interval=100, fargs=(ax,), repeat=False)

    if label == 0:
        animation.save('binary_cross_entropy/binary_cross_entropy_set1.gif', writer='imagemagick', fps=30)
    else:
        animation.save('binary_cross_entropy/binary_cross_entropy_set2.gif', writer='imagemagick', fps=30)
    plt.clf()



def multiple_points_cross_entropy():
    """
    Plot the binary cross entropy and adapt multiple points on the curve.
    """
    # Initialize plot
    fig, ax = plt.subplots()
    ax.plot(x_values, binary_cross_entropy(0, x_values), label='Set 1', c='darkorange')
    ax.plot(x_values, binary_cross_entropy(1, x_values), label='Set 2', c='blue')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    ax.set_xlabel('NN Output Score')
    ax.set_ylabel('Binary Cross Entropy')
    ax.legend()

    set1_points, = ax.plot([], [], 'o', alpha=0.4, c='darkorange')
    set2_points, = ax.plot([], [], 'o', alpha=0.4, c='blue')
    mean_line = ax.axhline(0, c='black', ls='--')
    text = ax.text(0.5, 3, '', ha='center', va='center')

    n_frames = 200

    # Update function for animation
    def update(frame, ax):
        set1_points_range = [(0.4, 0.2), (0.9, 0.3), (0.2, 0.28), (0.7, 0.1), (0.6, 0.43), (0.84, 0.52), (0.31, 0.35), (0.83, 0.75), (0.53, 0.71), (0.66, 0.05)]
        set2_points_range = [(0.52, 0.7), (0.4, 0.83), (0.1, 0.91), (0.32, 0.65), (0.81, 0.52), (0.1, 0.84), (0.32, 0.41), (0.23, 0.95), (0.73, 0.83), (0.04, 0.92)]
        
        set1_values = np.array([point_range[0] + frame / n_frames * (point_range[1] - point_range[0]) for point_range in set1_points_range])
        set2_values = np.array([point_range[0] + frame / n_frames * (point_range[1] - point_range[0]) for point_range in set2_points_range])
        
        # Calculate 
        set1_cross_entropy = binary_cross_entropy(0, set1_values)
        set2_cross_entropy = binary_cross_entropy(1, set2_values)
        mean_cross_entropy = np.mean(list(set1_cross_entropy) + list(set2_cross_entropy))
        mean_line.set_ydata(mean_cross_entropy)
        text.set_text(f'mean loss = {mean_cross_entropy:.3f}')
        text.set_position((0.5, mean_cross_entropy + 0.2))

        # Update point on the background curve
        set1_points.set_data(set1_values, set1_cross_entropy)
        set2_points.set_data(set2_values, set2_cross_entropy)
        ax.legend()

        return set1_points, set2_points,

    # Create animation
    animation = FuncAnimation(fig, update, frames=n_frames, interval=100, fargs=(ax,), repeat=False)

    animation.save('binary_cross_entropy/binary_cross_entropy_multiple_points.gif', writer='imagemagick', fps=30)
    plt.clf()
        

one_point_cross_entropy(0)
one_point_cross_entropy(1)
multiple_points_cross_entropy()