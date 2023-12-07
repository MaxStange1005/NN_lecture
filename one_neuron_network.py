import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import tensorflow as tf
from tensorflow.data import Dataset
import scipy.optimize
import time

# Set random seed for reproducibility
random_state = 42
np.random.seed(random_state)

def generate_gaussian_points(mean: float, cov_matrix: np.ndarray, num_points: int) -> np.ndarray:
    """
    Generate points from a Gaussian distribution.

    Parameters:
        mean (float): The mean of the Gaussian distribution.
        cov_matrix (np.ndarray): The covariance matrix of the Gaussian distribution.
        num_points (int): The number of points to generate.

    Returns:
        np.ndarray: The generated points.
    """
    points = np.random.multivariate_normal(mean, cov_matrix, num_points)
    return points


def custom_binary_crossentropy_loss(positions: np.ndarray, labels: np.ndarray, weight1: float, weight2: float) -> float:
    """
    Calculate the binary cross-entropy loss for a given set of points and labels.

    Parameters:
        positions (np.ndarray): The positions of the points.
        labels (np.ndarray): The labels of the points.
        weight1 (float): The first weight.
        weight2 (float): The second weight.

    Returns:
        float: The binary cross-entropy loss.
    """
    input1 = positions[:,0]
    input2 = positions[:,1]
    
    # Multiply the parameters with the input
    activation = weight1 * input1 + weight2 * input2
    
    # Apply sigmoid activation
    activation = 1 / (1 + np.exp(-activation))
    
    # Calculate binary cross-entropy loss
    loss = -np.mean(labels * np.log(activation) + (1 - labels) * np.log(1 - activation))
    
    return loss


def imitate_training():
    """
    Print a fake training process.
    """
    x = np.arange(1, 21)
    loss_values = 0.7 / x + 0.2
    accuracy_values = 0.9 - 0.2 / x

    # Add some noise to the data
    loss_values += np.random.normal(0, 0.01, size=len(x))
    accuracy_values += np.random.normal(0, 0.002, size=len(x))
    previous_output = ''
    for i, loss_value in enumerate(loss_values):
        batch = i % 4 + 1
        epoch = i // 4 + 1
        len_progress_bar = 30
        current_progress = int(len_progress_bar * batch / 4)
        if batch == 4:
            progress_bar = '[' + '=' * len_progress_bar + ']'
        else:
            progress_bar = '[' + '=' * current_progress + '>' + '.' * (len_progress_bar - current_progress - 1) + ']'
        current_output = f'Epoch {epoch}/5\n'
        current_output += f'{batch}/4 {progress_bar} - loss: {loss_value:.4f} - binary_accuracy: {accuracy_values[i]:.4f}\n'
        print(previous_output + current_output)
        if batch == 4:
            previous_output += current_output

# Define parameters for the two distributions
mean1 = np.array([-1, -1])
mean2 = np.array([1, 1])

# Covariance matrix (assuming diagonal covariance matrix for simplicity)
cov_matrix = np.eye(2)  # Identity matrix for simplicity

# Number of points in each set
num_points = 1000

# Generate points for each distribution
points_set1 = generate_gaussian_points(mean1, cov_matrix, num_points)
points_set2 = generate_gaussian_points(mean2, cov_matrix, num_points)

# Create labels for the two sets (0 for Set 1, 1 for Set 2)
labels_set1 = np.zeros(num_points)
labels_set2 = np.ones(num_points)

# Plot the generated points
plt.scatter(points_set1[:, 0], points_set1[:, 1], label='Set 1', alpha=0.3, s=10, c='darkorange')
plt.scatter(points_set2[:, 0], points_set2[:, 1], label='Set 2', alpha=0.3, s=10, c='blue')

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Add legend
plt.legend()
plt.savefig('one_neuron/gaussian_points.png', dpi=300, bbox_inches='tight')
plt.clf()

# Get some random points from each set
random_indices = np.random.choice(num_points, 4)
random_points_set1 = points_set1[random_indices]
random_points_set2 = points_set2[random_indices]

def latex_vector(name: str, values: list):
    """
    Write a vector in latex format.
    """
    equation = f'${name} = \\begin{{bmatrix}}'
    for value in values:
        equation += f' {value:.2f} \\\\'
    equation += '\\end{bmatrix}$'
    return equation

# Give latex equations for the random points
x_values = list(random_points_set1[:, 0]) + list(random_points_set2[:, 0])
y_values = list(random_points_set1[:, 1]) + list(random_points_set2[:, 1])
labels = [0]*len(random_points_set1) + [1]*len(random_points_set2)
# Create latex equations
print(latex_vector('x', x_values))
print(latex_vector('y', y_values))
print(latex_vector('label', labels))

# Highlight the random points
plt.scatter(points_set1[:, 0], points_set1[:, 1], label='Set 1', alpha=0.3, s=10, c='darkorange')
plt.scatter(points_set2[:, 0], points_set2[:, 1], label='Set 2', alpha=0.3, s=10, c='blue')
plt.scatter(random_points_set1[:, 0], random_points_set1[:, 1], alpha=1, s=10, c='r')
plt.scatter(random_points_set2[:, 0], random_points_set2[:, 1], alpha=1, s=10, c='r')

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Add legend
plt.legend()
plt.savefig('one_neuron/gaussian_points_random.png', dpi=300, bbox_inches='tight')
plt.clf()

# Combine the data and labels
point_coordinates = np.concatenate([points_set1, points_set2], axis=0)
point_labels = np.concatenate([labels_set1, labels_set2])

train_data = Dataset.from_tensor_slices((point_coordinates, point_labels))
train_data = train_data.shuffle(len(point_coordinates), seed=random_state)
# Set the batch size
train_data = train_data.batch(32)

# Loss function
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# Optimizer
adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.004, beta_1=0.7)

# Build the neural network with bias fixed at zero
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=2, activation='sigmoid', use_bias=False, kernel_initializer=tf.keras.initializers.Constant([0.0, 0.0]))
])

# Compilation
model.compile(optimizer=adam_optimizer, loss=loss_fn, metrics=['binary_accuracy'])

# Lists to store parameter values during training
weights_history = [[0, 0]]

# Custom callback to store weights during training
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        weights = weights[0].flatten()
        weights_history.append(weights)
        # Add a delay of 1 second after each epoch

model.fit(train_data, epochs=60, batch_size=32, callbacks=[CustomCallback()])

# Predict the labels for the points
point_predictions = model.predict(point_coordinates)

# Create a scatter plot with custom colors
scatter = plt.scatter(point_coordinates[:, 0], point_coordinates[:, 1], c=point_predictions.flatten(), cmap='coolwarm', vmin=0, vmax=1, alpha=0.5, s=10)

# Add a color bar
cbar = plt.colorbar(scatter)#, label='Prediction Score')

# Set custom color bar ticks and labels
cbar.set_ticks([0, 1])  # Set the ticks
cbar.set_ticklabels(['$0~=~$Set 1', '$1~=~$Set 2'])  # Set the tick labels

# Set axis labels
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Save or show the plot
plt.savefig('one_neuron/gaussian_points_predictions.png', dpi=300, bbox_inches='tight')
plt.clf()

def prediction_history_animation(point_coordinates, point_labels, weights_history):
    # Create figure and axes
    fig, ax = plt.subplots()
    # Create scatter plot
    scatter = ax.scatter(point_coordinates[:, 0], point_coordinates[:, 1], c=point_predictions.flatten(), cmap='coolwarm', vmin=0, vmax=1, alpha=0.5, s=10)
    # Add a color bar
    cbar = plt.colorbar(scatter)
    # Set custom color bar ticks and labels
    cbar.set_ticks([0, 1])  # Set the ticks
    cbar.set_ticklabels(['$0~=~$Set 1', '$1~=~$Set 2'])  # Set the tick labels
    # Set axis labels
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    n_frames = 100

    # Create text object above the plot
    text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, va='top', ha='left')

    # Animation function for updating the plot
    def update_plot(num, ax):
        epoch = int((num / n_frames)**2 * (len(weights_history) + 1))

        # Get prediction
        weight1 = weights_history[epoch][0]
        weight2 = weights_history[epoch][1]
        activation = weight1 * point_coordinates[:, 0] + weight2 * point_coordinates[:, 1]
        activation = 1 / (1 + np.exp(-activation))
        
        # Update the color array of the existing scatter plot
        scatter.set_array(activation)

        # Update text on the top of the graphic
        loss = custom_binary_crossentropy_loss(point_coordinates, point_labels, weight1, weight2)
        text.set_text(f'$loss_{{BCE}}({weight1:.5f}, {weight2:.5f}) = {loss:.5f}$')
        ax.set_title(f'Iteration: {epoch}')

    # Create animation
    animation = FuncAnimation(fig, update_plot, frames=n_frames, fargs=(ax,), interval=50)

    # Save animation as a video using PillowWriter
    animation.save('one_neuron/prediction_history.gif', writer=PillowWriter(fps=10))

prediction_history_animation(point_coordinates, point_labels, weights_history)

# Create a grid of weight values
weight1_values = np.linspace(0, 5, 100)
weight2_values = np.linspace(0, 5, 100)
weights1, weights2 = np.meshgrid(weight1_values, weight2_values)

# Calculate loss for each combination of weights
loss_values = np.zeros_like(weights1)
for i in range(len(weight1_values)):
    for j in range(len(weight2_values)):
        loss_values[i, j] = custom_binary_crossentropy_loss(point_coordinates, point_labels, weights1[i, j], weights2[i, j])


def plot_surface(weights1, weights2, loss_values):
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(weights1, weights2, loss_values, cmap='viridis')
    animation_interval = 720

    # Set labels and title
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss')
    #ax.set_title('Binary Cross-Entropy Loss Surface')

    # Animation function for rotating the plot
    def update_rotation(num, ax, surf):
        ax.view_init(elev=10, azim=num/animation_interval*360)

    # Create animation
    animation = FuncAnimation(fig, update_rotation, frames=np.arange(0, animation_interval, 1), fargs=(ax, surf), interval=50)

    # Adjust layout before saving
    plt.tight_layout()

    # Save animation as a video
    animation.save('one_neuron/loss_surface_rotation.gif', writer=PillowWriter(fps=30))

def plot_surface_training(weights_history, loss_values, point_coordinates, point_labels):
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    surf = ax.plot_surface(weights1, weights2, loss_values, cmap='viridis', alpha=0.8, zorder=4.4)

    # Set labels and title
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss')
    #ax.set_title('Binary Cross-Entropy Loss Surface')

    animation_interval = 720

    # Create text object for displaying prediction formula
    text = ax.text2D(0.5, 0.95, '', transform=ax.transAxes, fontsize=12, va='top', ha='center')

    # Initialize scatter plot
    scatter_current = ax.scatter([], [], [], c='red', marker='o', zorder=4.5)
    scatter_history = ax.scatter([], [], [], c='red', marker='o', alpha=0.2, zorder=4.5)

    # Animation function for rotating the plot and updating weights
    def update_weights_and_rotation(num, ax, surf, weights_history):
        # Update weights based on the history
        training_iteration = int(num / animation_interval * len(weights_history))
        current_training = weights_history[0:training_iteration + 1]
        training_weight1 = []
        training_weight2 = []
        training_loss = []
        for weight1, weight2 in current_training:
            training_weight1.append(weight1)
            training_weight2.append(weight2)
            training_loss.append(custom_binary_crossentropy_loss(point_coordinates, point_labels, weight1, weight2))

        ## Update the color array of the existing surface plot
        #surf.set_array(loss_values.flatten())

        # Set labels and title
        ax.set_xlabel('Weight 1')
        ax.set_ylabel('Weight 2')
        ax.set_zlabel('Loss')
        
        # Update the scatter plot data
        scatter_current._offsets3d = ([training_weight1[-1]], [training_weight2[-1]], [training_loss[-1]])
        scatter_history._offsets3d = (training_weight1[:-1], training_weight2[:-1], training_loss[:-1])

        # Rotate the plot
        ax.view_init(elev=10, azim=num/animation_interval*360)

        # Update text on the top of the graphic
        iteration_text = f'Iteration: {training_iteration}\n'
        iteration_text += f'$loss_{{BCE}}({training_weight1[-1]:.5f}, {training_weight2[-1]:.5f}) = {training_loss[-1]:.5f}$'
        text.set_text(iteration_text)

    # Create animation
    animation = FuncAnimation(fig, update_weights_and_rotation, frames=animation_interval, fargs=(ax, surf, weights_history), interval=50)

    # Adjust layout before saving
    plt.tight_layout()

    # Save animation as a video using PillowWriter
    animation.save('one_neuron/training_process_and_surface.gif', writer=PillowWriter(fps=30))


plot_surface_training(weights_history, loss_values, point_coordinates, point_labels)
#plot_surface(weights1, weights2, loss_values)