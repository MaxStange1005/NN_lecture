import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Set random seed
np.random.seed(0)


def generate_values(function: callable, x_values: np.ndarray, n_values: int) -> np.ndarray:
    """
    Generate random values from a function.

    Parameters:
        function (callable): The function to generate random values from.
        x_values (np.ndarray): The x values of the function.
        n_values (int): The number of random values to generate.

    Returns:
        np.ndarray: The generated random values.
    """
    # Integrate f(x) to get the cumulative distribution function (CDF)
    pdf_values = function(x_values)
    cdf_values = integrate.cumtrapz(pdf_values, x_values, initial=0)

    # Generate random values between 0 and 1
    random_values = np.random.rand(n_values)

    # Use the inverse of the CDF to map random values to x values
    inverse_cdf = interp1d(cdf_values, x_values, kind='linear', fill_value="extrapolate")
    generated_values = inverse_cdf(random_values)
    return generated_values


def roc_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the ROC value.

    Parameters:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        float: The ROC value.
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    sort_index = np.argsort(fpr)
    fpr = fpr[sort_index]
    tpr = tpr[sort_index]
    roc_auc = auc(fpr, tpr)
    return roc_auc


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the binary cross entropy.

    Parameters:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        float: The binary cross entropy.
    """
    return np.sum(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))) / len(y_true)


def get_gauss_function(scale_list: list, mean_list: list, std_list: list) -> callable:
    """
    Get the sum of Gaussian functions.

    Parameters:
        scale_list (list): The scale of the Gaussian functions.
        mean_list (list): The mean of the Gaussian functions.
        std_list (list): The standard deviation of the Gaussian functions.

    Returns:
        callable: The sum of Gaussian functions.
    """
    def gauss_function(x):
        # Sum the Gaussian functions
        gauss_sum = np.zeros_like(x)
        for scale, mean, std in zip(scale_list, mean_list, std_list):
            gauss_sum += scale * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
        
        # Normalize the Gaussian function
        gauss_sum /= integrate.simps(gauss_sum, x)
        return gauss_sum

    return gauss_function


def plot_roc_curve(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> (plt.Figure, plt.Axes):
    """
    Plot the ROC curve.

    Parameters:
        name (str): The name of the plot.
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        plt.Figure: The figure.
        plt.Axes: The axes.
    """
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, )

    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.5, label='random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc="lower right", fontsize=14)

    fig.savefig(f'roc_vs_bce/{name}_roc.png', dpi=300, bbox_inches='tight')
    return fig, ax


def plot_prediction(name: str, signal_gauss_values: dict, bkg_gauss_values: dict, n_values: int = 1000):
    """
    Plot the prediction of a neural network with the corresponding ROC value and loss value.

    Parameters:
        name (str): The name of the plot.
        signal_gauss_values (dict): The values of the signal Gaussian function.
        bkg_gauss_values (dict): The values of the background Gaussian function.
        n_values (int): The number of values to generate.
    """
    # Plot Gaussian function
    fig, ax = plt.subplots(figsize=(10, 7))
    x_values = np.linspace(0, 1, 1000)

    # Plot the Gaussian function
    signal_scale = signal_gauss_values['scale']
    signal_mean_list = signal_gauss_values['mean']
    signal_std_list = signal_gauss_values['std']
    bkg_scale = bkg_gauss_values['scale']
    bkg_mean_list = bkg_gauss_values['mean']
    bkg_std_list = bkg_gauss_values['std']
    signal_function = get_gauss_function(signal_scale, signal_mean_list, signal_std_list)
    bkg_function = get_gauss_function(bkg_scale, bkg_mean_list, bkg_std_list)

    # Create random values
    signal_values = generate_values(signal_function, x_values, n_values)
    signal_label = np.ones_like(signal_values)
    bkg_values = generate_values(bkg_function, x_values, n_values)
    bkg_label = np.zeros_like(bkg_values)

    # Plot the distributions
    ax.hist(signal_values, bins=50, range=(0, 1), label='Signal', color='darkorange', histtype='step', weights=np.ones_like(signal_values) / len(signal_values))
    ax.hist(bkg_values, bins=50, range=(0, 1), label='Background', color='blue', histtype='step', weights=np.ones_like(bkg_values) / len(bkg_values))

    # Get the ROC value and loss value
    y_true = np.concatenate([signal_label, bkg_label])
    y_pred = np.concatenate([signal_values, bkg_values])
    roc_auc = roc_value(y_true, y_pred)
    binary_loss = binary_cross_entropy(y_true, y_pred)

    ax.set_xlim(0, 1)
    ax.set_xlabel('NN Output Score', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Histogram of NN Output Scores', fontsize=14)
    ax.legend(loc='upper right', fontsize=14)

    # Increase the y-axis limit
    _, y_max = plt.ylim()
    ax.set_ylim(0, 1.2 * y_max)
    fig.savefig(f'roc_vs_bce/{name}.png', dpi=300, bbox_inches='tight')

    # Add text
    ax.text(0.02, 1.1 * y_max, f'ROC AUC = {roc_auc:.4f}', ha='left', va='center', fontsize=14)
    ax.text(0.02, 1.05 * y_max, f'Binary Cross Entropy = {binary_loss:.4f}', ha='left', va='center', fontsize=14)

    fig.savefig(f'roc_vs_bce/{name}_solution.png', dpi=300, bbox_inches='tight')
    print(name)
    print(f'ROC = {roc_auc:.4f}')
    print(f'BCE = {binary_loss:.4f}')

    # Plot ROC curve
    plot_roc_curve(name, y_true, y_pred)


n_values = 10000
# Example 1
signal_gauss_values = {
    'scale': [1, 1],
    'mean': [0.6, 0.7],
    'std': [0.2, 0.1]
}
bkg_gauss_values = {
    'scale': [1, 1],
    'mean': [0.3, 0.4],
    'std': [0.1, 0.2]
}
plot_prediction('example1_left', signal_gauss_values, bkg_gauss_values, n_values)

signal_gauss_values = {
    'scale': [0.4, 1],
    'mean': [0.6, 0.8],
    'std': [0.1, 0.1]
}
bkg_gauss_values = {
    'scale': [1, 0.4],
    'mean': [0.2, 0.4],
    'std': [0.1, 0.1]
}
plot_prediction('example1_right', signal_gauss_values, bkg_gauss_values, n_values)

# Example 2

signal_gauss_values = {
    'scale': [0.6, 1],
    'mean': [0.6, 0.8],
    'std': [0.1, 0.1]
}
bkg_gauss_values = {
    'scale': [0.4],
    'mean': [0.4],
    'std': [0.1]
}
plot_prediction('example2_left', signal_gauss_values, bkg_gauss_values, n_values)

signal_gauss_values = {
    'scale': [0.6, 1],
    'mean': [0.75, 0.85],
    'std': [0.04, 0.05]
}
bkg_gauss_values = {
    'scale': [0.4, 0.5],
    'mean': [0.45, 0.6],
    'std': [0.1, 0.03]
}
plot_prediction('example2_right', signal_gauss_values, bkg_gauss_values, n_values)

# Example 3
signal_gauss_values = {
    'scale': [0.5, 1],
    'mean': [0.7, 0.75],
    'std': [0.05, 0.02]
}
bkg_gauss_values = {
    'scale': [2, 2, 0.15],
    'mean': [0.15, 0.3, 0.45],
    'std': [0.07, 0.1, 0.4]
}
plot_prediction('example3_left', signal_gauss_values, bkg_gauss_values, n_values)

signal_gauss_values = {
    'scale': [0.5, 1],
    'mean': [0.7, 0.85],
    'std': [0.15, 0.1]
}
bkg_gauss_values = {
    'scale': [2, 0.5, 0.2],
    'mean': [0.1, 0.2, 0.5],
    'std': [0.07, 0.15, 0.4]
}
plot_prediction('example3_right', signal_gauss_values, bkg_gauss_values, n_values)

# Example 4
signal_gauss_values = {
    'scale': [1],
    'mean': [0.7],
    'std': [0.1]
}
bkg_gauss_values = {
    'scale': [1],
    'mean': [0.3],
    'std': [0.1]
}
plot_prediction('example4_left', signal_gauss_values, bkg_gauss_values, n_values)

signal_gauss_values = {
    'scale': [1],
    'mean': [0.6],
    'std': [0.02]
}
bkg_gauss_values = {
    'scale': [1],
    'mean': [0.4],
    'std': [0.02]
}
plot_prediction('example4_right', signal_gauss_values, bkg_gauss_values, n_values)