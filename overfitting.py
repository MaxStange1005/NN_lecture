import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
from typing import Optional

# Rangdom seed
np.random.seed(420)

def get_sample_data(x_range: list, size: int) -> (np.ndarray, np.ndarray):
    """
    Create sample data.

    Parameters:
        x_range (list): The range of the x values.
        size (int): The number of data points.

    Returns:
        np.ndarray: The x values.
        np.ndarray: The y values.
    """
    # Create random x values
    x_values = x_range[0] + np.random.rand(size) * (x_range[1] - x_range[0])
    y_data = x_values ** 3 + 2 * x_values ** 2 - 30 * x_values + 1 + np.random.normal(scale=30, size=len(x_values))
    y_data = y_data / 50
    return x_values, y_data


def chi_square(x_data: np.ndarray, y_data: np.ndarray, fit_function, params: tuple) -> float:
    """
    Calculate the MSE value.

    Parameters:
        x_data (np.ndarray): The x values.
        y_data (np.ndarray): The y values.
        fit_function (function): The fit function.
        params (tuple): The parameters of the fit function.

    Returns:
        float: The MSE value.
    """
    predicted_y = fit_function(x_data, *params)
    residuals = y_data - predicted_y
    chi_square = np.sum(residuals**2)
    return chi_square


def visualize_fit(name: str, x_data: np.ndarray, y_data: np.ndarray, x_range: list, fit_function, p0: tuple, val_x_data: Optional[np.ndarray] = None, val_y_data: Optional[np.ndarray] = None):
    """
    Visualize the fit.

    Parameters:
        name (str): The name of the fit.
        x_data (np.ndarray): The x values.
        y_data (np.ndarray): The y values.
        x_range (list): The range of the x values.
        fit_function (function): The fit function.
        p0 (tuple): The initial parameters of the fit function.
        val_x_data (Optional[np.ndarray]): The validation x values.
        val_y_data (Optional[np.ndarray]): The validation y values.
    """
    print(f'Fitting {name}...')
    # Define the objective function to minimize (chi-square)
    def _chi_square_objective(params):
        return chi_square(x_data, y_data, fit_function, params)
    
    # Callback function to store parameter values
    param_history = [p0]

    def callback(params):
        param_history.append(params)

    # Set the maximum number of iterations (adjust as needed)
    max_iterations = 50

    # Set other options for the minimization process
    options = {'maxiter': max_iterations, 'gtol': 1e-8, 'xrtol': 1e-10}

    # Perform the optimization using minimize
    result = minimize(_chi_square_objective, p0, callback=callback, method='BFGS', options=options)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the data
    ax.scatter(x_data, y_data, label='Data', s=10, alpha=0.5, c='blue')
    if val_x_data is not None and val_y_data is not None:
        ax.scatter(val_x_data, val_y_data, label='Validation Data', s=10, alpha=0.5, c='green')

    # Plot the fit function
    fit_range = np.linspace(*x_range, 100)
    fit_line, = ax.plot([], [], label='Fit', c='darkorange')
    chi_square_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, va='top', ha='left')

    def update(frame):
        fit_line.set_data(fit_range, fit_function(fit_range, *param_history[frame]))
        chi_square_data = _chi_square_objective(param_history[frame])
        ax.legend(loc='upper right')
        if val_x_data is not None and val_y_data is not None:
            chi_square_validation = chi_square(val_x_data, val_y_data, fit_function, param_history[frame])
            chi_square_text.set_text(f'$\\sum_i (f_{{fit}}(x_{{data}}) - y_{{data}})^2 = {chi_square_data:.3f}$\n$\\sum_i (f_{{fit}}(x_{{validation}}) - y_{{validation}})^2 = {chi_square_validation:.3f}$')
        else:
            chi_square_text.set_text(f'$\\sum_i (f_{{fit}}(x_{{data}}) - y_{{data}})^2 = {chi_square_data:.3f}$')
        return fit_line

    # Animate the plot
    animation = FuncAnimation(fig, update, frames=len(param_history), interval=100, repeat=False)

    ax.set_xlim(*x_range)
    ax.set_ylim(-3, 7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    animation.save(f'overfitting/{name}.gif', writer='imagemagick', fps= len(param_history) / 3)


x_range = [-8, 8]
x_data, y_data = get_sample_data(x_range, 10)
val_x_data, val_y_data = get_sample_data(x_range, 12)

# Visualize the data
fig, ax = plt.subplots()
ax.scatter(x_data, y_data, label='Data', s=10, alpha=0.5, c='blue')
ax.set_xlim(*x_range)
ax.set_ylim(-3, 7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right')
fig.savefig('overfitting/data.png', dpi=300)

fig, ax = plt.subplots()
ax.scatter(x_data, y_data, label='Data', s=10, alpha=0.5, c='blue')
ax.scatter(val_x_data, val_y_data, label='Validation Data', s=10, alpha=0.5, c='green')
ax.set_xlim(*x_range)
ax.set_ylim(-3, 7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right')
fig.savefig('overfitting/data_validation.png', dpi=300)

# Define the fitting functions
def function_2param(x, a, b):
    return a * x + b

def function_3param(x, a, b, c):
    return a * x**2 + b * x + c

def function_4param(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def function_5param(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def function_6param(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

def function_7param(x, a, b, c, d, e, f, g):
    return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g

def function_8param(x, a, b, c, d, e, f, g, h):
    return a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h

def function_9param(x, a, b, c, d, e, f, g, h, i):
    return a * x**8 + b * x**7 + c * x**6 + d * x**5 + e * x**4 + f * x**3 + g * x**2 + h * x + i

def function_10param(x, a, b, c, d, e, f, g, h, i, j):
    return a * x**9 + b * x**8 + c * x**7 + d * x**6 + e * x**5 + f * x**4 + g * x**3 + h * x**2 + i * x + j

for i in range(2, 11):
    visualize_fit(f'fit_{i}param', x_data, y_data, x_range, eval(f'function_{i}param'), np.zeros(i))
    visualize_fit(f'fit_{i}param_validation', x_data, y_data, x_range, eval(f'function_{i}param'), np.zeros(i), val_x_data, val_y_data)