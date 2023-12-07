# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from numpy.random import seed
import os

# Import the tensorflow module to create a neural network
import tensorflow as tf
from tensorflow.data import Dataset

# Import function to split data into train and test data
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

# Import some common functions created for this notebook
import common


def plot_zoom(variable: dict, data_frames: dict, zoom_range: list, new_y_max: float):
    """
    Plot the distribution of a variable for different processes and zoom in on a certain range.

    Parameters:
        variable (dict): The variable to plot.
        data_frames (dict): The data frames of the different processes.
        zoom_range (list): The range to zoom in on.
        new_y_max (float): The new maximum y value.
    """
    labels = []
    events = []
    weights = []
    colors = []
    for process in process_order:
        if process not in data_frames:
            continue
        values = data_frames[process]
        labels.append(process)
        events.append(np.array(values[variable['variable']]))
        weights.append(values['totalWeight'])
        colors.append(process_color[process])

    def _get_bins(zoom):
        bin_len = len(variable['binning'])
        left = zoom * zoom_range[0] + (1 - zoom) * variable['binning'][0]
        right = zoom * zoom_range[1] + (1 - zoom) * variable['binning'][-1]
        return np.linspace(left, right, bin_len)

    # Create figure and axes
    fig, ax = plt.subplots()
    zoom = 0
    hist_simulation = ax.hist(events,
                              weights=weights,
                              bins=variable['binning'],
                              label=labels,
                              color=colors,
                              stacked=True)

    # Style
    if 'binning' in variable:
        plt.xlim(variable['binning'][0], variable['binning'][-1])
    else:
        plt.xlim(hist_simulation[1][0], hist_simulation[1][-1])
    x_min, x_max = plt.xlim()
    ax.set_title('{} distribution'.format(variable['title']))
    ax.set(ylabel='Events')
    ax.set_ylim(bottom=0)
    ax.legend()
    # Get current max y value
    _, y_max = plt.ylim()
    ax.set(xlabel=variable['xlabel'])

    n_frames = 200

    def update(frame):
        nonlocal zoom
        zoom = frame / (n_frames - 1)
        if zoom <= 0.4:
            zoom_vertical = (zoom / 0.4) ** 0.3
            new_x_min = x_min * (1 - zoom_vertical) + zoom_range[0] * zoom_vertical
            new_x_max = x_max * (1 - zoom_vertical) + zoom_range[1] * zoom_vertical
            ax.set_xlim(new_x_min, new_x_max)
        elif zoom < 0.6:
            ax.set_xlim(zoom_range[0], zoom_range[1])
            return
        else:
            zoom_horizontal = ((zoom - 0.6) / 0.4) ** 0.3
            ax.set_ylim(0, new_y_max * zoom_horizontal + y_max * (1 - zoom_horizontal))
        return 
    
    # Animate the plot
    animation = FuncAnimation(fig, update, frames=n_frames, interval=100, repeat=False)

    # Save the animation
    animation.save('apply/zoom_{}.gif'.format(variable['variable']), writer=PillowWriter(fps=30))


# Order to plot
process_order = ['llll', 'Zee', 'Zmumu', 'ttbar_lep', 'VBFH125_ZZ4lep', 'WH125_ZZ4lep', 'ZH125_ZZ4lep',
                    'ggH125_ZZ4lep']
# Colors of processes
process_color = {
    'llll': 'blue',
    'Zee': 'blueviolet',
    'Zmumu': 'purple',
    'ttbar_lep': 'green',
    'VBFH125_ZZ4lep': 'gold',
    'WH125_ZZ4lep': 'orange',
    'ZH125_ZZ4lep': 'sienna',
    'ggH125_ZZ4lep': 'red'
}

# Random state
random_state = 21
_ = np.random.RandomState(random_state)

# Define the input samples
sample_list_signal = ['ggH125_ZZ4lep', 'VBFH125_ZZ4lep', 'WH125_ZZ4lep', 'ZH125_ZZ4lep']
sample_list_background = ['llll', 'Zee', 'Zmumu', 'ttbar_lep']

sample_path = 'input'
# Read all the samples
no_selection_data_frames = {}
for sample in sample_list_signal + sample_list_background:
    data = pd.read_csv(os.path.join(sample_path, sample + '.csv'))
    no_selection_data_frames[sample] = pd.read_csv(os.path.join(sample_path, sample + '.csv'))

# Create a copy of the original data frame to investigate later
data_frames = no_selection_data_frames.copy()

# Apply the chosen selection criteria
for sample in sample_list_signal + sample_list_background:
    # Selection on lepton type
    type_selection = np.vectorize(common.selection_lepton_type)(
        data_frames[sample].lep1_pdgId,
        data_frames[sample].lep2_pdgId,
        data_frames[sample].lep3_pdgId,
        data_frames[sample].lep4_pdgId)
    data_frames[sample] = data_frames[sample][type_selection]

    # Selection on lepton charge
    charge_selection = np.vectorize(common.selection_lepton_charge)(
        data_frames[sample].lep1_charge,
        data_frames[sample].lep2_charge,
        data_frames[sample].lep3_charge,
        data_frames[sample].lep4_charge)
    data_frames[sample] = data_frames[sample][charge_selection]

# Split data to keep 40% for testing
train_data_frames, test_data_frames = common.split_data_frames(data_frames, 0.6)

# The training input variables
training_variables = ['lep1_pt', 'lep2_pt', 'lep3_pt', 'lep4_pt']
training_variables += ['lep1_e', 'lep2_e', 'lep3_e', 'lep4_e']
training_variables += ['lep1_charge', 'lep2_charge', 'lep3_charge', 'lep4_charge']
training_variables += ['lep1_pdgId', 'lep2_pdgId', 'lep3_pdgId', 'lep4_pdgId']
training_variables += ['lep1_phi', 'lep2_phi', 'lep3_phi', 'lep4_phi']
training_variables += ['lep1_eta', 'lep2_eta', 'lep3_eta', 'lep4_eta']

# Load the model
model = tf.keras.models.load_model(f'models/medium_dnn_3/early_stopping_model')

# Apply the model on the test data
data_frames_apply_dnn = common.apply_dnn_model(model, test_data_frames, training_variables, sample_list_signal + sample_list_background)
model_prediction = {'variable': 'model_prediction',
                    'binning': np.linspace(0, 1, 100),
                    'xlabel': 'NN Output Score',
                    'title': 'Histogram of NN Output Scores'}

plot_zoom(model_prediction, data_frames_apply_dnn, [0.9, 1], 15)
common.plot_hist(model_prediction, data_frames_apply_dnn, show_data=False)
