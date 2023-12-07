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

# Import some common functions created for this notebook
import common


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


def get_events(data: pd.DataFrame, n_events: int) -> pd.DataFrame:
    """
    Get a random sample of events from the data frame.

    Parameters:
        data (pd.DataFrame): The data frame to get the events from.
        n_events (int): The number of events to get.

    Returns:
        pd.DataFrame: The data frame with the random events.
    """
    # Get the number of random events
    n_events = round(min(n_events, len(data)))
    # Get the random events
    random_events = data.sample(n_events, random_state=random_state)
    return random_events


def plot_raw_events(variable: dict, data_frames: dict):
    """
    Animate the simulation of raw events for each process.

    Parameters:
        variable (dict): The variable to plot.
        data_frames (dict): The data frames for each process.
    """
    # How many Events are expected per process
    expected_events = {}
    raw_events = {}
    for process in process_order:
        expected_events[process] = np.sum(data_frames[process]['totalWeight'])
        raw_events[process] = len(data_frames[process]['totalWeight'])

    def _get_n_events(more_events: float) -> dict:
        """
        Get the number of events for each process.

        Parameters:
            more_events (float): Parameter to control the number of events.

        Returns:
            dict: The data frames for each process.
        """
        current_events = {}
        for process in process_order:
            velocity = 8
            current_events[process] = expected_events[process] + (np.exp(more_events * velocity) - 1) / (np.exp(velocity) - 1) * (raw_events[process] - expected_events[process])
        current_dataframes = {}
        for process in process_order:
            current_dataframes[process] = get_events(data_frames[process], current_events[process])
            current_dataframes[process]['totalWeight'] = 1
        return current_dataframes
    
    def _get_hist(current_dataframes: dict) -> (list, list, list, list):
        """
        Get the histogram values for each process.

        Parameters:
            current_dataframes (dict): The data frames for each process.

        Returns:
            list: The labels of the processes.
            list: The events for each process.
            list: The weights for each process.
            list: The colors for each process.
        """
        # Extract the histogram values
        labels = []
        events = []
        weights = []
        colors = []
        for process in process_order:
            if process not in current_dataframes:
                continue
            values = current_dataframes[process]
            labels.append(process)
            events.append(np.array(values[variable['variable']]))
            weights.append(values['totalWeight'])
            colors.append(process_color[process])
        return labels, events, weights, colors

    # Create figure and axes
    fig, ax = plt.subplots()
    more_events = 0
    current_dataframes = _get_n_events(more_events)
    labels, events, weights, colors = _get_hist(current_dataframes)
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
    ax.set_title('{} distribution'.format(variable['title']))
    ax.set(ylabel='Events')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set(xlabel=variable['xlabel'])

    n_frames = 400

    def update(frame):
        nonlocal more_events
        more_events = frame/(n_frames - 1)
        current_dataframes = _get_n_events(more_events)
        labels, events, weights, colors = _get_hist(current_dataframes)

        # Clear the previous plot
        ax.clear()

        # Plot the new histogram for each frame
        hist_simulation = ax.hist(events,
                                weights=weights,
                                bins=var_lep1_pt['binning'],
                                label=labels,
                                color=colors,
                                stacked=True)

        # Style
        if 'binning' in var_lep1_pt:
            plt.xlim(var_lep1_pt['binning'][0], var_lep1_pt['binning'][-1])
        else:
            plt.xlim(hist_simulation[1][0], hist_simulation[1][-1])
        ax.set_title('{} distribution'.format(var_lep1_pt['title']))
        ax.set(ylabel='Events')
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.set(xlabel=var_lep1_pt['xlabel'])

        return hist_simulation
    
    # Animate the plot
    animation = FuncAnimation(fig, update, frames=n_frames, interval=100, repeat=False)

    # Save the animation
    animation.save('variables/raw_events_{}.gif'.format(variable['variable']), writer=PillowWriter(fps=30))


def plot_event_weights(variable: dict, data_frames: dict):
    """
    Animation of the event weights for each process.

    Parameters:
        variable (dict): The variable to plot.
        data_frames (dict): The data frames for each process.
    """
    # How many Events are expected per process
    expected_events = {}
    raw_events = {}
    for process in process_order:
        expected_events[process] = np.sum(data_frames[process]['totalWeight'])
        raw_events[process] = len(data_frames[process]['totalWeight'])

    def _get_weighted_events(apply_weight: float) -> dict:
        """
        Get the weighted events for each process.

        Parameters:
            apply_weight (float): Parameter to control the weights.

        Returns:
            dict: The data frames for each process.
        """
        current_dataframes = {}
        for process in process_order:
            current_dataframes[process] = data_frames[process].copy()
            velocity = 8
            current_weight = np.exp(-apply_weight * velocity) - np.exp(-velocity) + apply_weight * data_frames[process]['totalWeight']
            current_dataframes[process]['totalWeight'] = current_weight
        return current_dataframes
    
    def _get_hist(current_dataframes: dict) -> (list, list, list, list):
        """
        Get the histogram values for each process.

        Parameters:
            current_dataframes (dict): The data frames for each process.

        Returns:
            list: The labels of the processes.
            list: The events for each process.
            list: The weights for each process.
            list: The colors for each process.
        """
        # Extract the histogram values
        labels = []
        events = []
        weights = []
        colors = []
        for process in process_order:
            if process not in current_dataframes:
                continue
            values = current_dataframes[process]
            labels.append(process)
            events.append(np.array(values[variable['variable']]))
            weights.append(values['totalWeight'])
            colors.append(process_color[process])
        return labels, events, weights, colors

    # Create figure and axes
    fig, ax = plt.subplots()
    apply_weight = 0
    current_dataframes = _get_weighted_events(apply_weight)
    labels, events, weights, colors = _get_hist(current_dataframes)
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
    ax.set_title('{} distribution'.format(variable['title']))
    ax.set(ylabel='Events')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set(xlabel=variable['xlabel'])

    n_frames = 400

    def update(frame):
        nonlocal apply_weight
        apply_weight = frame / (n_frames - 1)
        current_dataframes = _get_weighted_events(apply_weight)
        labels, events, weights, colors = _get_hist(current_dataframes)

        # Clear the previous plot
        ax.clear()

        # Plot the new histogram for each frame
        hist_simulation = ax.hist(events,
                                weights=weights,
                                bins=var_lep1_pt['binning'],
                                label=labels,
                                color=colors,
                                stacked=True)

        # Style
        if 'binning' in var_lep1_pt:
            plt.xlim(var_lep1_pt['binning'][0], var_lep1_pt['binning'][-1])
        else:
            plt.xlim(hist_simulation[1][0], hist_simulation[1][-1])
        ax.set_title('{} distribution'.format(var_lep1_pt['title']))
        ax.set(ylabel='Events')
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.set(xlabel=var_lep1_pt['xlabel'])

        return hist_simulation
    
    # Animate the plot
    animation = FuncAnimation(fig, update, frames=n_frames, interval=100, repeat=False)

    # Save the animation
    animation.save('variables/weighted_events_{}.gif'.format(variable['variable']), writer=PillowWriter(fps=30))


def plot_normalisation(variable: dict, data_frames: dict):
    """
    Animation of the normalisation of the variable distribution.

    Parameters:
        variable (dict): The variable to plot.
        data_frames (dict): The data frames for each process.
    """
    # How many Events are expected per process
    expected_events = {}
    raw_events = {}
    for process in process_order:
        expected_events[process] = np.sum(data_frames[process]['totalWeight'])
        raw_events[process] = len(data_frames[process]['totalWeight'])
    
    def _get_hist(current_dataframes: dict) -> (list, list, list, list):
        """
        Get the histogram values for each process.

        Parameters:
            current_dataframes (dict): The data frames for each process.

        Returns:
            list: The labels of the processes.
            list: The events for each process.
            list: The weights for each process.
            list: The colors for each process.
        """
        # Extract the histogram values
        labels = []
        events = []
        weights = []
        colors = []
        for process in process_order:
            if process not in current_dataframes:
                continue
            values = current_dataframes[process]
            labels.append(process)
            events.append(np.array(values[variable['variable']]))
            weights.append(values['totalWeight'])
            colors.append(process_color[process])
        return labels, events, weights, colors

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(7, 7))
    apply_scale = 0
    labels, events, weights, colors = _get_hist(data_frames)
    hist_simulation = ax.hist(events,
                              weights=weights,
                              bins=variable['binning'],
                              label=labels,
                              color=colors,
                              stacked=True)

    # Concatenate the events and weights
    all_events = np.concatenate(events)
    all_weights = np.concatenate(weights)

    # Get weighted mean and standard deviation
    weighted_mean = np.average(all_events, weights=all_weights)
    weighted_std = np.sqrt(np.average((all_events - weighted_mean)**2, weights=all_weights))

    def _normalisation(value: float, apply_scale: float) -> float:
        """
        Normalise the value.

        Parameters:
            value (float): The value to normalise.
            apply_scale (float): Parameter to control the normalisation.

        Returns:
            float: The normalised value.
        """
        return (value - weighted_mean * apply_scale) / (weighted_std * apply_scale + 1 - apply_scale)

    # Style
    x_min = variable['binning'][0]
    x_max = variable['binning'][-1]
    ax.set_xlim(x_min, x_max)
    ax.set_title('{} distribution'.format(variable['title']))
    ax.set(ylabel='Events')
    # Get current max y value
    _, y_max = plt.ylim()
    ax.set_ylim(0, y_max * 1.2)
    ax.legend(loc='upper right')
    ax.set(xlabel=variable['xlabel'])

    n_frames = 100

    def update(frame):
        nonlocal apply_scale
        apply_scale = frame / (n_frames - 1)
        labels, events, weights, colors = _get_hist(data_frames)

        # Normalize the events
        events_normalized = events.copy()
        for i in range(len(events)):
            events_normalized[i] = _normalisation(events[i], apply_scale)
        binning_normalized = _normalisation(variable['binning'], apply_scale)

        # Clear the previous plot
        ax.clear()

        # Plot the new histogram for each frame
        hist_simulation = ax.hist(events_normalized,
                                weights=weights,
                                bins=binning_normalized,
                                label=labels,
                                color=colors,
                                stacked=True)

        # Style
        current_x_min = _normalisation(x_min, apply_scale)
        current_x_max = _normalisation(x_max, apply_scale)
        ax.set_xlim(current_x_min, current_x_max)
        ax.set_title('{} distribution'.format(variable['title']))
        ax.set(ylabel='Events')
        ax.set_ylim(0, y_max * 1.2)
        ax.legend(loc='upper right')

        # Change x label
        x_label = variable['xlabel']
        if '[GeV]' in x_label:
            x_label = f'({x_label.replace("[GeV]", "").strip()} - mean) / std'
        else:
            x_label = f'({x_label.strip()} - mean) / std'
        ax.set(xlabel=x_label)

        return hist_simulation
    
    # Animate the plot
    animation = FuncAnimation(fig, update, frames=n_frames, interval=100, repeat=False)

    # Save the animation
    animation.save('variables/normalisation_{}.gif'.format(variable['variable']), writer=PillowWriter(fps=30))


def plot_zoom(variable: dict, data_frames: dict, zoom_range: list):
    """
    Animation of the zoom of the variable distribution.

    Parameters:
        variable (dict): The variable to plot.
        data_frames (dict): The data frames for each process.
        zoom_range (list): The range of the zoom.
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

    def _get_bins(zoom: float) -> np.ndarray:
        """
        Get the bins for the zoom.

        Parameters:
            zoom (float): The zoom value.

        Returns:
            np.ndarray: The bins for the zoom.
        """
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
    ax.set_title('{} distribution'.format(variable['title']))
    ax.set(ylabel='Events')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set(xlabel=variable['xlabel'])

    n_frames = 100

    def update(frame):
        nonlocal zoom
        zoom = frame / (n_frames - 1)
        current_bins = _get_bins(zoom)
        ax.set_xlim(current_bins[0], current_bins[-1])
        return 
    
    # Animate the plot
    animation = FuncAnimation(fig, update, frames=n_frames, interval=100, repeat=False)

    # Save the animation
    animation.save('variables/zoom_{}.gif'.format(variable['variable']), writer=PillowWriter(fps=30))


# Variables
var_lep1_pt = {'variable': 'lep1_pt',
               'title': '$p_T$ (lep 1)',
               'binning': np.linspace(0, 200, 50),
               'xlabel': '$p_T$ (lep 1) [GeV]'}

var_lep1_eta = {'variable': 'lep1_eta',
                'title': '$\eta$ (lep 1)',
                'binning': np.linspace(-2.5, 2.5, 20),
                'xlabel': '$\eta$ (lep 1)'}

var_lep1_phi = {'variable': 'lep1_phi',
                'title': '$\phi$ (lep 1)',
                'binning': np.linspace(-np.pi, np.pi, 20),
                'xlabel': '$\phi$ (lep 1)'}

var_lep1_E = {'variable': 'lep1_e',
                'title': '$E$ (lep 1)',
                'binning': np.linspace(0, 500, 50),
                'xlabel': '$E$ (lep 1) [GeV]'}

var_lep1_charge = {'variable': 'lep1_charge',
                'title': 'Charge (lep 1)',
                'binning': np.linspace(-1.5, 1.5, 4),
                'xlabel': 'Charge (lep 1)'}

var_lep1_pdgId = {'variable': 'lep1_pdgId',
                'title': 'PDG ID (lep 1)',
                'binning': np.linspace(10.5, 13.5, 4),
                'xlabel': 'PDG ID (lep 1)'}

var_lep2_pt = {'variable': 'lep2_pt',
                'title': '$p_T$ (lep 2)',
                'binning': np.linspace(0, 200, 50),
                'xlabel': '$p_T$ (lep 2) [GeV]'}

var_lep2_eta = {'variable': 'lep2_eta',
                'title': '$\eta$ (lep 2)',
                'binning': np.linspace(-2.5, 2.5, 20),
                'xlabel': '$\eta$ (lep 2)'}

var_lep2_phi = {'variable': 'lep2_phi',
                'title': '$\phi$ (lep 2)',
                'binning': np.linspace(-np.pi, np.pi, 20),
                'xlabel': '$\phi$ (lep 2)'}

var_lep2_E = {'variable': 'lep2_e',
                'title': '$E$ (lep 2)',
                'binning': np.linspace(0, 500, 50),
                'xlabel': '$E$ (lep 2) [GeV]'}

var_lep2_charge = {'variable': 'lep2_charge',
                'title': 'Charge (lep 2)',
                'binning': np.linspace(-1.5, 1.5, 4),
                'xlabel': 'Charge (lep 2)'}

var_lep2_pdgId = {'variable': 'lep2_pdgId',
                'title': 'PDG ID (lep 2)',
                'binning': np.linspace(10.5, 13.5, 4),
                'xlabel': 'PDG ID (lep 2)'}

var_lep3_pt = {'variable': 'lep3_pt',
                'title': '$p_T$ (lep 3)',
                'binning': np.linspace(0, 200, 50),
                'xlabel': '$p_T$ (lep 3) [GeV]'}

var_lep3_eta = {'variable': 'lep3_eta',
                'title': '$\eta$ (lep 3)',
                'binning': np.linspace(-2.5, 2.5, 20),
                'xlabel': '$\eta$ (lep 3)'}

var_lep3_phi = {'variable': 'lep3_phi',
                'title': '$\phi$ (lep 3)',
                'binning': np.linspace(-np.pi, np.pi, 20),
                'xlabel': '$\phi$ (lep 3)'}

var_lep3_E = {'variable': 'lep3_e',
                'title': '$E$ (lep 3)',
                'binning': np.linspace(0, 500, 50),
                'xlabel': '$E$ (lep 3) [GeV]'}

var_lep3_charge = {'variable': 'lep3_charge',
                'title': 'Charge (lep 3)',
                'binning': np.linspace(-1.5, 1.5, 4),
                'xlabel': 'Charge (lep 3)'}

var_lep3_pdgId = {'variable': 'lep3_pdgId',
                'title': 'PDG ID (lep 3)',
                'binning': np.linspace(10.5, 13.5, 4),
                'xlabel': 'PDG ID (lep 3)'}

var_lep4_pt = {'variable': 'lep4_pt',
                'title': '$p_T$ (lep 4)',
                'binning': np.linspace(0, 200, 50),
                'xlabel': '$p_T$ (lep 4) [GeV]'}

var_lep4_eta = {'variable': 'lep4_eta',
                'title': '$\eta$ (lep 4)',
                'binning': np.linspace(-2.5, 2.5, 20),
                'xlabel': '$\eta$ (lep 4)'}

var_lep4_phi = {'variable': 'lep4_phi',
                'title': '$\phi$ (lep 4)',
                'binning': np.linspace(-np.pi, np.pi, 20),
                'xlabel': '$\phi$ (lep 4)'}

var_lep4_E = {'variable': 'lep4_e',
                'title': '$E$ (lep 4)',
                'binning': np.linspace(0, 500, 50),
                'xlabel': '$E$ (lep 4) [GeV]'}

var_lep4_charge = {'variable': 'lep4_charge',
                'title': 'Charge (lep 4)',
                'binning': np.linspace(-1.5, 1.5, 4),
                'xlabel': 'Charge (lep 4)'}

var_lep4_pdgId = {'variable': 'lep4_pdgId',
                'title': 'PDG ID (lep 4)',
                'binning': np.linspace(10.5, 13.5, 4),
                'xlabel': 'PDG ID (lep 4)'}

var_m_llll = {'variable': 'lep_m_llll',
                'title': '$m_{llll}$',
                'binning': np.linspace(50, 500, 91),
                'xlabel': '$m_{llll}$ [GeV]'}

variables = [
    var_lep1_pt, var_lep1_eta, var_lep1_phi, var_lep1_E, var_lep1_charge, var_lep1_pdgId,
    var_lep2_pt, var_lep2_eta, var_lep2_phi, var_lep2_E, var_lep2_charge, var_lep2_pdgId,
    var_lep3_pt, var_lep3_eta, var_lep3_phi, var_lep3_E, var_lep3_charge, var_lep3_pdgId,
    var_lep4_pt, var_lep4_eta, var_lep4_phi, var_lep4_E, var_lep4_charge, var_lep4_pdgId,
    var_m_llll
]

# Create a copy of the original data frame to investigate later
data_frames = no_selection_data_frames.copy()
type_selection_data_frames = no_selection_data_frames.copy()
charge_selection_data_frames = no_selection_data_frames.copy()

# Apply selections
for sample in sample_list_signal + sample_list_background:
    # Selection on lepton type
    type_selection = np.vectorize(common.selection_lepton_type)(
        data_frames[sample].lep1_pdgId,
        data_frames[sample].lep2_pdgId,
        data_frames[sample].lep3_pdgId,
        data_frames[sample].lep4_pdgId)

    # Selection on lepton charge
    charge_selection = np.vectorize(common.selection_lepton_charge)(
        data_frames[sample].lep1_charge,
        data_frames[sample].lep2_charge,
        data_frames[sample].lep3_charge,
        data_frames[sample].lep4_charge)
    
    # Apply selections
    type_selection_data_frames[sample] = data_frames[sample][type_selection]
    charge_selection_data_frames[sample] = data_frames[sample][charge_selection]
    data_frames[sample] = data_frames[sample][type_selection & charge_selection]

#plot_raw_events(var_lep1_pt, no_selection_data_frames)
#plot_event_weights(var_lep1_pt, no_selection_data_frames)
#plot_zoom(var_m_llll, data_frames, [100, 210])

common.plot_hist(var_lep1_pt, no_selection_data_frames)
plt.ylim(0, 90)
plt.title(var_lep1_pt['title'])
plt.savefig('variables/no_selection_lep1_pt.png', dpi=300, bbox_inches='tight')

common.plot_hist(var_lep1_pt, type_selection_data_frames)
plt.ylim(0, 90)
plt.title(var_lep1_pt['title'])
plt.savefig('variables/type_selection_lep1_pt.png', dpi=300, bbox_inches='tight')

common.plot_hist(var_lep1_pt, charge_selection_data_frames)
plt.ylim(0, 90)
plt.title(var_lep1_pt['title'])
plt.savefig('variables/charge_selection_lep1_pt.png', dpi=300, bbox_inches='tight')

common.plot_hist(var_lep1_pt, data_frames)
plt.ylim(0, 90)
plt.title(var_lep1_pt['title'])
plt.savefig('variables/full_selection_lep1_pt.png', dpi=300, bbox_inches='tight')

print_event_numbers = False
if print_event_numbers:
    # Print the number of events after each selection
    print('No selection:')
    for sample in sample_list_signal + sample_list_background:
        spaces = ' ' * (17 - len(sample) - max(0,int(np.log10(np.sum(no_selection_data_frames[sample]["totalWeight"])))))
        fraction = np.sum(no_selection_data_frames[sample]["totalWeight"]) / np.sum(no_selection_data_frames[sample]["totalWeight"])
        fraction_space = ' ' * (2 - max(0,int(np.log10(fraction))))
        print(f'{sample}:{spaces}{np.sum(no_selection_data_frames[sample]["totalWeight"]):.3f}{fraction_space}({100*fraction:.0f}%)')

    print('\nWith type selection:')
    for sample in sample_list_signal + sample_list_background:
        spaces = ' ' * (17 - len(sample) - max(0,int(np.log10(np.sum(type_selection_data_frames[sample]["totalWeight"])))))
        fraction = np.sum(type_selection_data_frames[sample]["totalWeight"]) / np.sum(no_selection_data_frames[sample]["totalWeight"])
        fraction_space = ' ' * (2 - max(0,int(np.log10(fraction))))
        print(f'{sample}:{spaces}{np.sum(type_selection_data_frames[sample]["totalWeight"]):.3f}{fraction_space}({100*fraction:.0f}%)')

    print('\nWith charge selection:')
    for sample in sample_list_signal + sample_list_background:
        spaces = ' ' * (17 - len(sample) - max(0,int(np.log10(np.sum(charge_selection_data_frames[sample]["totalWeight"])))))
        fraction = np.sum(charge_selection_data_frames[sample]["totalWeight"]) / np.sum(no_selection_data_frames[sample]["totalWeight"])
        fraction_space = ' ' * (2 - max(0,int(np.log10(fraction))))
        print(f'{sample}:{spaces}{np.sum(charge_selection_data_frames[sample]["totalWeight"]):.3f}{fraction_space}({100*fraction:.0f}%)')

    print('\nWith type and charge selection:')
    for sample in sample_list_signal + sample_list_background:
        spaces = ' ' * (17 - len(sample) - max(0,int(np.log10(np.sum(data_frames[sample]["totalWeight"])))))
        fraction = np.sum(data_frames[sample]["totalWeight"]) / np.sum(no_selection_data_frames[sample]["totalWeight"])
        fraction_space = ' ' * (2 - max(0,int(np.log10(fraction))))
        print(f'{sample}:{spaces}{np.sum(data_frames[sample]["totalWeight"]):.3f}{fraction_space}({100*fraction:.0f}%)')

variables = [var_lep1_pt, var_lep1_E, var_lep4_eta]
for variable in variables:
    # Plot the normalization
    print(variable['variable'])
    plot_normalisation(variable, data_frames)

    # Plot the variable
    common.plot_hist(variable, data_frames)
    plt.title(variable['title'])
    if 'pdgId' in variable['variable']:
        plt.xticks([11, 13], ['$e^\pm$', '$\mu^\pm$'])
    if 'charge' in variable['variable']:
        plt.xticks([-1, 0, 1])
    plt.legend(loc='upper right')
    # Get current max y value
    _, y_max = plt.ylim()
    plt.ylim(0, y_max * 1.2)
    plt.savefig('variables/var_{}.png'.format(variable['variable']), dpi=300)#, bbox_inches='tight')
    plt.clf()
