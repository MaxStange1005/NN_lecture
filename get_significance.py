import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import common


def get_model_significances(
        model: tf.keras.Model,
        signal_values: np.ndarray,
        bkg_values: np.ndarray,
        signal_weights: np.ndarray,
        bkg_weights: np.ndarray) -> (float, float, float, float):
    """
    Calculate the significance for different cut values in a for loop.

    Parameters:
        model (tf.keras.Model): The model.
        signal_values (np.ndarray): The signal values.
        bkg_values (np.ndarray): The background values.
        signal_weights (np.ndarray): The signal weights.
        bkg_weights (np.ndarray): The background weights.

    Returns:
        float: The best significance.
        float: The best cut value.
        float: The number of signal events.
        float: The number of background events.
    """
    # Model prediction
    signal_prediction = model.predict(signal_values)
    bkg_prediction = model.predict(bkg_values)

    # Transform predicton to array
    signal_prediction = np.array([element[0] for element in signal_prediction])
    bkg_prediction = np.array([element[0] for element in bkg_prediction])
    
    # Calculate the significance for different cut values in a for loop
    cut_values = []
    significances = []
    n_signal_list = []
    n_bkg_list = []
    for cut_value in np.linspace(0, 1, 100):
        # Number of signal and background events passing the prediction selection
        n_signal = signal_weights[signal_prediction > cut_value].sum()
        n_bkg = bkg_weights[bkg_prediction > cut_value].sum()

        # Break if less than 10 background events
        if n_bkg < 10:
            break

        # Significance calculation
        significance = n_signal / np.sqrt(n_bkg)
        
        # Append the cut value and the significances to their lists
        cut_values.append(cut_value)
        significances.append(significance)
        n_signal_list.append(n_signal)
        n_bkg_list.append(n_bkg)
    
    # Plot the significance as a function of the cut value
    plt.plot(cut_values, significances, label='DNN')

    # Get best significance and best cut value
    best_significance = max(significances)
    best_cut_value = cut_values[significances.index(best_significance)]
    best_n_signal = n_signal_list[significances.index(best_significance)]
    best_n_bkg = n_bkg_list[significances.index(best_significance)]
    return best_significance, best_cut_value, best_n_signal, best_n_bkg


def get_m_llll_significance(signal_data_frame: pd.DataFrame, bkg_data_frame: pd.DataFrame) -> (float, float, float, float):
    """
    Calculate the significance for different cut values in a for loop.

    Parameters:
        signal_data_frame (pd.DataFrame): The signal data frame.
        bkg_data_frame (pd.DataFrame): The background data frame.

    Returns:
        float: The best significance.
        float: The best cut value.
        float: The number of signal events.
        float: The number of background events.
    """
    # Calculate the significance for different cut values in a for loop
    cut_values = []
    significances = []
    n_signal_list = []
    n_bkg_list = []
    for cut_value in np.linspace(10, 0, 100):
        peak_signal = signal_data_frame[abs(signal_data_frame.lep_m_llll - 125) < cut_value]
        peak_bkg = bkg_data_frame[abs(bkg_data_frame.lep_m_llll - 125) < cut_value]

        # Get the sum of the weights
        n_signal = peak_signal.totalWeight.sum()
        n_bkg = peak_bkg.totalWeight.sum()

        # Break if less than 10 background events
        if n_bkg < 10:
            break

        # Significance calculation
        significance = n_signal / np.sqrt(n_bkg)
        
        # Append the cut value and the significances to their lists
        cut_values.append(cut_value)
        significances.append(significance)
        n_signal_list.append(n_signal)
        n_bkg_list.append(n_bkg)
    
    # Plot the significance as a function of the cut value
    plt.plot(cut_values, significances, label='m_llll')
    
    # Get best significance and best cut value
    best_significance = max(significances)
    best_cut_value = cut_values[significances.index(best_significance)]
    best_n_signal = n_signal_list[significances.index(best_significance)]
    best_n_bkg = n_bkg_list[significances.index(best_significance)]
    return best_significance, best_cut_value, best_n_signal, best_n_bkg



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

# Merge the signal and background data frames
data_frame_signal = common.merge_data_frames(sample_list_signal, test_data_frames)
data_frame_background = common.merge_data_frames(sample_list_background, test_data_frames)

# Extract the values, weights, and classification of the test dataset
test_values, test_weights, test_classification = common.get_dnn_input(test_data_frames, training_variables, sample_list_signal, sample_list_background)

# Split the data in signal and background
test_signal_values = test_values[test_classification > 0.5]
test_signal_weights = test_weights[test_classification > 0.5]
test_bkg_values = test_values[test_classification < 0.5]
test_bkg_weights = test_weights[test_classification < 0.5]

best_significance, best_cut_value, best_n_signal, best_n_bkg = get_m_llll_significance(data_frame_signal, data_frame_background)
print(f'cut_value: {best_cut_value:.2f}')
print(f'significance: {best_significance:.2f}')
print(f'n_signal: {best_n_signal:.2f}')
print(f'n_bkg:    {best_n_bkg:.2f}')

# Get the model
model = tf.keras.models.load_model(f'models/medium_dnn_3/early_stopping_model')

best_significance, best_cut_value, best_n_signal, best_n_bkg = get_model_significances(model, test_signal_values, test_bkg_values, test_signal_weights, test_bkg_weights)
print(f'cut_value: {best_cut_value:.2f}')
print(f'significance: {best_significance:.2f}')
print(f'n_signal: {best_n_signal:.2f}')
print(f'n_bkg:    {best_n_bkg:.2f}')
plt.legend()
plt.show()