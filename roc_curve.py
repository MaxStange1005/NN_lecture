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


def weighted_tpr_fpr_at_threshold(score_threshold: float, y_pred: np.ndarray, y_true: np.ndarray, sample_weights: np.ndarray) -> (float, float):
    """
    Calculate the weighted true positive rate and false positive rate at a given threshold.

    Parameters:
        score_threshold (float): The threshold.
        y_pred (np.ndarray): The predictions.
        y_true (np.ndarray): The true labels.
        sample_weights (np.ndarray): The sample weights.

    Returns:
        float: The weighted true positive rate.
        float: The weighted false positive rate.
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    sample_weights = sample_weights.flatten()
    signal_weights = sample_weights[y_true == 1]
    background_weights = sample_weights[y_true == 0]
    signal_pred = y_pred[y_true == 1]
    background_pred = y_pred[y_true == 0]
    tpr = np.sum(signal_weights[signal_pred >= score_threshold]) / np.sum(signal_weights)
    fpr = np.sum(background_weights[background_pred >= score_threshold]) / np.sum(background_weights)
    return tpr, fpr


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, y_weights: np.ndarray):
    """
    Plot the ROC curve.

    Parameters:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predictions.
        y_weights (np.ndarray): The sample weights.
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_true.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i], sample_weight=y_weights)
        sort_index = np.argsort(fpr[i])
        fpr[i] = fpr[i][sort_index]
        tpr[i] = tpr[i][sort_index]
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(7, 5))
    for i in range(y_true.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'ROC (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.5, label='random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=14)


def plot_roc_curve_with_histogram(fig: plt.Figure, y_true: np.ndarray, y_pred: np.ndarray, y_weights: np.ndarray, score_threshold: float = 0.5, show_curve: bool = True):
    """
    Plot the ROC curve with a histogram of the NN output scores.

    Parameters:
        fig (plt.Figure): The figure to plot on.
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predictions.
        y_weights (np.ndarray): The sample weights.
        score_threshold (float): The threshold.
        show_curve (bool): Whether to show the ROC curve.
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel(), sample_weight=y_weights)
    sort_index = np.argsort(fpr)
    fpr = fpr[sort_index]
    tpr = tpr[sort_index]
    roc_auc = auc(fpr, tpr)

    # Top subplot: Histogram with vline at score_threshold
    signal_pred = y_pred[y_true == 1].flatten()
    signal_weights = y_weights[(y_true == 1).flatten()]
    background_pred = y_pred[y_true == 0].flatten()
    background_weights = y_weights[(y_true == 0).flatten()]
    plt.subplot(2, 1, 1)
    plt.hist(signal_pred, bins=50, range=(0, 1), weights=signal_weights / signal_weights.sum(), label='Higgs', color='darkorange', histtype='step')
    plt.hist(background_pred, bins=50, range=(0, 1), weights=background_weights / background_weights.sum(), label='Background', color='blue', histtype='step')
    plt.axvline(x=score_threshold, color='red', linestyle='--', label=f'Threshold')
    plt.xlabel('NN Output Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Histogram of NN Output Scores', fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.xlim(0, 1)

    # Bottom subplot: ROC curve
    plt.subplot(2, 1, 2)
    current_tpr, current_fpr = weighted_tpr_fpr_at_threshold(score_threshold, y_pred, y_true, y_weights)
    if show_curve:
        plt.plot(fpr[fpr >= current_fpr], tpr[fpr >= current_fpr], label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random', alpha=0.5)
    else:
        print(f'Treshold: {score_threshold:.2f} fpr = {current_fpr:.3f} tpr = {current_tpr:.3f}')
    plt.scatter([current_fpr], [current_tpr], c='red', marker='o', label=f'Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=14)

    plt.tight_layout()


def animated_roc_curve_with_histogram(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_weights: np.ndarray):
    """
    Animate the ROC curve with a histogram of the NN output scores.

    Parameters:
        name (str): The name of the model.
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predictions.
        y_weights (np.ndarray): The sample weights.
    """ 
    # Plot ROC curve with highlighted point at the specified threshold
    fig = plt.figure(figsize=(7, 10))
    score_threshold = 0.0
    # Plot roc curve
    plot_roc_curve_with_histogram(fig, y_true, y_pred, y_weights, score_threshold)

    n_frames = 300

    # Create animation
    def update(frame):
        nonlocal score_threshold
        score_threshold = frame / (n_frames - 1)
        plt.clf()
        plot_roc_curve_with_histogram(fig, y_true, y_pred, y_weights, score_threshold)

    animation = FuncAnimation(fig, update, frames=n_frames, interval=1000, repeat=False)

    # Save animation
    animation.save(f'roc_curve/{name}_roc_curve_hist.gif', writer=PillowWriter(fps=30))



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

# Extract the values, weights, and classification of the data
values, weights, classification = common.get_dnn_input(train_data_frames, training_variables, sample_list_signal, sample_list_background)

# Split into train and validation data
train_values, val_values, train_classification, val_classification = train_test_split(values, classification, test_size=1/3, random_state=random_state)
train_weights, val_weights = train_test_split(weights, classification, test_size=1/3, random_state=random_state)[:2]

# Get reweighted weights
train_weights_reweighted = common.reweight_weights(train_weights, train_classification)
val_weights_reweighted = common.reweight_weights(val_weights, val_classification)

models = ['tiny_dnn', 'small_dnn', 'medium_dnn', 'medium_dnn_3', 'costum_dnn_5', 'huge_dnn']

for model_name in models:
    print(model_name)
    # Load the model
    model = tf.keras.models.load_model(f'models/{model_name}/early_stopping_model')

    # Make predictions on the validation data
    y_pred = model.predict(val_values)

    # Binarize the true labels
    y_true = label_binarize(val_classification, classes=np.unique(val_classification))

    plot_roc_curve(y_true, y_pred, val_weights_reweighted)
    plt.savefig(f'roc_curve/{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')

    animated_roc_curve_with_histogram(model_name, y_true, y_pred, val_weights_reweighted)
    plt.clf()
    fig = plt.figure(figsize=(7, 10))
    plot_roc_curve_with_histogram(fig, y_true, y_pred, val_weights_reweighted, score_threshold=0.1, show_curve=False)
    plt.savefig(f'roc_curve/{model_name}_roc_curve_hist_0_1.png', dpi=300)
    plt.clf()
    plot_roc_curve_with_histogram(fig, y_true, y_pred, val_weights_reweighted, score_threshold=0.3, show_curve=False)
    plt.savefig(f'roc_curve/{model_name}_roc_curve_hist_0_3.png', dpi=300)
    plt.clf()
    plot_roc_curve_with_histogram(fig, y_true, y_pred, val_weights_reweighted, score_threshold=0.5, show_curve=False)
    plt.savefig(f'roc_curve/{model_name}_roc_curve_hist_0_5.png', dpi=300)
    plt.clf()
    plot_roc_curve_with_histogram(fig, y_true, y_pred, val_weights_reweighted, score_threshold=0.7, show_curve=False)
    plt.savefig(f'roc_curve/{model_name}_roc_curve_hist_0_7.png', dpi=300)
    plt.clf()
    plot_roc_curve_with_histogram(fig, y_true, y_pred, val_weights_reweighted, score_threshold=0.9, show_curve=False)
    plt.savefig(f'roc_curve/{model_name}_roc_curve_hist_0_9.png', dpi=300)
    plt.clf()
