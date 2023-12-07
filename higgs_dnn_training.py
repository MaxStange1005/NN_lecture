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
from typing import Optional

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

# Shuffle the training data
shuffling_index = np.arange(len(train_values))
np.random.shuffle(shuffling_index)
train_values = train_values[shuffling_index]
train_classification = train_classification[shuffling_index]
train_weights = train_weights[shuffling_index]

# Get reweighted weights
train_weights_reweighted = common.reweight_weights(train_weights, train_classification)
val_weights_reweighted = common.reweight_weights(val_weights, val_classification)

# Convert the data to tensorflow datasets
train_data = Dataset.from_tensor_slices((train_values, train_classification, train_weights_reweighted))
# Initially shuffle the data
train_data
train_data = train_data.batch(128)
train_data = train_data.shuffle(100, seed=random_state)

val_data = Dataset.from_tensor_slices((val_values, val_classification, val_weights_reweighted))
val_data = val_data.shuffle(10, seed=random_state, reshuffle_each_iteration=False)
val_data = val_data.batch(128)


# Loss function
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Optimizer
adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005, beta_1=0.9)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


def train_model(name: str, layer_nodes: list, epochs: int, patience: Optional[int] = None) -> (tf.keras.Model, list, list):
    """
    Train a model with the given name and layer nodes.

    Parameters:
        name (str): The name of the model.
        layer_nodes (list): The number of nodes in each layer.
        epochs (int): The number of epochs to train the model.
        patience (Optional[int]): The number of epochs to wait before early stopping.

    Returns:
        tf.keras.Model: The trained model.
        list: The validation loss.
        list: The training loss.
    """
    # Normalization layer
    normalization_layer = tf.keras.layers.Normalization()
    normalization_layer.adapt(train_values)

    # Create the model
    model_layers = [normalization_layer]
    for nodes in layer_nodes:
        model_layers.append(tf.keras.layers.Dense(nodes, activation='relu'))
    model_layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    model = tf.keras.Sequential(model_layers)

    # Callback to get the validation and training loss
    validation_loss = []
    training_loss = []
    class LossHistory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Get the validation loss
            validation_loss.append(logs['val_loss'])
            # Evaluate the model on the validation and training data and append the loss to the list
            training_loss.append(self.model.evaluate(train_data, verbose=0)[0])

    # Compile the model
    model.compile(optimizer=adam_optimizer, loss=loss_fn, weighted_metrics=['accuracy'])

    # Train the model
    if patience is not None:
        model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[LossHistory(), early_stopping])
    else:
        model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[LossHistory()])

    # Create subdirectory for model
    model_path = f'models/{name}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    if patience is not None:
        # Save the model
        model.save(os.path.join(model_path, 'early_stopping_model'))

        # Save the training and validation loss
        np.save(os.path.join(model_path, 'early_stopping_val_loss'), validation_loss)
        np.save(os.path.join(model_path, 'early_stopping_train_loss'), training_loss)
    else:
        # Save the model
        model.save(os.path.join(model_path, 'model'))

        # Save the training and validation loss
        np.save(os.path.join(model_path, 'val_loss'), validation_loss)
        np.save(os.path.join(model_path, 'train_loss'), training_loss)

    return model, validation_loss, training_loss


dnn_shapes = {
    'tiny_dnn': [5, 5],
    'small_dnn': [60, 60],
    'medium_dnn': [200, 200],
    'small_dnn_3': [60, 60, 60],
    'medium_dnn_3': [200, 200, 200],
    'small_dnn_4': [60, 60, 60, 60],
    'medium_dnn_4': [200, 200, 200, 200],
    'costum_dnn_3': [50, 200, 50],
    'costum_dnn_4': [50, 200, 200, 50],
    'costum_dnn_5': [50, 200, 300, 200, 50],
    'huge_dnn': [1000, 1000]
}

for name, shape in dnn_shapes.items():
    train_model(name, shape, 200)
    train_model(name, shape, 200, patience=10)