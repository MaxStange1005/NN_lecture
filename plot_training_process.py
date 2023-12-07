import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import tensorflow as tf

def plot_training_process(name: str, plot_val: bool = False):
    """
    Plot the training process of a model.

    Parameters:
        name (str): The name of the model.
        plot_val (bool): Whether to plot the validation loss.
    """
    # Read the training history
    train_loss = np.load(f'models/{name}/train_loss.npy')[:100]
    if plot_val:
        val_loss = np.load(f'models/{name}/val_loss.npy')[:100]

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlim(1, len(train_loss))
    ax.set_ylim(0, 0.7)

    # Plot the training loss
    train_loss_line, = ax.plot([], [], label='Training loss', c='blue')
    if plot_val:
        val_loss_line, = ax.plot([], [], label='Validation loss', c='darkorange')

    # Plot the legend
    ax.legend(loc='upper right')

    best_train_loss = train_loss[0]
    ax.text(5, 0.65, 'Train loss:', ha='left', va='center', fontsize=14)
    train_loss_text = ax.text(24, 0.65, '', ha='left', va='center', fontsize=14)
    if plot_val:
        best_val_loss = val_loss[0]
        ax.text(5, 0.6, 'Val loss:', ha='left', va='center', fontsize=14)
        val_loss_text = ax.text(24, 0.6, '', ha='left', va='center', fontsize=14)

    # Function to update the plot
    def update(epoch: int):
        nonlocal best_train_loss, best_val_loss
        epoch += 1
        train_loss_line.set_data(np.arange(1, epoch + 1), train_loss[:epoch])
        if train_loss[epoch - 1] < best_train_loss:
            best_train_loss = train_loss[epoch - 1]
        train_loss_text.set_text(f'current={train_loss[epoch - 1]:.4f}, best={best_train_loss:.4f}')
        if plot_val:
            val_loss_line.set_data(np.arange(1, epoch + 1), val_loss[:epoch])
            if val_loss[epoch - 1] < best_val_loss:
                best_val_loss = val_loss[epoch - 1]
            val_loss_text.set_text(f'current={val_loss[epoch - 1]:.4f}, best={best_val_loss:.4f}')
        return
    
    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(train_loss), interval=1)

    # Save the animation
    if plot_val:
        animation.save(f'model_training/{name}_with_val.gif', writer=PillowWriter(fps=5))
    else:
        animation.save(f'model_training/{name}.gif', writer=PillowWriter(fps=5))


def plot_best_epoch(name: str):
    """
    Plot the training process of a model and mark the best epoch.

    Parameters:
        name (str): The name of the model.
    """
    # Read the training history
    train_loss = np.load(f'models/{name}/train_loss.npy')[:100]
    val_loss = np.load(f'models/{name}/val_loss.npy')[:100]

    # Best loss
    best_train_loss = np.min(train_loss)
    best_val_loss = np.min(val_loss)
    best_val_epoch = np.argmin(val_loss) + 1

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlim(0, len(train_loss))
    ax.set_ylim(0, 0.7)

    # Plot the  training and validation loss
    ax.plot(np.arange(1, len(train_loss) + 1), train_loss, label='Training loss')
    ax.plot(np.arange(1, len(val_loss) + 1), val_loss, label='Validation loss')

    # Text for best loss
    ax.text(5, 0.65, 'Train loss:', ha='left', va='center', fontsize=14)
    ax.text(24, 0.65, f'best = {best_train_loss:.4f}', ha='left', va='center', fontsize=14)
    ax.text(5, 0.6, 'Val loss:', ha='left', va='center', fontsize=14)
    ax.text(24, 0.6, f'best = {best_val_loss:.4f}', ha='left', va='center', fontsize=14)

    # Plot vline for best epoch
    ax.axvline(best_val_epoch, color='red', linestyle='--', label='Best epoch')

    # Plot the legend
    ax.legend(loc='upper right')

    # Save the figure
    fig.savefig(f'model_training/{name}_best_epoch.png', dpi=300)


def plot_early_stopping(name: str, patience: int):
    """
    Animate the training process of a model with early stopping.

    Parameters:
        name (str): The name of the model.
        patience (int): The number of epochs to wait before early stopping.
    """
    # Read the training history
    train_loss = np.load(f'models/{name}/train_loss.npy')[:100]
    val_loss = np.load(f'models/{name}/val_loss.npy')[:100]

    best_epoch = 0
    best_val_loss = val_loss[0]
    best_train_loss = train_loss[0]

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlim(1, len(train_loss))
    ax.set_ylim(0, 0.7)

    # Plot the training loss
    train_loss_line, = ax.plot([], [], label='Training loss', c='blue')
    train_stopped_line, = ax.plot([], [], alpha=0.2, c='blue')
    val_loss_line, = ax.plot([], [], label='Validation loss', c='darkorange')
    val_stopped_line, = ax.plot([], [], alpha=0.2, c='darkorange')
    best_epoch_line = ax.axvline(best_epoch, color='red', linestyle='--', label='Best epoch')
    stop_text = ax.text(1, 0.5, '', ha='center', va='center', fontsize=14)

    # Text for best loss
    ax.text(5, 0.65, 'Train loss:', ha='left', va='center', fontsize=14)
    train_loss_text = ax.text(24, 0.65, '', ha='left', va='center', fontsize=14)
    ax.text(5, 0.6, 'Val loss:', ha='left', va='center', fontsize=14)
    val_loss_text = ax.text(24, 0.6, '', ha='left', va='center', fontsize=14)

    # Plot the legend
    ax.legend(loc='upper right')

    stopped = False
    stopped_epoch = 0
    stopped_val_loss = 0
    # Function to update the plot
    def update(epoch: int):
        nonlocal stopped, stopped_epoch, stopped_val_loss, best_epoch, best_val_loss, best_train_loss
        epoch += 1
        if not stopped:
            train_loss_line.set_data(np.arange(1, epoch + 1), train_loss[:epoch])
            val_loss_line.set_data(np.arange(1, epoch + 1), val_loss[:epoch])

            # Get the best loss
            current_train_loss = train_loss[epoch - 1]
            current_val_loss = val_loss[epoch - 1]
            if current_train_loss < best_train_loss:
                best_train_loss = current_train_loss
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
            best_epoch_line.set_xdata(best_epoch)
            if epoch - best_epoch > patience:
                stopped = True
                stopped_epoch = epoch
                stopped_val_loss = current_val_loss
            
            # Update text
            train_loss_text.set_text(f'current={current_train_loss:.4f}, best={best_train_loss:.4f}')
            val_loss_text.set_text(f'current={current_val_loss:.4f}, best={best_val_loss:.4f}')
            return
        
        train_loss_line.set_data(np.arange(1, stopped_epoch + 1), train_loss[:stopped_epoch])
        train_stopped_line.set_data(np.arange(stopped_epoch, epoch + 1), train_loss[stopped_epoch - 1:epoch])
        val_loss_line.set_data(np.arange(1, stopped_epoch + 1), val_loss[:stopped_epoch])
        val_stopped_line.set_data(np.arange(stopped_epoch, epoch + 1), val_loss[stopped_epoch - 1:epoch])
        stop_text.set_text(f'Stop')
        stop_text.set_position((stopped_epoch, stopped_val_loss + 0.02))
        return
    
    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(train_loss), interval=1)

    # Save the animation
    animation.save(f'model_training/{name}_early_stopping.gif', writer=PillowWriter(fps=5))


def plot_ddn_shape(name: str):
    """
    Plot the shape of a DDN.

    Parameters:
        name (str): The name of the model.
    """
    # Load DDN
    model = tf.keras.models.load_model(f'models/{name}/model')

    # Plot model
    tf.keras.utils.plot_model(model, to_file=f'model_training/{name}_model.png', show_dtype=False, show_shapes=True, show_layer_names=False, show_layer_activations=True, dpi=300)

    # Print model summary
    model.summary()


dnns = ['tiny_dnn', 'small_dnn', 'medium_dnn', 'small_dnn_3', 'medium_dnn_3', 'small_dnn_4', 'medium_dnn_4', 'costum_dnn_3', 'costum_dnn_4', 'costum_dnn_5', 'huge_dnn']
plot_early_stopping('medium_dnn_3', 10)

for dnn in dnns:
    print(dnn)
    plot_ddn_shape(dnn)
    #continue
    plot_training_process(dnn, plot_val=True)
    plot_training_process(dnn, plot_val=False)
    plot_best_epoch(dnn)
    plot_early_stopping(dnn, 10)