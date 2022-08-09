from inspect import currentframe

import numpy as np
from matplotlib import pyplot as plt


def getLineNumber():
    return currentframe().f_back.f_lineno


def infoMessage(line, string):
    print(f'[line {line}]: {string}')


# def graph_create(num_steps, l2_loss_array, lpips_loss_array, secret_loss_array, G_loss_array):
#     infoMessage(getLineNumber(), f'Displaying loss functions after {num_steps} iterations')
#     epochs = range(0, num_steps)
#     plt.plot(epochs, l2_loss_array, 'g', label='l2 loss')
#     plt.plot(epochs, lpips_loss_array, 'b', label='lpips loss')
#     plt.plot(epochs, secret_loss_array, 'r', label='secret loss')
#     plt.plot(epochs, G_loss_array, 'm', label='G loss')
#     plt.title('Loss results')
#     plt.xlabel(f'Epochs = {epochs}')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(f'loss_graphs/0608/loss_graph_{num_steps}_iterations')


def graph_create(num_steps, loss, secret_loss, D_loss):
    loss = np.array(loss)
    secret_loss = np.array(secret_loss)
    D_loss = np.array(D_loss)
    infoMessage(getLineNumber(), f'Displaying loss functions after {num_steps} iterations')
    epochs = range(0, num_steps)
    plt.plot(epochs, loss, 'g', label='loss')
    plt.plot(epochs, secret_loss, 'r', label='secret loss')
    plt.plot(epochs, D_loss, 'm', label='D loss')
    plt.title('Loss results')
    plt.xlabel(f'Epochs = {epochs}')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_graphs/0908/loss_graph_{num_steps}_iterations')


