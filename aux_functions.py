from inspect import currentframe

import numpy as np
from matplotlib import pyplot as plt


def getLineNumber():
    return currentframe().f_back.f_lineno


def infoMessage(line, string):
    print(f'[line {line}]: {string}')

def infoMessage0(string):
    print(f'[-----]: {string}')


def graph_create_prev_version(num_steps, loss, secret_loss, D_loss, labels_created):
    loss = np.array(loss)
    secret_loss = np.array(secret_loss)
    D_loss = np.array(D_loss)
    infoMessage(getLineNumber(), f'Displaying loss functions after {num_steps} iterations')
    epochs = range(0, num_steps)
    if labels_created is False:
        plt.plot(epochs, loss, 'g', label='loss')
        plt.plot(epochs, secret_loss, 'r', label='secret loss')
        plt.plot(epochs, D_loss, 'm', label='D loss')
    else:
        plt.plot(epochs, loss, 'g')
        plt.plot(epochs, secret_loss, 'r')
        plt.plot(epochs, D_loss, 'm')
    plt.title('Total loss results')
    plt.xlabel(f'Epochs = {epochs}')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_graphs/0908_2nd_version/loss_graph_{num_steps}_iterations')


def graph_create(num_steps, loss, secret_loss, D_loss, labels_created):
    loss = np.array(loss)
    secret_loss = np.array(secret_loss)
    D_loss = np.array(D_loss)
    infoMessage(getLineNumber(), f'Displaying loss functions after {num_steps} iterations')
    create_graph_total_loss(num_steps, loss, secret_loss, D_loss, labels_created)
    create_graph_secret_loss(num_steps, secret_loss, D_loss, labels_created)



def create_graph_total_loss(num_steps, loss, secret_loss, D_loss, labels_created):
    epochs = range(0, num_steps)
    total_plt = plt.figure()
    plt.plot(epochs, loss, 'g', label='loss')
    plt.plot(epochs, secret_loss, 'r', label='secret loss')
    plt.plot(epochs, D_loss, 'm', label='D loss')
    plt.title('Total loss results')
    plt.xlabel(f'Epochs = {epochs}')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_graphs/1208_version/graph_iterations_total_loss_{num_steps}')


def create_graph_secret_loss(num_steps, secret_loss, D_loss, labels_created):
    epochs = range(0, num_steps)
    secret_plt = plt.figure()
    plt.plot(epochs, secret_loss, 'r', label='secret loss')
    plt.plot(epochs, D_loss, 'm', label='D loss')
    plt.title('Total loss results')
    plt.xlabel(f'Epochs = {epochs}')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_graphs/1208_version/graph_iterations_secret_loss_{num_steps}')


