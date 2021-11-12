from matplotlib import pyplot as plt
import numpy as np

def plot_graph(arr, name):
    plt.plot(range(1, 21), arr, marker='o')
    plt.title(name + ' vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.show()

# Example where: recons_err_arr is an array containing the reconstruction error for various values of k

# plot_graph(recons_err_arr, 'Reconstruction Error')
