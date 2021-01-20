import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_history(history, model_name):
    df = pd.DataFrame(history.history)
    df.plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(model_name + ' Model : Training History')
    plt.xlabel('Epochs')
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.show()
