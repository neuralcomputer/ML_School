import matplotlib.pyplot as plt
import numpy as np

def plot_line_or_scatter(type_plot, x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):
    # Create the plot object
    _, ax = plt.subplots()
    
    if type_plot == 'scatter':
        ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)
    
    else:
        ax.plot(x_data, y_data, lw = 2, color = color, alpha = 1)
        
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)   