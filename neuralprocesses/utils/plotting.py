
"""

Various plotting routines inspired by the Wolfram Language.

"""

import matplotlib.pyplot as plt
import numpy as np


def list_plot(data, step=(), plot_range=(), axes_label=()):
    """
    list_plot([y_1, y_2, ...]) plots the points (1, y_1), (2, y_2), ...
    list_plot([[x_1, y_1], [x_2, y_2], ...]) plots the points (x_1, y_1), (x_2, y_2), ...
    :param data: Points or ordinates to be plotted.
    :param step: If only ordinates are given, then 'step' controls the range of abscissas (optional).
    :param plot_range: Range of coordinates that is to be displayed (optional).
    :param axes_label: Labels of the two axes (optional).
    """
    if len(axes_label) == 0:
        x_label = ""
        y_label = ""
    elif len(axes_label) == 2:
        x_label, y_label = axes_label
    else:
        raise ValueError("The 'axes_label' option must be () or (x_label, y_label).")

    if len(np.shape(data)) == 1:
        if len(step) == 0:
            x0 = 1
            x1 = len(data)
            dx = 1
        elif len(step) == 2:
            x0, x1 = step
            dx = 1
        elif len(step) == 3:
            x0, x1, dx = step
        else:
            raise ValueError("The 'step' option must be (), (x0, x1), or (x0, x1, dx).")

        if len(plot_range) == 0:
            y0 = np.min(data)
            y1 = np.max(data)
        elif len(plot_range) == 2:
            y0, y1 = plot_range
        else:
            raise ValueError("The 'plot_range' option must be () or (y0, y1).")

        x = np.arange(x0, x1 + 1, dx)
        y = data

    elif len(np.shape(data)) == 2:
        x, y = list(np.transpose(data))

        x0 = np.min(x)
        x1 = np.max(x)
        y0 = np.min(y)
        y1 = np.max(y)

    else:
        raise ValueError("Data must be either a vector or an array.")

    margin_x = 0.05 * (x1 - x0)
    margin_y = 0.05 * (y1 - y0)
    x0, x1, y0, y1 = (x0 - margin_x, x1 + margin_x, y0 - margin_y, y1 + margin_y)

    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis([x0, x1, y0, y1])
    plt.show()
