
"""

Various plotting routines inspired by the Wolfram Language.

"""

import matplotlib.pyplot as plt
import numpy as np
from copy import copy


class Color:

    def __init__(self, spec=None):
        if spec is None:
            self.red = 0.790588
            self.green = 0.201176
            self.blue = 0.
            self.alpha = 1.0
        else:
            if len(spec) == 3:
                self.red, self.green, self.blue = spec
                self.alpha = 1.0
            else:
                self.red, self.green, self.blue, self.alpha = spec

    @property
    def rgb(self):
        return tuple([self.red, self.green, self.blue])

    @property
    def rgba(self):
        return tuple([self.red, self.green, self.blue, self.alpha])

    def __str__(self):
        if self.alpha < 1:
            return "RGBA(" + str(self.red) + ", " + str(self.green) + ", " \
                   + str(self.blue) + ", " + str(self.alpha) + ")"
        else:
            return "RGB(" + str(self.red) + ", " + str(self.green) + ", " + str(self.blue) + ")"

    @staticmethod
    def color_data(name="112", n=0):
        """
        Yields colors from named color schemes.
        :param name: Name of the color scheme.
        :param n: Index of the requested color in that scheme.
        :return: Color object.
        """
        return {
            "112": {
                0: Color((0.790588, 0.201176, 0.)),
                1: Color((0.192157, 0.388235, 0.807843)),
                2: Color((1., 0.607843, 0.)),
                3: Color((0., 0.596078, 0.109804)),
                4: Color((0.567426, 0.32317, 0.729831)),
                5: Color((0., 0.588235, 0.705882))
            }.get(n % 5)
        }.get(name)


def _option_value(value, index=None, tag=None, default=None):
    """
    Helper function for options that can be single values or lists of values.
    :param value: Single value or list of values.
    :param index: Index of the requested element, if values is a list.
    :param default: Value to be returned if `value` is a list and `index` is `None`, or if `value` is `None`
    :return: Value of the option.
    """
    if type(value) is list:
        if index is None:
            return default
        else:
            result = value[index % len(value)]
            if result is None:
                return default
            elif type(result) is dict:
                return result.get(tag, default)
            else:
                if type(result) is type(default):
                    return result
                else:
                    return default
    else:
        if value is None:
            return default
        elif type(value) is dict:
            return value.get(tag, default)
        else:
            if type(value) is type(default):
                return value
            else:
                return default


# Default area of a point in a 'scatter plot'
_default_point_size = 20


def list_plot(data, step=(), plot_range=(), axes_label=(), plot_style=None, joined=None, mesh=None, filling=None):
    """
    list_plot([y_1, y_2, ...]) plots the points (1, y_1), (2, y_2), ...
    list_plot([[x_1, y_1], [x_2, y_2], ...]) plots the points (x_1, y_1), (x_2, y_2), ...
    :param data: Points or ordinates to be plotted.
    :param step: If only ordinates are given, then 'step' controls the range of abscissas (optional).
    :param plot_range: Range of coordinates that is to be displayed (optional).
    :param axes_label: Labels of the two axes (optional).
    :param plot_style: Specify the style (color, point size, etc.) of the points / lines.
    :param joined: Bool, or list of booleans that indicate if points should be joined by a line.
    :param mesh: If False, only the joining-lines are plotted (if present).
    :param filling: Specifies filled areas. E.g. `filling=[[0,2]]` causes the area between curves 0 and 2 to be filled.
    """
    if len(axes_label) == 0:
        x_label = ""
        y_label = ""
    elif len(axes_label) == 2:
        x_label, y_label = axes_label
    else:
        raise ValueError("The 'axes_label' option must be () or (x_label, y_label).")

    if type(data[0]) is float or type(data[0]) is int:
        # `data` is a list of y-values
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

        # Pick settings from plot_style
        color = _option_value(plot_style, tag="Color", default=Color.color_data())
        marker = _option_value(plot_style, tag="Marker", default=".")
        point_size = _option_value(plot_style, tag="PointSize", default=_default_point_size)

        if _option_value(mesh, default=True):
            plt.scatter(x, y, color=color.rgba, s=point_size, marker=marker)
        if _option_value(joined, default=False):
            plt.plot(x, y, color=color.rgba)

    elif len(np.shape(data)) == 2:
        # `data` is a list of [x, y]-tuples
        x, y = list(np.transpose(data))

        x0 = np.min(x)
        x1 = np.max(x)
        y0 = np.min(y)
        y1 = np.max(y)

        # Pick settings from plot_style
        color = _option_value(plot_style, tag="Color", default=Color.color_data())
        marker = _option_value(plot_style, tag="Marker", default=".")
        point_size = _option_value(plot_style, tag="PointSize", default=_default_point_size)

        if _option_value(mesh, default=True):
            plt.scatter(x, y, color=color.rgba, s=point_size, marker=marker)
        if _option_value(joined, default=False):
            plt.plot(x, y, color=color.rgba)

    elif len(np.shape(data[0])) == 2:
        # `data` is a list of lists of [x. y]-tuples (multiple graphs)

        # Draw fillings first, so they are in the background
        if filling is not None:
            if type(filling) is list:
                for i, j in filling:
                    # Pick color from plot_style
                    color = _option_value(plot_style, i, tag="Color", default=Color.color_data(n=i))
                    color = copy(color)  # Create a copy of that color, so we can modify it
                    color.alpha = 0.5    # Make the color transparent (for filling)

                    x, ya = list(np.transpose(data[i]))
                    _, yb = list(np.transpose(data[j]))
                    plt.fill_between(x, ya, yb, facecolor=color.rgba)

        # Plot the data and determine the plot range
        x0, x1, y0, y1 = None, None, None, None
        for i in range(len(data)):
            x, y = list(np.transpose(data[i]))  # Use first set to determine bounds

            if x0 is None:
                x0 = np.min(x)
                x1 = np.max(x)
                y0 = np.min(y)
                y1 = np.max(y)
            else:
                x0 = min([x0, np.min(x)])
                x1 = max([x1, np.max(x)])
                y0 = min([y0, np.min(y)])
                y1 = max([y1, np.max(y)])

            # Pick settings from plot_style
            color = _option_value(plot_style, i, tag="Color", default=Color.color_data(n=i))
            marker = _option_value(plot_style, i, tag="Marker", default=".")
            point_size = _option_value(plot_style, i, tag="PointSize", default=_default_point_size)

            if _option_value(mesh, i, default=True):
                plt.scatter(x, y, color=color.rgba, s=point_size, marker=marker)

            if _option_value(joined, i, default=False):
                plt.plot(x, y, color=color.rgba)
    else:
        raise ValueError("Data must be either a list of y-values, a list of (x,y)-tuples, or a list of lists of "
                         "(x,y)-tuples.")

    margin_x = 0.05 * (x1 - x0)
    margin_y = 0.05 * (y1 - y0)
    x0, x1, y0, y1 = (x0 - margin_x, x1 + margin_x, y0 - margin_y, y1 + margin_y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis([x0, x1, y0, y1])
    plt.show()
