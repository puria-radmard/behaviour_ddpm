import numpy as np


def symmetrize_and_square_axis(axes, min_size=None):
    x_max = np.max(np.abs(axes.get_xlim()))
    y_max = np.max(np.abs(axes.get_ylim()))
    xy_max = max(x_max, y_max)
    if min_size is not None:
        xy_max = max(xy_max, min_size)
    axes.set_ylim(ymin=-xy_max, ymax=xy_max)
    axes.set_xlim(xmin=-xy_max, xmax=xy_max)
