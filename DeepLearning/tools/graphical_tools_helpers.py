import numpy as np


def _show_values_on_bars(axs, data, pct=False):
    """Helper function to show values on barplots

    Adapted from: https://stackoverflow.com/a/51535326

    Args:
        axs (matplotlib axes): Plot axes
        data (array): Data to place atop each bar
        pct (bool, optional): Whether to display % symbol. Defaults to False.
    """
    def _show_on_single_plot(ax):
        for i, p in enumerate(ax.patches):
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            if pct:
                value = str(data[i]) + "%"
            else:
                value = data[i]
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for _, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def _display_plot_values_labels(ax, title, x_lbl, y_lbl, values_on_bars=None,
                                pct=False, x_tick_labels=None):
    """Displays x and y labels, title, and optional values on axes

    Args:
        ax (matplotlib axes): Axes to modify
        title (str): Title text
        x_lbl (str): X-label text
        y_lbl (str): Y-label text
        values_on_bars (array, optional): Values to display atop bars
            in barplot. Defaults to None.
        pct (bool, optional): Whether values are percents. Defaults to False.
        x_tick_labels (array, optional): X-tick-labels to update.
            Defaults to None.

    Returns:
        matplotlib axes: Modified axes
    """
    if values_on_bars is not None:
        _show_values_on_bars(ax, values_on_bars, pct=pct)

    ax.set_title(title)
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)

    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)

    return ax
