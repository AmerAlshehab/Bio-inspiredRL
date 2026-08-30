"""Shared figure styling: bold, larger text and heavier lines so the report
figures stay legible when scaled down to column width. Call apply_bold_style()
inside a plotting function, before the figure is created."""


def apply_bold_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.weight": "bold",
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 13, "axes.labelweight": "bold",
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "legend.title_fontsize": 12,
        "figure.titlesize": 15, "figure.titleweight": "bold",
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })
