"""
Plots: Ticks
============

This example illustrates how to customize the Ticks of a figure or subplot displayed in PyAutoLens, by
wrapping the inputs of the Matplotlib methods `plt.tick_params`, `plt.yticks` and `plt.xticks`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
First, lets load an example Hubble Space Telescope image of a real strong lens as an `Array2D`.
"""
dataset_path = path.join("dataset", "slacs", "slacs1430+4105")
image_path = path.join(dataset_path, "image.fits")
image = al.Array2D.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)

"""
We can customize the ticks using the `YTicks` and `XTicks matplotlib wrapper object which wraps the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.tick_params.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.yticks.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xticks.html
"""
tickparams = aplt.TickParams(
    axis="y",
    which="major",
    direction="out",
    color="b",
    labelsize=20,
    labelcolor="r",
    length=2,
    pad=5,
    width=3,
    grid_alpha=0.8,
)

yticks = aplt.YTicks(alpha=0.8, fontsize=10, rotation="vertical")
xticks = aplt.XTicks(alpha=0.5, fontsize=5, rotation="horizontal")

mat_plot_2d = aplt.MatPlot2D(tickparams=tickparams, yticks=yticks, xticks=xticks)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

"""
Finish.
"""
