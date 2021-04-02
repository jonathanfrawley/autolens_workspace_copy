"""
Plots: Labels
=============

This example illustrates how to customize the Labels of a figure or subplot displayed in PyAutoLens.
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

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.title.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.ylabel.html
 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xlabel.html

We can manually specify the label of the title, ylabel and xlabel.
"""
title = aplt.Title(label="This is the title", color="r", fontsize=20)

ylabel = aplt.YLabel(label="Label of Y", color="b", fontsize=5, position=(0.2, 0.5))

xlabel = aplt.XLabel(label="Label of X", color="g", fontsize=10)

mat_plot_2d = aplt.MatPlot2D(title=title, ylabel=ylabel, xlabel=xlabel)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

"""
If we do not manually specify a label, the name of the function used to plot the image will be used as the title 
and the units of the image will be used for the ylabel and xlabel.
"""
title = aplt.Title()
ylabel = aplt.YLabel()
xlabel = aplt.XLabel()

mat_plot_2d = aplt.MatPlot2D(title=title, ylabel=ylabel, xlabel=xlabel)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

"""
The units can be manually specified.
"""
mat_plot_2d = aplt.MatPlot2D(units=aplt.Units(in_kpc=True, conversion_factor=5.0))

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

"""
Finish.
"""
