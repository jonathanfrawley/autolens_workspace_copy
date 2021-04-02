"""
Plots: ArrayOverlay
===================

This example illustrates how to overlay a 2D `Array2D` over PyAutoLens figures and subplots.
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
We next need the 2D `Array2D` we overlay. We'll create a simple 3x3 array.
"""
arr = al.Array2D.manual_native(
    array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=0.5
)

"""
We input this `Array2D` into the `Visuals2D` object, which plots it over the figure.
"""
visuals_2d = aplt.Visuals2D(array_overlay=arr)

"""
We now plot the image with the array overlaid.
"""
array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)
array_plotter.figure_2d()

"""
We customize the overlaid array using the `ArrayOverlay` matplotlib wrapper object which wraps the following method(s):

To overlay the array this objects wrap the following matplotlib method:

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
"""
array_overlay = aplt.ArrayOverlay(alpha=0.5)

mat_plot_2d = aplt.MatPlot2D(array_overlay=array_overlay)

array_plotter = aplt.Array2DPlotter(
    array=image, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
)
array_plotter.figure_2d()

"""
Finish.
"""
