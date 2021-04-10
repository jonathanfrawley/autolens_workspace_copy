"""
Plots: Array2DPlotter
=====================

This example illustrates how to plot an `Array2D` data structure using an `Array2DPlotter`.
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
First, lets load an example image of of a strong lens as an `Array2D`.
"""
dataset_path = path.join("dataset", "slacs", "slacs1430+4105")
image_path = path.join(dataset_path, "image.fits")
image = al.Array2D.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)

"""
We now pass the array to an `Array2DPlotter` and call the `figure` method.
"""
array_plotter = aplt.Array2DPlotter(array=image)
array_plotter.figure_2d()

"""
An `Array2D` contains the following attributes which can be plotted automatically via the `Include2D` object.

(By default, an `Array2D` does not contain a `Mask2D`, we therefore manually created an `Array2D` with a mask to illustrate
plotting its mask and border below).
"""
mask = al.Mask2D.circular_annular(
    shape_native=image.shape_native,
    pixel_scales=image.pixel_scales,
    inner_radius=0.3,
    outer_radius=3.0,
)
masked_image = al.Array2D.manual_mask(array=image.native, mask=mask)

include_2d = aplt.Include2D(origin=True, mask=True, border=True)

array_plotter = aplt.Array2DPlotter(array=masked_image, include_2d=include_2d)
array_plotter.figure_2d()

"""
Finish.
"""
