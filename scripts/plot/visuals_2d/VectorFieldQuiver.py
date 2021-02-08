"""
Plots: VectorFieldQuiver
========================

This example illustrates how to plot and customize vector fields in PyAutoLens figures and subplots.
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
We need a `VectorField` to plot over the image. We make a simple example of a vector field below.
"""
vector_field = al.VectorField2DIrregular(
    vectors=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
)

"""
To plot the vector field manually, we can pass it into a` Visuals2D` object.
"""
visuals_2d = aplt.Visuals2D(vector_field=vector_field)

array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)
array_plotter.figure()

"""
We can customize the appearance of the vectors using the `VectorFieldQuiver matplotlib wrapper object which wraps 
the following method(s):

 https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html
"""
quiver = aplt.VectorFieldQuiver(
    headlength=1,
    pivot="tail",
    color="w",
    linewidth=10,
    units="width",
    angles="uv",
    scale=None,
    width=0.5,
    headwidth=3,
    alpha=0.5,
)

mat_plot_2d = aplt.MatPlot2D(vector_field_quiver=quiver)

array_plotter = aplt.Array2DPlotter(
    array=image, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
)
array_plotter.figure()

"""
Finish.
"""
