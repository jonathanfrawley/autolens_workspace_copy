"""
Plots: EinsteinRadiusAXVLine
=============================

This example illustrates how to plot the einstein radius of a `MassProfile` on 1D figures of its properties.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
First, lets create a simple `MassProfile` which we'll plot.
"""
mass = al.mp.EllipticalIsothermal(
    centre=(0.0, 0.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, phi=45.0),
    einstein_radius=1.0,
)

"""
We also need the 2D grid the `MassProfile` is evaluated on.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
We now pass the mass profile and grid to a `MassProfilePlotter` and call the `figures_1d` methods to plot its 
convergence as a function of radius.

The `MassProfile` includes the einstein radius as an internal property, meaning we can plot it via an `Include1D` 
object.
"""
include_1d = aplt.Include1D(einstein_radius=True)
mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=mass, grid=grid, include_1d=include_1d
)
mass_profile_plotter.figures_1d(convergence=True)

"""
The appearance of the einstein radius is customized using a `EinsteinRadiusAXVLine` object.

To plot the einstein radius as a vertical line this wraps the following matplotlib method:

 plt.axvline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html
"""
einstein_radius_axvline = aplt.EinsteinRadiusAXVLine(
    linestyle="-.", c="r", linewidth=20
)

mat_plot_1d = aplt.MatPlot1D(einstein_radius_axvline=einstein_radius_axvline)

mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=mass, grid=grid, mat_plot_1d=mat_plot_1d, include_1d=include_1d
)
mass_profile_plotter.figures_1d(convergence=True)

"""
To plot the einstein radius manually, we can pass it into a` Visuals1D` object.
"""
visuals_1d = aplt.Visuals1D(einstein_radius=mass.einstein_radius_from_grid(grid=grid))

mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=mass, grid=grid, visuals_1d=visuals_1d
)
mass_profile_plotter.figures_1d(convergence=True)

"""
Finish.
"""
