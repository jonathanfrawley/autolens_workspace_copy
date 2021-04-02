"""
Plots: LightProfileCentreScatter
================================

This example illustrates how to customize the light profile centres plotted over data.
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
This means the centre of every `LightProfile` of every `Galaxy` in a plot are plotted on the figure. 
A `Tracer` object is a good example of an object with many `LightProfiles`, so lets make one with three.

We will show the plots in the image-plane, however it is the centre's of the source galaxy `LightProfile`'s in the 
source-plane that are plotted.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, phi=45.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.2, 0.2)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge_0=al.lp.SphericalSersic(
        centre=(0.1, 0.1), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
    bulge_1=al.lp.SphericalSersic(
        centre=(0.4, 0.3), intensity=0.3, effective_radius=1.0, sersic_index=2.5
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
We also need the `Grid2D` that we can use to make plots of the `Tracer`'s properties.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
The light profile centres are an internal property of the `Tracer`, so we can plot them via an `Include2D` object.
"""
include_2d = aplt.Include2D(
    light_profile_centres=True,
    mass_profile_centres=False,
    critical_curves=False,
    caustics=False,
)
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, include_2d=include_2d)
tracer_plotter.figures_2d(image=True)


"""
The appearance of the light profile centres are customized using a `LightProfileCentresScatter` object.

To plot the light profile centres this object wraps the following matplotlib method:

 https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.scatter.html
"""
light_profile_centres_scatter = aplt.LightProfileCentresScatter(
    marker="o", c="r", s=150
)
mat_plot_2d = aplt.MatPlot2D(
    light_profile_centres_scatter=light_profile_centres_scatter
)
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, include_2d=include_2d, mat_plot_2d=mat_plot_2d
)
tracer_plotter.figures_2d(image=True)

"""
By specifying two colors to the `LightProfileCentresScatter` object the light profile centres of each plane
are plotted in different colors.
"""
light_profile_centres_scatter = aplt.LightProfileCentresScatter(c=["r", "w"], s=150)
mat_plot_2d = aplt.MatPlot2D(
    light_profile_centres_scatter=light_profile_centres_scatter
)
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, include_2d=include_2d, mat_plot_2d=mat_plot_2d
)
tracer_plotter.figures_2d(image=True)


"""
To plot the light profile centres manually, we can pass them into a` Visuals2D` object. This is useful for plotting 
the centres on figures where they are not an internal property, like an `Array2D`.
"""
light_profile_centres = tracer.extract_attribute(
    cls=al.lp.LightProfile, attr_name="centre"
)

visuals_2d = aplt.Visuals2D(light_profile_centres=light_profile_centres)
image = tracer.image_from_grid(grid=grid)

array_plotter = aplt.Array2DPlotter(
    array=image, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
)
array_plotter.figure_2d()

"""
Finish.
"""
