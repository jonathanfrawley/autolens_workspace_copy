"""
Plots: MultiSubPlots
====================

This example illustrates how to plot figures from different plotters on the same subplot, using the example of
combining an `ImagingPlotter` and `FitImagingPlotter`.
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
First, lets load example imaging of of a strong lens as an `Imaging` object.
"""
dataset_name = "light_sersic__mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
We now mask the data and fit it with a `Tracer` to create a `FitImaging` object.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.05, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(imaging=imaging, tracer=tracer)

"""
We now pass the imaging to an `ImagingPlotter` and the fit to an `FitImagingPlotter`.
"""
imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)

"""
We next pair the `MatPlot2D` objects of the two plotters, which ensures the figures plot on the same subplot.
"""
imaging_plotter.mat_plot_2d = fit_imaging_plotter.mat_plot_2d

"""
We next open the subplot figure, specifying how many subplot figures will be on our image.
"""
imaging_plotter.open_subplot_figure(number_subplots=2)

"""
We now call the `figures_2d` method of all the plots we want to be included on our subplot. These figures will appear
seqeuencially in the subplot in the order we call them.
"""
imaging_plotter.figures_2d(image=True)
fit_imaging_plotter.figures_2d(residual_map=True)

"""
This outputs the figure, which in this example goes to your display as we did not specify a file format.
"""
imaging_plotter.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot")

"""
Close the subplot figure, in case we were to make another subplot.
"""
imaging_plotter.close_subplot_figure()
