"""
Plots: EmceePlotter
=====================

This example illustrates how to plot visualization summarizing the results of a emcee non-linear search using
a `ZeusPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
First, lets create a result via emcee by repeating the simple model-fit that is performed in 
the `modeling/mass_total__source_parametric.py` example.

We use a model with an initialized starting point, which is necessary for lens modeling with emcee.
"""
dataset_name = "mass_sie__source_sersic"

search = af.Emcee(
    path_prefix=path.join("plot"),
    name="EmceePlotter",
    unique_tag=dataset_name,
    nwalkers=30,
    nsteps=500,
)

dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

mass = af.Model(al.mp.EllIsothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.elliptical_comps.elliptical_comps_0 = af.UniformPrior(
    lower_limit=-0.3, upper_limit=0.3
)
mass.elliptical_comps.elliptical_comps_1 = af.UniformPrior(
    lower_limit=-0.3, upper_limit=0.3
)
mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=2.0)

shear = af.Model(al.mp.ExternalShear)
shear.elliptical_comps.elliptical_comps_0 = af.UniformPrior(
    lower_limit=-0.1, upper_limit=0.1
)
shear.elliptical_comps.elliptical_comps_1 = af.UniformPrior(
    lower_limit=-0.1, upper_limit=0.1
)

bulge = af.Model(al.lp.EllSersic)
bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.elliptical_comps.elliptical_comps_0 = af.UniformPrior(
    lower_limit=-0.3, upper_limit=0.3
)
bulge.elliptical_comps.elliptical_comps_1 = af.UniformPrior(
    lower_limit=-0.3, upper_limit=0.3
)
bulge.intensity = af.UniformPrior(lower_limit=0.1, upper_limit=0.5)
bulge.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.4)
bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=2.0)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=imaging)

result = search.fit(model=model, analysis=analysis)
"""
We now pass the samples to a `EmceePlotter` which will allow us to use emcee's in-built plotting libraries to 
make figures.

The emcee readthedocs describes fully all of the methods used below 

 - https://emcee.readthedocs.io/en/stable/user/sampler/
 
 The plotter wraps the `corner` method of the library `corner.py` to make corner plots of the PDF:

- https://corner.readthedocs.io/en/latest/index.html
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
emcee_plotter = aplt.EmceePlotter(samples=result.samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter in the model fit.
"""
emcee_plotter.corner(
    bins=20,
    range=None,
    color="k",
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    divergences=False,
    divergences_kwargs=None,
    labeller=None,
)


"""
The `trajectories` method shows the likelihood of every parameter as a function of parameter value, colored by every
individual walker.
"""
emcee_plotter.trajectories()

"""
The `likelihood_series` method shows the likelihood as a function of step number, colored by every individual walker.
"""
emcee_plotter.time_series()

"""
The `time_series` method shows the likelihood of every parameter as a function of step number, colored by every
individual walker.
"""
emcee_plotter.likelihood_series()


"""
Finish.
"""
