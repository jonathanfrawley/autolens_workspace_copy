"""
Customize: Non-linear Searches
==============================

All example model-fits are performed using the nested sampling algorithm `Dynesty`, which we have found to be the most
effective non-linear search for performing lens modeling.

However, **PyAutoLens** supports a range of non-linear searches which are described here, which you may wish to
experiment with to see if they can outperform Dynesty for your lens modeling problem.

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
__Dataset + Masking__

Load and plot the strong lens dataset `mass_sie__source_sersic` via .fits files, which we will fit with the lens model.
"""
dataset_name = "mass_sie__source_sersic"
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

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

"""
__Model + Analysis__ 

The code below performs the normal steps to set up a model and analysis class. We omit comments of this code as you 
should be familiar with it and it is not specific to this example!
"""
lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic)

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(lens=lens, source=source)
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

"""
__Search: Emcee (MCMC)__

Emcee (https://github.com/dfm/emcee) is an ensemble MCMC sampler that is very popular in Astrophysics.

An MCMC algorithm only seeks to map out the posterior of parameter space, unlike a nested sampling algorithm like 
Dynesty, which also aims to estimate the Bayesian evidence if the model. Therefore, in principle, an MCMC approach like
Emcee should be faster than Dynesty. However, in our experience this is not the case for lens model, if you can
demonstrate that Emcee is please let us know on the PyAutoLens Github!
"""
search = af.Emcee(
    path_prefix=path.join("imaging", "customize", "non_linear_searches"),
    name="emcee",
    nwalkers=50,
    nsteps=1000,
)

search.fit(model=model, analysis=analysis)

"""
__Search: PySwarms (Optimizer)__

PySwarms (https://pyswarms.readthedocs.io/en/latest/index.html) is a particle swarm optimizer, which supports both
local optimization (e.g. finding a local maximum in the likelihood given the starting point) and global optimization
(e.g. finding the global maxima).

An `optimizer` seeks to only maximize the log likelihood of the fit and does not attempt to infer the errors on the 
model parameters. Optimizers are therefore useful when we want to find a lens model that fits the data well, but do 
not care about the full posterior of parameter space (e.g. the errors). 

However, much like our attempts with MCMC metohds like Emcee, we have found Optimizers to be inaccurate when performing
lens modeling. Again, if you can demonstrate PySwarms working better than Dynbesty, please contact us on the PyAutoLens
GitHub to know how you managed it!
"""
search = af.PySwarmsGlobal(
    path_prefix=path.join("imaging", "customize", "non_linear_searches"),
    name="pyswarms_global",
    n_particles=50,
    iters=5000,
)

search.fit(model=model, analysis=analysis)

search = af.PySwarmsLocal(
    path_prefix=path.join("imaging", "customize", "non_linear_searches"),
    name="pyswarms_local",
    n_particles=50,
    iters=5000,
)

search.fit(model=model, analysis=analysis)

"""
Finish.
"""
