"""
Searches: PySwarms
==================

PySwarms is a  particle swarm optimization (PSO) algorithm.

Information about PySwarms can be found at the following links:

 - https://github.com/ljvmiranda921/pyswarms
 - https://pyswarms.readthedocs.io/en/latest/index.html
 - https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html#module-pyswarms.single.global_best

An PSO algorithm only seeks to only find the maximum likelihood lens model, unlike MCMC or nested sampling algorithms 
like Zzeus and dynesty, which aims to map-out parameter space and infer errors on the parameters.Therefore, in 
principle, a PSO like PySwarm should fit a lens model very fast.

In our experience, the parameter spaces fitted by lens models are too complex for `PySwarms` to be used without a lot
of user attention and care.  Nevertheless, we encourage you to give it a go yourself, and let us know on the PyAutoLens 
GitHub if you find an example of a problem where `PySwarms` outperforms Dynesty!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np
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

imaging = imaging.apply_mask(mask=mask)

"""
__Model__ 

In our experience, pyswarms is ineffective at initializing a lens model and therefore needs a the initial swarm of
particles to surround the the highest likelihood lens models. We set this starting point up below by manually inputting 
`GaussianPriors` on every parameter, where the centre of these priors is near the true values of the simulated lens data.

Given this need for a robust starting point, PySwarms is only suited to model-fits where we have this information. It may
therefore be useful when performing lens modeling search chaining (see HowToLens chapter 3). However, even in such
circumstances, we have found that is often unrealible and often infers a local maxima.
"""
mass = af.Model(al.mp.EllIsothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
mass.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
mass.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
mass.einstein_radius = af.GaussianPrior(mean=1.4, sigma=0.4)

shear = af.Model(al.mp.ExternalShear)
shear.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
shear.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

bulge = af.Model(al.lp.EllSersic)
bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.intensity = af.GaussianPrior(mean=0.3, sigma=0.3)
bulge.effective_radius = af.GaussianPrior(mean=0.2, sigma=0.2)
bulge.sersic_index = af.GaussianPrior(mean=1.0, sigma=1.0)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__ 

We create the Analysis as per using.
"""
analysis = al.AnalysisImaging(dataset=imaging)

"""
__Search__

Below we use `PySwarmsGlobal` to fit the lens model, using the model where the particles start as described above. 
See the PySwarms docs for a description of what the input parameters below do and what the `Global` search technique is.
"""
search = af.PySwarmsGlobal(
    path_prefix=path.join("imaging", "searches"),
    name="PySwarmsGlobal",
    unique_tag=dataset_name,
    n_particles=30,
    iters=300,
    cognitive=0.5,
    social=0.3,
    inertia=0.9,
    ftol=-np.inf,
    iterations_per_update=1000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can use an `PySwarmsPlotter` to create a corner plot, which shows the probability density function (PDF) of every
parameter in 1D and 2D.
"""
pyswarms_plotter = aplt.PySwarmsPlotter(samples=result.samples)
pyswarms_plotter.cost_history()

"""
__Search__

We can also use a `PySwarmsLocal` to fit the lens model
"""
search = af.PySwarmsLocal(
    path_prefix=path.join("imaging", "searches"),
    name="PySwarmsLocal",
    unique_tag=dataset_name,
    n_particles=30,
    iters=300,
    cognitive=0.5,
    social=0.3,
    inertia=0.9,
    ftol=-np.inf,
    iterations_per_update=1000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

pyswarms_plotter = aplt.PySwarmsPlotter(samples=result.samples)
pyswarms_plotter.cost_history()

"""
Finish.
"""
