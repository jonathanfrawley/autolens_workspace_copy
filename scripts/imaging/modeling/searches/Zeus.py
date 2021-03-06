"""
Searches: Zeus
==============

Zeus (https://zeus-mcmc.readthedocs.io/en/latest/) is an ensemble MCMC slice sampler.

An MCMC algorithm only seeks to map out the posterior of parameter space, unlike a nested sampling algorithm like
Dynesty, which also aims to estimate the Bayesian evidence if the model. Therefore, in principle, an MCMC approach like
Zeus should be faster than Dynesty.

In our experience, `Zeus`'s performance is on-par with `Dynesty`, except for initializing the lens model using broad
uniformative priors. We use Dynesty by default in all examples because it requires less tuning, but we encourage
you to give Zeus a go yourself, and let us know on the PyAutoLens GitHub if you find an example of a problem where
`Zeus` outperforms Dynesty!
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

imaging = imaging.apply_mask(mask=mask)

"""
__Model__ 

In our experience, zeus is ineffective at initializing a lens model and therefore needs a 'starting point' which is
near the highest likelihood lens models. We set this starting point up below by manually inputting `UniformPriors` on
every parameter, where the centre of these priors is near the true values of the simulated lens data.

Given this need for a robust starting point, Zeus is only suited to model-fits where we have this information. It may
therefore be useful when performing lens modeling search chaining (see HowToLens chapter 3). However, even in such
circumstances, we have found that is often outperformed by other searches such as Dynesty and Zeus for both speed
and accuracy.
"""
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

"""
__Analysis__ 

We create the Analysis as per using.
"""
analysis = al.AnalysisImaging(dataset=imaging)

"""
__Search__

Below we use zeus to fit the lens model, using the model with start points as described above. See the Zeus docs
for a description of what the input parameters below do.
"""
search = af.Zeus(
    path_prefix=path.join("imaging", "searches"),
    name="Zeus",
    unique_tag=dataset_name,
    nwalkers=30,
    nsteps=200,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    tune=False,
    tolerance=0.05,
    patience=5,
    maxsteps=10000,
    mu=1.0,
    maxiter=10000,
    vectorize=False,
    check_walkers=True,
    shuffle_ensemble=True,
    light_mode=False,
    iterations_per_update=5000,
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can use an `ZeusPlotter` to create a corner plot, which shows the probability density function (PDF) of every
parameter in 1D and 2D.
"""
zeus_plotter = aplt.ZeusPlotter(samples=result.samples)
zeus_plotter.corner()

"""
Finish.
"""
