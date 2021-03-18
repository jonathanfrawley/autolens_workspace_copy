"""
Chaining: SIE to Power-law
==========================

In this script, we chain two searches to fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `EllipticalPowerLaw`.
 - The source galaxy's light is a parametric `EllipticalSersic`.

The two searches break down as follows:

 1) Models the lens galaxy's mass as an `EllipticalIsothermal` and the source galaxy's light as an `EllipticalSersic`.
 2) Models the lens galaxy's mass an an `EllipticalPowerLaw` and the source galaxy's light as an `EllipticalSersic`.

__Why Chain?__

The `EllipticalPower` is a general form of the `EllipticalIsothermal` which has one additional parameter, the `slope`,
which controls the inner mass distribution as follows:

 - A higher slope concentrates more mass in the central regions of the mass profile relative to the outskirts.
 - A lower slope shallows the inner mass distribution reducing its density relative to the outskirts. 

By allowing the lens model to vary the mass profile's inner distribution, its non-linear parameter space becomes
significantly more complex and a notable degeneracy appears between the mass model`s mass normalization, elliptical
components and slope. This is challenging to sample in an efficient and robust manner, especially when the non-linear
search's initial samples use broad uniform priors on the lens and source parameters.

Search chaining allows us to begin by fitting an `EllipticalIsothermal` model and therefore estimate the lens's mass
model and the source parameters via a non-linear parameter space that does not have a strong of a parameter degeneracy
present. This makes the model-fit more efficient and reliable.

The second search then fits the `EllipticalPowerLaw`, using prior passing to initialize the mass and elliptical
components of the lens galaxy as well as the source galaxy's light profile.
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

Load, plot and mask the `Imaging` data.
"""
dataset_name = "mass_power_law__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=masked_imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "sie_to_power_law")

"""
__Model (Search 1)__

In search 1 we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` with `ExternalShear` [7 parameters].
 - The source galaxy's light is a parametric `EllipticalSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
lens = al.GalaxyModel(
    redshift=0.5, mass=al.mp.EllipticalIsothermal, shear=al.mp.ExternalShear
)
source = al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic)

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(lens=lens, source=source)
)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]__sie", n_live_points=50
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is an `EllipticalPowerLaw` with `ExternalShear` [8 parameters: priors 
 initialized from search 1].
 - The source galaxy's light is again a parametric `EllipticalSersic` [7 parameters: priors initialized from search 1].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.

The term `model` below tells PyAutoLens to pass the source model as model-components that are to be fitted for by the 
non-linear search. Because the source model does not change we can pass its priors by simply using the`model` attribute
of the result:
"""
source = result_1.model.galaxies.source

"""
However, we cannot use this to pass the lens galaxy, because its mass model must change from an `EllipticalIsothermal` 
to an `EllipticalPowerLaw`. The following code would not change the mass model to an `EllipticalPowerLaw`:
 
 `lens = result.model.galaxies.lens`
 
We can instead use the `take_attributes` method to pass the priors. Below, we pass the lens of the result above to a
new `EllipticalPowerLaw`, which will find all parameters in the `EllipticalIsothermal` model that share the same name
as parameters in the `EllipticalPowerLaw` and pass their priors (in this case, the `centre`, `elliptical_comps` 
and `einstein_radius`).

This leaves the `slope` parameter of the `EllipticalPowerLaw` with its default `UniformPrior` which has a 
`lower_limit=1.5` and `upper_limit=3.0`.
"""
mass = af.PriorModel(al.mp.EllipticalPowerLaw)
mass.take_attributes(result_1.model.galaxies.lens.mass)
shear = result_1.model.galaxies.lens.shear

lens = al.GalaxyModel(redshift=0.5, mass=mass, shear=shear)

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(lens=lens, source=source)
)

"""
__Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the `model.info` file of the search 2 model-fit to ensure the priors were passed correctly, as 
well as the checkout the results to ensure an accurate power-law mass model is inferred.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]__power_law", n_live_points=75
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a lens mass model as an `EllipticalIsothermal` and 
passed its priors to then fit the more complex `EllipticalPowerLaw` model. 

This removed difficult-to-fit degeneracies from the non-linear parameter space in search 1, providing a more robust 
and efficient model-fit.

__Pipelines__

Advanced search chaining uses `pipelines` that chain together multiple searches to perform very complex lens modeling 
in a robust and efficient way. 

The following example pipelines fits a power-law, using the same approach demonstrated in this script of first 
fitting an `EllipticalIsothermal`:

 `autolens_workspace/imaging/chaining/pipelines/no_lens_light/mass_total__source_parametric.py`
 
 __SLaM (Source, Light and Mass)__
 
An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling 
processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
mass model. 

The SLaM pipelines assume an `EllipticalIsothermal` throughout the Source and Light pipelines, and only switch to a
more complex mass model (like the `EllipticalPowerLaw`) in the final Mass pipeline.
"""
