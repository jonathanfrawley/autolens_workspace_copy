"""
Pipelines: Light Parametric + Mass Light Dark + Source Inversion
================================================================

By chaining together five searches this script fits strong lens `Imaging`, where in the final model:

 - The lens galaxy's light is a parametric bulge+disk `EllipticalSersic` and `EllipticalExponential`.
 - The lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
 - The source galaxy is modeled using an `Inversion`.
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
dataset_name = "light_sersic_exp__mass_mlr_nfw__source_sersic"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

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
path_prefix = path.join(
    "imaging", "chaining", "light_parametric__mass_light_dark__source_parametric"
)

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).

In this analysis, they are used to explicitly set the `mass_at_200` of the elliptical NFW dark matter profile, which is
a model parameter that is fitted for.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In search 1 we fit a lens model where:

 - The lens galaxy's light is a parametric `EllipticalSersic` bulge and `EllipticalExponential` disk, the centres of 
 which are aligned [11 parameters].

 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
bulge = af.PriorModel(al.lp.EllipticalSersic)
disk = af.PriorModel(al.lp.EllipticalExponential)

bulge.centre = disk.centre

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(redshift=0.5, bulge=bulge, disk=disk)
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_light[parametric]", n_live_points=50
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's light and stellar mass is an `EllipticalSersic` bulge and `EllipticalExponential` disk [Parameters 
 fixed to results of search 1].

 - The lens galaxy's dark matter mass distribution is a `EllipticalNFWMCRLudlow` whose centre is aligned with the 
 `EllipticalSersic` bulge and stellar mass model above [3 parameters].

 - The lens mass model also includes an `ExternalShear` [2 parameters].

 - The source galaxy's light is a parametric `EllipticalSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

NOTES:

 - By using the fixed `bulge` and `disk` model from the result of search 1, we are assuming this is a sufficiently 
 accurate fit to the lens's light that it can reliably represent the stellar mass.
"""
bulge = result_1.instance.galaxies.lens.bulge
disk = result_1.instance.galaxies.lens.disk

dark = af.PriorModel(al.mp.EllipticalNFWMCRLudlow)
dark.centre = bulge.centre
dark.mass_at_200 = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e15)
dark.redshift_object = redshift_lens
dark.redshift_source = redshift_source

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=0.5,
            bulge=bulge,
            disk=disk,
            dark=af.PriorModel(al.mp.EllipticalNFW),
            shear=al.mp.ExternalShear,
        ),
        source=al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]_light[fixed]_mass[light_dark]_source[parametric]",
    n_live_points=75,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

We use the results of searches 1 and 2 to create the lens model fitted in search 3, where:

 - The lens galaxy's light and stellar mass is a parametric `EllipticalSersic` bulge and `EllipticalExponential` disk 
 [8 parameters: priors initialized from search 1].

 - The lens galaxy's dark matter mass distribution is a `EllipticalNFWMCRLudlow` whose centre is aligned with the 
 `EllipticalSersic` bulge and stellar mass model above [3 parameters: priors initialized from search 2].

 - The lens mass model also includes an `ExternalShear` [2 parameters: priors initialized from search 2].

 - The source galaxy's light is a parametric `EllipticalSersic` [7 parameters: priors initialized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.

Notes:

 - This phase attempts to address any issues there may have been with the bulge's stellar mass model.
"""
bulge = result_1.model.galaxies.lens.bulge
disk = result_1.model.galaxies.lens.disk

dark = result_2.model.galaxies.lens.dark
dark.centre = bulge.centre

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=0.5,
            bulge=bulge,
            dark=dark,
            shear=result_2.model.galaxies.lens.shear,
        ),
        source=al.GalaxyModel(redshift=1.0, bulge=result_2.model.galaxies.source.bulge),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[3]_light[parametric]_mass[light_dark]_source[parametric]",
    n_live_points=100,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

We use the results of searches 3 to create the lens model fitted in search 4, where:

 - The lens galaxy's light and stellar mass is an `EllipticalSersic` bulge and `EllipticalExponential` 
 disk [Parameters fixed to results of search 3].

 - The lens galaxy's dark matter mass distribution is a `EllipticalNFWMCRLudlow` [Parameters fixed to results of 
 search 3].

 - The lens mass model also includes an `ExternalShear` [Parameters fixed to results of search 3].

 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [2 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

NOTES:

 - This phase allows us to very efficiently set up the resolution of the pixelization and regularization coefficient 
 of the regularization scheme, before using these models to refit the lens mass model.
"""
model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=redshift_lens,
            bulge=result_3.instance.galaxies.lens.bulge,
            disk=result_3.instance.galaxies.lens.disk,
            dark=result_3.instance.galaxies.lens.dark,
            shear=result_3.instance.galaxies.lens.shear,
        ),
        source=al.GalaxyModel(
            redshift=redshift_source,
            pixelization=al.pix.VoronoiMagnification,
            regularization=al.reg.Constant,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[4]_light[fixed]_mass[fixed]_source[inversion_initialization]",
    n_live_points=20,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_4 = search.fit(model=model, analysis=analysis)

"""
__Model +  Search (Search 5)__

We use the results of searches 3 and 4 to create the lens model fitted in search 5, where:

 - The lens galaxy's light and stellar mass is an `EllipticalSersic` bulge and `EllipticalExponential` 
 disk [11 parameters: priors initialized from search 3].

 - The lens galaxy's dark matter mass distribution is a `EllipticalNFWMCRLudlow` [8 parameters: priors initialized 
 from search 3].

The lens mass model also includes an `ExternalShear` [2 parameters: priors initialized from search 3].

 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [parameters fixed to results of search 4].

 - This pixelization is regularized using a `Constant` scheme [parameters fixed to results of search 4]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.
"""
model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=redshift_lens,
            bulge=result_3.model.galaxies.lens.bulge,
            disk=result_3.model.galaxies.lens.disk,
            dark=result_3.model.galaxies.lens.dark,
            shear=result_3.model.galaxies.lens.shear,
        ),
        source=al.GalaxyModel(
            redshift=redshift_source,
            pixelization=result_4.instance.galaxies.source.pixelization,
            regularization=result_4.instance.galaxies.source.regularization,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[5]_light[parametric]_mass[light_dark]_source[inversion]",
    n_live_points=20,
)

"""
__Positions + Analysis + Model-Fit (Search 5)__

We use the `auto_positions` feature, described in `chaining/examples/parametric_to_inversion.py` to remove unphysical
solutions from the `Inversion` model-fitting.
"""
settings_lens = al.SettingsLens(
    auto_positions_factor=3.0, auto_positions_minimum_threshold=0.2
)

analysis = al.AnalysisImaging(dataset=masked_imaging, settings_lens=settings_lens)

result_5 = search.fit(model=model, analysis=analysis)

"""
Finish.
"""