"""
Pipelines: Mass Total + Source Inversion
========================================

By chaining together three searches this script fits strong lens `Imaging`, where in the final model:
.
 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllipticalPowerLaw`.
 - The source galaxy is an `Inversion`.
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
dataset_name = "mass_sie__source_sersic_x2"
pixel_scales = 0.1

dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scales,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "mass_total__source_inversion")

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In search 1 we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric `EllipticalSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=redshift_lens,
            mass=al.mp.EllipticalIsothermal,
            shear=al.mp.ExternalShear,
        ),
        source=al.GalaxyModel(redshift=redshift_source, bulge=al.lp.EllipticalSersic),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[1]_mass[sie]_source[parametric]",
    n_live_points=50,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` and `ExternalShear` [Parameters fixed to 
 results of search 1].
 
 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [2 parameters].
 
 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This phase allows us to very efficiently set up the resolution of the pixelization and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.
"""
model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=redshift_lens,
            mass=result_1.instance.galaxies.lens.mass,
            shear=result_1.instance.galaxies.lens.shear,
        ),
        source=al.GalaxyModel(
            redshift=redshift_source,
            pixelization=al.pix.VoronoiMagnification,
            regularization=al.pix.Rectangular,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]_mass[sie]_source[inversion_initialization]",
    n_live_points=20,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search (Search 3)__

We use the results of searches 1 and 2 to create the lens model fitted in search 3, where:

 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` and `ExternalShear` [7 parameters: priors 
 initialized from search 1].
 
 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [parameters fixed to results of search 2].
 
 - This pixelization is regularized using a `Constant` scheme [parameters fixed to results of search 2]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This phase allows us to very efficiently set up the resolution of the pixelization and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.
"""
model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=redshift_lens,
            mass=result_1.model.galaxies.lens.mass,
            shear=result_1.model.galaxies.lens.shear,
        ),
        source=al.GalaxyModel(
            redshift=redshift_source,
            pixelization=result_2.instance.galaxies.source.pixelization,
            regularization=result_2.instance.galaxies.source.regularization,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[3]_mass[sie]_source[inversion]",
    n_live_points=50,
)

"""
__Positions + Analysis + Model-Fit (Search 3)__

We use the `auto_positions` feature, described in `chaining/examples/parametric_to_inversion.py` to remove unphysical
solutions from the `Inversion` model-fitting.
"""
settings_lens = al.SettingsLens(
    auto_positions_factor=3.0, auto_positions_minimum_threshold=0.2
)

analysis = al.AnalysisImaging(
    dataset=masked_imaging, results=result_2, settings_lens=settings_lens
)

result_3 = search.fit(model=model, analysis=analysis)


"""
__Refine__
"""


"""
__Model + Search + Analysis + Model-Fit (Search 4)__

We use the results of searches 2 and 4 to create the lens model fitted in search 4, where:

 - The lens galaxy's total mass distribution is an `EllipticalPowerLaw` and `ExternalShear` [8 parameters: priors 
 initialized from search 3].
 
 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [parameters fixed to results of search 2].
 
 - This pixelization is regularized using a `Constant` scheme [parameters fixed to results of search 2]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This phase allows us to very efficiently set up the resolution of the pixelization and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.
"""
mass = af.PriorModel(al.mp.EllipticalPowerLaw)
mass.take_attributes(result_3.model.galaxies.lens.mass)

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=redshift_lens, mass=mass, shear=result_3.model.galaxies.lens.shear
        ),
        source=al.GalaxyModel(
            redshift=redshift_source,
            pixelization=result_2.instance.galaxies.source.pixelization,
            regularization=result_2.instance.galaxies.source.regularization,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[4]_mass[total]_source[inversion]",
    n_live_points=100,
)

analysis = al.AnalysisImaging(dataset=masked_imaging, settings_lens=settings_lens)

result_4 = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
