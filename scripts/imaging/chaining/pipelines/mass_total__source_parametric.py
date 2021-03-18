"""
Pipelines: Mass Total + Source Parametric
=========================================

By chaining together two searches this script fits strong lens `Imaging`, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal`.
 - The source galaxy's light is a bulge+disk parametric `EllipticalSersic`'s.
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
path_prefix = path.join("imaging", "chaining", "mass_total__source_parametric")

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
            redshift=0.5, mass=al.mp.EllipticalIsothermal, shear=al.mp.ExternalShear
        ),
        source=al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic),
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

 - The lens galaxy's total mass distribution is an `EllipticalPowerLaw` with `ExternalShear` [8 parameters: priors 
 initialized from search 1].
 
 - The source galaxy's light is a bulge+disk using two parametric `EllipticalSersic`'s whose centres are shared
 [12 parameters: priors of bulge initialized from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=20.

The lens mass model priors of the `EllipticalPowerLaw` and `ExternalShear` are passed from search 1. 
The source light model priors of the `EllipticalSersic` bulge are passed from search 1.
"""
mass = af.PriorModel(al.mp.EllipticalPowerLaw)
mass.take_attributes(result_1.model.galaxies.lens.mass)

bulge = result_1.model.galaxies.source.bulge
disk = af.PriorModel(al.lp.EllipticalSersic)

bulge.centre = disk.centre

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=0.5, mass=mass, shear=result_1.model.galaxies.lens.shear
        ),
        source=al.GalaxyModel(redshift=1.0, bulge=bulge, disk=disk),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]_mass[total]_source[parametric]",
    n_live_points=100,
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_2 = search.fit(model=model, analysis=analysis)

"""
Finish.
"""