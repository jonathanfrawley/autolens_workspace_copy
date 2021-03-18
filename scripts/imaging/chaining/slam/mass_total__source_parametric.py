"""
SLaM (Source, Light and Mass): Mass Total + Source Parametric
=============================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllipticalSersic` profile for the bulge, this will be used in the subsequent MASS PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE and a MASS PIPELINE this SLaM script fits `Imaging` of a strong lens system, where
in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllipticalPowerLaw`.
 - The source galaxy's light is a parametric `EllipticalSersic`.

This uses the SLaM pipelines:

 `source__parametric/source_parametric__no_lens_light`
 `mass__total/mass__total__no_lens_light`

Check them out for a full description of the analysis!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
from .pipelines import source__parametric, mass__total
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "mass_sie__source_sersic"
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
path_prefix = path.join("imaging", "slam", "mass_total__source_parametric")

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit as is used identically to the
hyper pipeline examples.

The `SetupHyper` input `hyper_fixed_after_source` fixes the hyper-parameters to the values computed by the hyper 
extension at the end of the SOURCE PIPELINE. By fixing the hyper-parameter values at this point, model comparison 
of different models in the LIGHT PIPELINE and MASS PIPELINE can be performed consistently.
"""
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__SOURCE PARAMETRIC PIPELINE (no lens light)__

The SOURCE PARAMETRIC PIPELINE (no lens light) uses one search to initialize a robust model for the source galaxy's 
light, which in this example:
 
 - Uses a parametric `EllipticalSersic` bulge for the source's light (omitting a disk / envelope).
 - Uses an `EllipticalIsothermal` model for the lens's total mass distribution with an `ExternalShear`.
 - Fixes the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS PIPELINE).
"""
source_results = source__parametric.source_parametric__no_lens_light(
    path_prefix=path_prefix,
    analysis=al.AnalysisImaging(dataset=masked_imaging),
    setup_hyper=setup_hyper,
    mass=af.PriorModel(al.mp.EllipticalIsothermal),
    shear=af.PriorModel(al.mp.ExternalShear),
    source_bulge=af.PriorModel(al.lp.EllipticalSersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)


"""
__MASS TOTAL PIPELINE (no lens light)__

The MASS TOTAL PIPELINE (no lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors. In this example it:

 - Uses an `EllipticalPowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS PIPELINE.
"""
mass_results = mass__total.mass__total__no_lens_light(
    path_prefix=path_prefix,
    analysis=al.AnalysisImaging(dataset=masked_imaging, results=source_results),
    setup_hyper=setup_hyper,
    source_results=source_results,
    mass=af.PriorModel(al.mp.EllipticalPowerLaw),
)

"""
Finish.
"""
