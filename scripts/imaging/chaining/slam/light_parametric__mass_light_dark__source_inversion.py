"""
SLaM (Source, Light and Mass): Light Parametric + Mass Total + Source Parametric
================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllSersic` profile for the bulge, this will be used in the subsequent MASS LIGHT DARK PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE, LIGHT PIPELINE and a MASS LIGHT DARK PIPELINE this SLaM script fits `Imaging` of
a strong lens system, where in the final model:

 - The lens galaxy's light is a bulge+disk `EllSersic` and `EllSersic`.
 - The lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
 - The lens galaxy's dark matter mass distribution is modeled as a `EllNFWMCRLudlow`.
 - The source galaxy's light is a parametric `Inversion`.

This runner uses the SLaM pipelines:

 `source_parametric/with_lens_light`
 `source_inversion/with_lens_light`
 `light_parametric/with_lens_light`
 `mass_total/mass_light_dark`

Check them out for a detailed description of the analysis!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import sys
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam

"""
__Dataset__ 

Load the `Imaging` data, define the `Mask2D` and plot them.
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

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "slam", dataset_name)

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit.
"""
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__SOURCE PARAMETRIC PIPELINE (with lens light)__

The SOURCE PARAMETRIC PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:

 - Uses a parametric `EllSersic` bulge and `EllSersic` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS LIGHT DARK 
 PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=imaging)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllSersic)
bulge.centre = (0.0, 0.0)
disk.centre = (0.0, 0.0)

source_parametric_results = slam.source_parametric.with_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    lens_bulge=bulge,
    lens_disk=disk,
    mass=af.Model(al.mp.EllIsothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__SOURCE INVERSION PIPELINE (with lens light)__

The SOURCE INVERSION PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion` 
that reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `VoronoiBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the
 SOURCE INVERSION PIPELINE.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_inversion.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
settings_lens = al.SettingsLens(
    positions_threshold=source_parametric_results.last.positions_threshold_from(
        factor=3.0, minimum_threshold=0.2
    )
)

analysis = al.AnalysisImaging(
    dataset=imaging,
    positions=source_parametric_results.last.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
)

source_inversion_results = slam.source_inversion.no_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_parametric_results=source_parametric_results,
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
)


"""
__LIGHT PARAMETRIC PIPELINE__

The LIGHT PARAMETRIC PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE INVERSION PIPELINE.
In this example it:

 - Uses a parametric `EllSersic` bulge and `EllSersic` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE PARAMETRIC PIPELINE to initialize priors].

 - Uses an `EllIsothermal` model for the lens's total mass distribution [fixed from SOURCE PARAMETRIC PIPELINE].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE INVERSION PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
 
__Preloads__: 
 
 - Inversion: We preload linear algebra matrices used by the inversion using the maximum likelihood hyper-result of the 
 SOURCE INVERSION PIPELINE. This ensures these matrices are not recalculated every iteration of the log likelihood 
 function, speeding up the model-fit (this is possible because the mass model and source pixelization are fixed). 
"""
preloads = al.Preloads.setup(result=source_inversion_results.last.hyper, inversion=True)

analysis = al.AnalysisImaging(
    dataset=imaging, hyper_result=source_inversion_results.last, preloads=preloads
)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllSersic)
bulge.centre = disk.centre

light_results = slam.light_parametric.with_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_inversion_results,
    lens_bulge=bulge,
    lens_disk=disk,
)

"""
__MASS LIGHT DARK PIPELINE (with lens light)__

The MASS LIGHT DARK PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of 
accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT PARAMETRIC PIPELINE to 
initialize the model priors . In this example it:

 - Uses a parametric `EllSersic` bulge and `EllSersic` disk with centres aligned for the lens galaxy's 
 light and its stellar mass [12 parameters: fixed from LIGHT PARAMETRIC PIPELINE].

 - The lens galaxy's dark matter mass distribution is a `EllNFWMCRLudlow` whose centre is aligned with bulge of 
 the light and stellar mass mdoel above [5 parameters].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE INVERSION PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.
 
__Settings__:

 - Hyper: We may be using hyper features and therefore pass the result of the SOURCE INVERSION PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_inversion.py`) to remove unphysical solutions from the `Inversion` model-fitting.
 
__Preloads__:
 
 - Pixelization: We preload the pixelization using the maximum likelihood hyper-result of the SOURCE INVERSION PIPELINE. 
 This ensures the source pixel-grid is not recalculated every iteration of the log likelihood function, speeding up 
 the model-fit (this is only possible because the source pixelization is fixed). 
"""
settings_lens = al.SettingsLens(
    positions_threshold=source_inversion_results.last.positions_threshold_from(
        factor=3.0, minimum_threshold=0.2
    )
)

preloads = al.Preloads.setup(
    result=source_inversion_results.last.hyper, pixelization=True
)

analysis = al.AnalysisImaging(
    dataset=imaging,
    hyper_result=source_inversion_results.last,
    positions=source_inversion_results.last.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
    preloads=preloads,
)

lens_bulge = af.Model(al.lmp.EllSersic)
dark = af.Model(al.mp.EllNFWMCRLudlow)

dark.centre = lens_bulge.centre

mass_results = slam.mass_light_dark.with_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_inversion_results,
    light_results=light_results,
    lens_bulge=lens_bulge,
    lens_disk=af.Model(al.lmp.EllSersic),
    lens_envelope=None,
    dark=dark,
)

"""
Finish.
"""
