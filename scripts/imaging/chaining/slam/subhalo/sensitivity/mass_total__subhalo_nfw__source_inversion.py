"""
SLaM (Source, Light and Mass): Mass Total + Subhalo NFW + Source Parametric Sensitivity Mapping
===============================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllSersic` profile for the bulge, this will be used in the subsequent MASS TOTAL PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE, SOURCE INVERSION PIPELINE and a MASS TOTAL PIPELINE this SLaM script fits `Imaging` of a
strong lens system, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllIsothermal`.
 - The source galaxy is an `Inversion`.

It ends by performing sensitivity mapping of the data using the above model, so as to determine where in the data
subhalos of a given mass could have been detected if present.

This runner uses the SLaM pipelines:

 `source_parametric/no_lens_light`
 `source__inversion/source_inversion__no_lens_light`
 `mass_total/no_lens_light`
 `subhalo/sensitivity_mapping`

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
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.
"""
dataset_name = "mass_sie__subhalo_nfw__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

masked_imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=masked_imaging)
imaging_plotter.subplot_imaging()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "slam", "mass_total__subhalo_nfw__source_inversion")

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
__SOURCE PARAMETRIC PIPELINE (no lens light)__

The SOURCE PARAMETRIC PIPELINE (no lens light) uses one search to initialize a robust model for the source galaxy's 
light, which in this example:

 - Uses a parametric `EllSersic` bulge for the source's light (omitting a disk / envelope).
 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.

__Settings__:
 
 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the SOURCE INVERSION 
 PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=masked_imaging)

source_parametric_results = slam.source_parametric.no_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    mass=af.Model(al.mp.EllIsothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)

"""
__SOURCE INVERSION PIPELINE (no lens light)__

The SOURCE INVERSION PIPELINE (no lens light) uses four searches to initialize a robust model for the `Inversion` that
reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `VoronoiBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the
 SOURCE INVERSION PIPELINE.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_inversion.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
settings_lens = al.SettingsLens(positions_threshold=0.2)

analysis = al.AnalysisImaging(
    dataset=masked_imaging,
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
__MASS TOTAL PIPELINE (no lens light)__

The MASS TOTAL PIPELINE (no lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors. In this example it:

 - Uses an `EllPowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
 
 - Uses the `EllSersic` model representing a bulge for the source's light.
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS TOTAL 
 PIPELINE.

__Preloads__:
 
 - Pixelization: We preload the pixelization using the maximum likelihood hyper-result of the SOURCE INVERSION PIPELINE. 
 This ensures the source pixel-grid is not recalculated every iteration of the log likelihood function, speeding up 
 the model-fit (this is only possible because the source pixelization is fixed).    
"""
preloads = al.Preloads.setup(
    result=source_inversion_results.last.hyper, pixelization=True
)

analysis = al.AnalysisImaging(
    dataset=masked_imaging,
    positions=source_inversion_results.last.image_plane_multiple_image_positions,
    preloads=preloads,
)

mass_results = slam.mass_total.no_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_inversion_results,
    mass=af.Model(al.mp.EllPowerLaw),
)

"""
__SUBHALO PIPELINE (sensitivity mapping)__

The SUBHALO PIPELINE (sensitivity mapping) performs sensitivity mapping of the data using the lens model
fitted above, so as to determine where subhalos of what mass could be detected in the data. A full description of
sensitivty mapping if given in the script `sensitivity_mapping.py`.

Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
data simulated by the `simulate_function` for that model. This requires us to write a wrapper around the 
PyAutoLens `AnalysisImaging` class.

__Preloads__:
 
 - Pixelization: We preload the pixelization using the maximum likelihood hyper-result of the SOURCE INVERSION PIPELINE. 
 This ensures the source pixel-grid is not recalculated every iteration of the log likelihood function, speeding up 
 the model-fit (this is only possible because the source pixelization is fixed).   
"""
preloads = al.Preloads.setup(
    result=source_inversion_results.last.hyper, pixelization=True
)


class AnalysisImagingSensitivity(al.AnalysisImaging):
    def __init__(self, dataset):

        super().__init__(dataset=dataset, preloads=preloads)

        self.hyper_galaxy_image_path_dict = (
            mass_results.last.hyper_galaxy_image_path_dict
        )
        self.hyper_model_image = mass_results.last.hyper_model_image


subhalo_results = slam.subhalo.sensitivity_mapping_imaging(
    path_prefix=path_prefix,
    analysis_cls=AnalysisImagingSensitivity,
    mask=mask,
    psf=masked_imaging.psf,
    mass_results=mass_results,
    subhalo_mass=af.Model(al.mp.SphNFWMCRLudlow),
    grid_dimension_arcsec=3.0,
    number_of_steps=5,
    number_of_cores=2,
)

"""
Finish.
"""
