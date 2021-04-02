"""
SLaM (Source, Light and Mass): Mass Total + Subhalo NFW + Source Parametric Sensitivity Mapping
===============================================================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllSersic` profile for the bulge, this will be used in the subsequent MASS TOTAL PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE, MASS TOTAL PIPELINE and SUBHALO PIPELINE this SLaM script fits `Interferometer` of a strong lens
system, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllIsothermal`.
 - The source galaxy is an `EllSersic`.

It ends by performing sensitivity mapping of the data using the above model, so as to determine where in the data
subhalos of a given mass could have been detected if present.

This runner uses the SLaM pipelines:

 `source_parametric/no_lens_light`
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

Load the `Interferometer` data, define the visibility and real-space masks and plot them.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "interferometer", dataset_name)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    settings=al.SettingsInterferometer(transformer_class=al.TransformerNUFFT),
)

interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.subplot_interferometer()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join(
    "interferometer", "slam", "mass_total__subhalo_nfw__source_parametric"
)

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

 We use the following optional settings:
 
 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
"""
analysis = al.AnalysisInterferometer(dataset=interferometer)

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
__MASS TOTAL PIPELINE (no lens light)__

The MASS TOTAL PIPELINE (no lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors. In this example it:

 - Uses an `EllPowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
 - Uses the `EllSersic` model representing a bulge for the source's light.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.
"""
analysis = al.AnalysisInterferometer(dataset=interferometer)

mass_results = slam.mass_total.no_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_parametric_results,
    mass=af.Model(al.mp.EllPowerLaw),
)

"""
__SUBHALO PIPELINE (sensitivity mapping)__

The SUBHALO PIPELINE (sensitivity mapping) performs sensitivity mapping of the data using the lens model
fitted above, so as to determine where subhalos of what mass could be detected in the data. A full description of
sensitivty mapping if given in the script `sensitivity_mapping.py`.

Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
data simulated by the `simulate_function` for that model. This requires us to write a wrapper around the 
PyAutoLens `AnalysisInterferometer` class.
"""


class AnalysisInterferometerSensitivity(al.AnalysisInterferometer):
    def __init__(self, dataset):

        super().__init__(dataset=dataset)

        self.hyper_galaxy_image_path_dict = (
            mass_results.last.hyper_galaxy_image_path_dict
        )
        self.hyper_model_image = mass_results.last.hyper_model_image


subhalo_results = slam.subhalo.sensitivity_mapping_interferometer(
    path_prefix=path_prefix,
    analysis_cls=AnalysisInterferometerSensitivity,
    uv_wavelengths=interferometer.uv_wavelengths,
    real_space_mask=real_space_mask,
    mass_results=mass_results,
    subhalo_mass=af.Model(al.mp.SphNFWMCRLudlow),
    grid_dimension_arcsec=3.0,
    number_of_steps=2,
    number_of_cores=2,
)

"""
Finish.
"""
