"""
SLaM (Source, Light and Mass): Mass Total + Source Parametric
=============================================================

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own inputs which
which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `EllipticalSersic` profile for the bulge, this will be used in the subsequent MASS TOTAL PIPELINE.

Using a SOURCE PARAMETRIC PIPELINE and a MASS TOTAL PIPELINE this SLaM script fits `Interferometer` of a strong lens system, where
in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllipticalPowerLaw`.
 - The source galaxy's light is a parametric `EllipticalSersic`.

This uses the SLaM pipelines:

 `source_parametric/no_lens_light`
 `mass_total/no_lens_light`

Check them out for a full description of the analysis!
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

sys.path.insert(0,os.getcwd())
import slam

"""
__Dataset + Masking__ 

Load the `Interferometer` data, define the visibility and real-space masks and plot them.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "interferometer", dataset_name)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
)

real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

visibilities_mask = np.full(fill_value=False, shape=interferometer.visibilities.shape)

settings_masked_interferometer = al.SettingsMaskedInterferometer(
    transformer_class=al.TransformerNUFFT
)

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    visibilities_mask=visibilities_mask,
    real_space_mask=real_space_mask,
    settings=settings_masked_interferometer,
)

interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.subplot_interferometer()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("interferometer", "slam", "mass_total__source_parametric")

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
 
 - Uses a parametric `EllipticalSersic` bulge for the source's light (omitting a disk / envelope).
 - Uses an `EllipticalIsothermal` model for the lens's total mass distribution with an `ExternalShear`.

 We use the following optional settings:
 
 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
"""
analysis = al.AnalysisInterferometer(dataset=masked_interferometer)

source_parametric_results = slam.source_parametric.no_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    mass=af.Model(al.mp.EllipticalIsothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllipticalSersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)


"""
__MASS TOTAL PIPELINE (no lens light)__

The MASS TOTAL PIPELINE (no lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIPELINE to initialize the model priors. In this example it:

 - Uses an `EllipticalPowerLaw` model for the lens's total mass distribution [priors initialized from SOURCE 
 PARAMETRIC PIPELINE + The centre if unfixed from (0.0, 0.0)].
 
 - Uses the `EllipticalSersic` model representing a bulge for the source's light [priors initialized from SOURCE 
 PARAMETRIC PIPELINE].
 
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS TOTAL PIPELINE.
"""
analysis = al.AnalysisInterferometer(dataset=masked_interferometer)

mass_results = slam.mass_total.no_lens_light(
    path_prefix=path_prefix,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_parametric_results,
    mass=af.Model(al.mp.EllipticalPowerLaw),
)

"""
Finish.
"""
