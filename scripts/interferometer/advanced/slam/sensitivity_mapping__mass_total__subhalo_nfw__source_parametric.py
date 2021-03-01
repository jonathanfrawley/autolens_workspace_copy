"""
SLaM (Source, Light and Mass): Mass Total + Subhalo NFW + Source Parametric Sensitivity Mapping
===============================================================================================

Using 1 source pipeline, a mass pipeline and a subhalo pipeline this SLaM runner fits `Interferometer` of a strong
lens system, where in the final phase of the pipeline:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is modeled as an `EllipticalIsothermal`.
 - A dark matter subhalo`s within the lens galaxy is modeled as a `SphericalNFWMCRLudLow`.
 - The source galaxy is modeled as an `EllipticalSersic`.

This runner uses the SLaM pipelines:

 `slam/pipelines/source__mass_sie__source_parametric.py`.
 `slam/pipelines/mass__mass_power_law__source.py`.
 `slam/pipelines/subhalo__mass__subhalo_nfw__source.py`.

Check them out for a detailed description of the analysis!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
from astropy import cosmology as cosmo
from autolens.pipeline.phase.interferometer import analysis as a
import autolens as al
import autolens.plot as aplt
import numpy as np

dataset_name = "mass_sie__subhalo_nfw__source_sersic"
pixel_scales = 0.05

dataset_path = path.join("dataset", "interferometer", dataset_name)

"""
Using the dataset path, load the data (image, noise-map, PSF) as an `Interferometer` object from .fits files.
"""
interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
)

interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.subplot_interferometer()

"""
The perform a fit, we need two masks, firstly a ‘real-space mask’ which defines the grid the image of the lensed 
source galaxy is evaluated using.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

"""
We also need a ‘visibilities mask’ which defining which visibilities are omitted from the chi-squared evaluation.
"""
visibilities_mask = np.full(fill_value=False, shape=interferometer.visibilities.shape)

"""
__Settings__

The `SettingsPhaseInterferometer` describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the `autolens_workspace/notebooks/interferometer/modeling` example 
scripts, with a complete description of all settings given 
in `autolens_workspace/notebooks/interferometer/modeling/customize/settings.py`.

The settings chosen here are applied to all phases in the pipeline.
"""
settings_masked_interferometer = al.SettingsMaskedInterferometer(
    grid_class=al.Grid2D, sub_size=2
)

settings = al.SettingsPhaseInterferometer(
    settings_masked_interferometer=settings_masked_interferometer
)

"""
__PIPELINE SETUP__

Pipelines use the `SetupPipeline` object to customize the analysis performed by the pipeline,
for example if a shear was included in the mass model and the model used for the source galaxy.

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong 
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own setup object 
which is equivalent to the `SetupPipeline` object, customizing the analysis in that pipeline. Each pipeline therefore
has its own `SetupMass`, `SetupLightParametric` and `SetupSourceParametric` object.

The `Setup` used in earlier pipelines determine the model used in later pipelines. For example, if the `Source` 
pipeline is given a `Pixelization` and `Regularization`, than this `Inversion` will be used in the 
subsequent `SLaMPipelineLightParametric` and  Mass pipelines. The assumptions regarding the lens light chosen by 
the `Light` object are carried forward to the  `Mass`  pipeline.

The `Setup` again tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same `SLaMPipelineSource`. they will reuse those results before branching off to fit different models in 
the `SLaMPipelineLightParametric` and / or `SLaMPipelineMass` pipelines. 

__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit and is used identically to the
hyper pipeline examples.

The `SetupHyper` object has a new input available, `hyper_fixed_after_source`, which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
_SLaMPipelineLight_ and `SLaMPipelineMass` pipelines, model comparison can be performed in a consistent fashion.
"""
hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__SLaMPipelineSourceParametric__

The parametric source pipeline aims to initialize a robust model for the source galaxy using `LightProfile` objects. 

_SLaMPipelineSourceParametric_ determines the source model used by the parametric source pipeline. A full description 
of all options can be found ? and ?.

By default, this assumes an `EllipticalIsothermal` profile for the lens galaxy's mass. Our experience with lens 
modeling has shown they are the simpliest models that provide a good fit to the majority of strong lenses.

For this runner the `SLaMPipelineSourceParametric` customizes:

 - The `MassProfile` fitted by the pipeline (and the following `SLaMPipelineSourceInversion`.
 - If there is an `ExternalShear` in the mass model or not.
"""
setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalIsothermal, with_shear=True, mass_centre=(0.0, 0.0)
)
setup_source = al.SetupSourceParametric(disk_prior_model=al.lp.EllipticalExponential)

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_mass=setup_mass, setup_source=setup_source
)

"""
__SLaMPipelineMassTotal__

The `SLaMPipelineMassTotal` pipeline fits the model for the lens galaxy's total mass distribution. 

A full description of all options can be found ? and ?.

The model used to represent the lens galaxy's mass is input into `SLaMPipelineMass` and this runner uses an 
`EllipticalIsothermal` in this example.

For this runner the `SLaMPipelineMass` customizes:

 - The `MassProfile` fitted by the pipeline.
 - If there is an `ExternalShear` in the mass model or not.
"""
setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalIsothermal, with_shear=True
)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

"""
__SetupSubhalo__

The final pipeline fits the lens and source model including a `SphericalNFW` subhalo, using a grid-search of non-linear
searches. 

A full description of all options can be found ? and ?.

The models used to represent the lens galaxy's mass and the source are those used in the previous pipelines.

For this runner the `SetupSubhalo` customizes:

 - If the source galaxy (parametric or _Inversion) is treated as a model (all free parameters) or instance (all fixed) 
   during the subhalo detection grid search.
 - The NxN size of the grid-search.
"""
setup_subhalo = al.SetupSubhalo(
    source_is_model=True, number_of_steps=5, number_of_cores=2
)

"""
__SLaM__

We combine all of the above `SLaM` pipelines into a `SLaM` object.

The `SLaM` object contains a number of methods used in the make_pipeline functions which are used to compose the model 
based on the input values. It also handles pipeline tagging and path structure.
"""
slam = al.SLaM(
    path_prefix=path.join("slam", dataset_name),
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_mass=pipeline_mass,
    setup_subhalo=setup_subhalo,
)

"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then run each pipeline, passing the results of previous pipelines to subsequent pipelines.
"""
from pipelines import source__parametric
from pipelines import mass__total
from pipelines import subhalo

source__parametric = source__parametric.make_pipeline(
    slam=slam, settings=settings, real_space_mask=real_space_mask
)
source_results = source__parametric.run(dataset=interferometer, mask=visibilities_mask)

mass__total = mass__total.make_pipeline(
    slam=slam,
    settings=settings,
    real_space_mask=real_space_mask,
    source_results=source_results,
)
mass_results = mass__total.run(dataset=interferometer, mask=visibilities_mask)

"""
__Sensitivity Mapping__

Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
data simulated by the `simulate_function` for that model.

This requires us to write a wrapper around the PyAutoLens `Analysis` class.
"""
class Analysis(a.Analysis):
    def __init__(self, masked_interferometer):

        super().__init__(
            masked_interferometer=masked_interferometer, settings=settings, cosmology=cosmo.Planck15
        )

        self.hyper_galaxy_image_path_dict = mass_results.last.hyper_galaxy_image_path_dict
        self.hyper_model_image = mass_results.last.hyper_model_image

subhalo = subhalo.sensitivity_mapping(
    slam=slam,
    uv_wavelengths=interferometer.uv_wavelengths,
    visibilities_mask=visibilities_mask,
    real_space_mask=real_space_mask,
    mass_results=mass_results,
    analysis_cls=Analysis
)
subhalo.run()

"""
Finish.
"""
