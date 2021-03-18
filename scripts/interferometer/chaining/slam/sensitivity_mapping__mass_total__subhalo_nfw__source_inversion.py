"""
SLaM (Source, Light and Mass): Mass Total + Subhalo NFW + Source Inversion Sensitivity Mapping
==============================================================================================

Using 1 source pipeline, a mass pipeline and a subhalo pipeline this SLaM runner fits `Interferometer` of a strong
lens system, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal`.
 - A dark matter subhalo near the lens galaxy is included as a`SphericalNFWMCRLudLow`.
 - The source galaxy is an `EllipticalSersic`.

This runner uses the SLaM pipelines:

 `slam/pipelines/source__mass_sie__source_parametric.py`.
 `slam/pipelines/source__mass_sie__source_inversion.py`.
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

"""
__Dataset__ 

Load the `Interferometer` data, define the visibility and real-space masks and plot them.
"""
dataset_name = "mass_sie__subhalo_nfw__source_sersic"
pixel_scales = 0.05

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

interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.subplot_interferometer()

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

"""
We also specify the *SettingsInversion*, which describes how the `Inversion` fits the source `Pixelization` and 
with `Regularization`. 

This can perform the linear algebra calculation that performs the `Inversion` using two options: 

 - As matrices: this is numerically more accurate and does not approximate the `log_evidence` of the `Inversion`. For
  datasets of < 100 0000 visibilities we recommend that you use this option. However, for > 100 000 visibilities this
  approach requires excessive amounts of memory on your computer (> 16 GB) and thus becomes unfeasible. 

 - As linear operators: this numerically less accurate and approximates the `log_evidence` of the `Inversioon`. However,
 it is the only viable options for large visibility datasets. It does not represent the linear algebra as matrices in
 memory and thus makes the analysis of > 10 million visibilities feasible.

By default we use the linear operators approach.  
"""
settings_inversion = al.SettingsInversion(use_linear_operators=True)

settings = al.SettingsPhaseInterferometer(
    settings_masked_interferometer=settings_masked_interferometer,
    settings_inversion=settings_inversion,
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
subsequent `SLaMPipelineLightParametric` and Mass pipelines. The assumptions regarding the lens light chosen by 
the `Light` object are carried forward to the `Mass`  pipeline.

The `Setup` again tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same `SLaMPipelineSource`. they will reuse those results before branching off to fit different models in the 
`SLaMPipelineLightParametric` and / or `SLaMPipelineMass` pipelines. 

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
    mass_prior_model=al.mp.EllipticalIsothermal, with_shear=True
)
setup_source = al.SetupSourceParametric()

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_mass=setup_mass, setup_source=setup_source
)

"""
__SLaMPipelineSourceInversion__

The Source inversion pipeline aims to initialize a robust model for the source galaxy using an `Inversion`.

_SLaMPipelineSourceInversion_ determines the `Inversion` used by the inversion source pipeline. A full description of all 
options can be found ? and ?.

By default, this again assumes `EllipticalIsothermal` profile for the lens galaxy's mass model.

For this runner the `SLaMPipelineSourceInversion` customizes:

 - The `Pixelization` used by the `Inversion` of this pipeline.
 - The `Regularization` scheme used by the `Inversion` of this pipeline.

The `SLaMPipelineSourceInversion` use`s the `SetupMass` of the `SLaMPipelineSourceParametric`.

The `SLaMPipelineSourceInversion` determines the source model used in the `SLaMPipelineLightParametric` 
and `SLaMPipelineMass` pipelines, which in this example therefore both use an `Inversion`.
"""
setup_source = al.SetupSourceInversion(
    pixelization_prior_model=al.pix.VoronoiBrightnessImage,
    regularization_prior_model=al.reg.AdaptiveBrightness,
)

pipeline_source_inversion = al.SLaMPipelineSourceInversion(setup_source=setup_source)

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
searchesn. 

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
    path_prefix=path.join("fits/slam", dataset_name),
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_source_inversion=pipeline_source_inversion,
    pipeline_mass=pipeline_mass,
    setup_subhalo=setup_subhalo,
)

"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then run each pipeline, passing the results of previous pipelines to subsequent pipelines.
"""
import source__parametric
import source__inversion
import mass__total
import subhalo

source__parametric = source__parametric.make_pipeline(
    slam=slam, settings=settings, real_space_mask=real_space_mask
)
source_results = source__parametric.run(dataset=interferometer, mask=visibilities_mask)

source__inversion = source__inversion.make_pipeline(
    slam=slam,
    settings=settings,
    real_space_mask=real_space_mask,
    source_parametric_results=source_results,
)
source_results = source__inversion.run(dataset=interferometer, mask=visibilities_mask)

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
            masked_interferometer=masked_interferometer,
            settings=settings,
            cosmology=cosmo.Planck15,
        )

        self.hyper_galaxy_image_path_dict = (
            mass_results.last.hyper_galaxy_image_path_dict
        )
        self.hyper_model_image = mass_results.last.hyper_model_image


subhalo = subhalo.sensitivity_mapping(
    slam=slam,
    uv_wavelengths=interferometer.uv_wavelengths,
    visibilities_mask=visibilities_mask,
    real_space_mask=real_space_mask,
    mass_results=mass_results,
    analysis_cls=Analysis,
)
subhalo.run()

"""
Finish.
"""
