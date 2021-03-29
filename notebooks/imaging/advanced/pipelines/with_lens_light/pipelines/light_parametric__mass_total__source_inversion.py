from os import path
import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is a parametric `EllipticalSersic` and `EllipticalExponential`.
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is an `Inversion`.

The pipeline is five searches:

Search 1:

    Fit and subtract the lens light with the parametric profiles input into `SetupLightParametric` (e.g. the 
    `bulge_prior_model`, `disk_prior_model`, etc). The default is :
    
    - `SetupLightParametric.bulge_prior_model=EllipticalSersic`, 
    - `SetupLightParametric.disk_prior_model=EllipticalExponential`
    - `SetupLightParametric.align_bulge_disk_centre=True` (meaning the two profiles above have aligned centre.
    
    Lens Light: SetupLightParametric.bulge_prior_model + SetupLightParametric.disk_prior_model + others
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Search 2:

    Fit the source `Inversion` using the lens `EllipticalIsothermal` (and optional shear) inferred in search 1. The
    `Pixelization` uses `SetupSourceInversion.pixelization_prior_model` (default=`Rectangular`) and 
    `Regulaization` uses `SetupSourceInversion.regularization_prior_model` (default=`Constant`).
    
    Lens Light: SetupLightParametric.bulge_prior_model + SetupLightParametric.disk_prior_model + others
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens Light (instance -> search 1).
    Notes: Uses the lens subtracted image from search 1.

Search 3:

    Refine the lens light and mass models and source light model using priors initialized from searches 1 and 2.
    
    Lens Light: SetupLightParametric.bulge_prior_model + SetupLightParametric.disk_prior_model + others
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> search 1), lens mass and source light (model -> search 2).
    Notes: None

Search 4:

    Fit the source `Inversion` using the lens `EllipticalIsothermal` (and optional shear) inferred in search 1. The
    `Pixelization` uses `SetupSourceInversion.pixelization_prior_model` (default=`Rectangular`) and 
    `Regulaization` uses `SetupSourceInversion.regularization_prior_model` (default=`Constant`).
    
    Lens Light: SetupLightParametric.bulge_prior_model + SetupLightParametric.disk_prior_model + others
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: SetupSourceInversion.pixelization_prior_model + SetupSourceInversion.regularization_prior_model
    Prior Passing: Lens Light & Mass (instance -> phase3).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Search 5:

    Fit the `SetupMassTotal.mass_prior_model` (default=`EllipticalPowerLaw`) model, using priors from the  
    `EllipticalIsothermal` mass model of search 3 and the source `Inversion` of search 2.
    
    Lens Light: SetupLightParametric.bulge_prior_model + SetupLightParametric.disk_prior_model + others
    Lens Mass: SetupMassTotal.mass_prior_model + ExternalShear
    Source Light: SetupSourceInversion.pixelization_prior_model + SetupSourceInversion.regularization_prior_model
    Prior Passing: Lens Light & Mass (model -> search 3), Source `Inversion` (instance -> search 4)
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings):

    pipeline_name = "pipeline_light[parametric]_mass[total]_source[inversion]"

    """
    This pipeline is tagged according to whether:

        1) The bulge model, disk model, etc. and whether their centres / elliptical_comps are aligned.
        2) The lens galaxy mass model includes an `ExternalShear`.
    """

    path_prefix = path.join(setup.path_prefix, pipeline_name, setup.tag)

    """
    Search 1; Fit only the lens galaxy's light, where we:

        1) Use the light model determined from `SetupLightParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.)
    """
    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(name="phase[1]_light[parametric]", n_live_points=50),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=setup.setup_light.bulge_prior_model,
                disk=setup.setup_light.disk_prior_model,
                envelope=setup.setup_light.envelope_prior_model,
            )
        ),
        settings=settings,
    )

    """
    Search 2: Fit the lens's `MassProfile`'s and source galaxy's light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy bulge+disk model from search 1.
    """

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_light[fixed]_mass[sie]_source[bulge]", n_live_points=60
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=phase1.result.instance.galaxies.lens.bulge,
                disk=phase1.result.instance.galaxies.lens.disk,
                envelope=phase1.result.instance.galaxies.lens.envelope,
                mass=al.mp.EllipticalIsothermal,
                shear=setup.setup_mass.shear_prior_model,
            ),
            source=af.Model(
                al.Galaxy, redshift=setup.redshift_source, bulge=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
    )

    """
    Search 3: Fit simultaneously the lens and source galaxies, where we:

        1) Set the lens's bulge, disk, mass, and source model and priors using the results of searches 1 and 2.
    """

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_light[parametric]_mass[sie]_source[bulge]", n_live_points=100
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=phase1.result.model.galaxies.lens.bulge,
                disk=phase1.result.model.galaxies.lens.disk,
                envelope=phase1.result.model.galaxies.lens.envelope,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=setup.redshift_source,
                bulge=phase2.result.model.galaxies.source.bulge,
            ),
        ),
        settings=settings,
    )

    """
    Search 4: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens's bulge, disk and mass model to the results of search 3.
    """

    phase4 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[4]_light[fixed]_mass[fixed]_source[inversion_initialization]",
            n_live_points=20,
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=phase3.result.instance.galaxies.lens.bulge,
                disk=phase3.result.instance.galaxies.lens.disk,
                envelope=phase3.result.instance.galaxies.lens.envelope,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.mass,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=setup.redshift_source,
                pixelization=setup.setup_source.pixelization_prior_model,
                regularization=setup.setup_source.regularization_prior_model,
            ),
        ),
        settings=settings,
    )

    """
    Search 5: Fit the lens's bulge, disk and mass using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Inversion` parameters to the results of search 4.
        2) Set priors on the lens galaxy bulge, disk and mass using the results of search 3.
    """

    """
    If the `mass_prior_model` is an `EllipticalPowerLaw` `MassProfile` we can initialize its priors from the 
    `EllipticalIsothermal` fitted previously. If it is not an `EllipticalPowerLaw` we omit this setting up of
    priors, still benefitting from the initialized `Inversion` parameters..
    """

    mass = setup.setup_mass.mass_prior_model_with_updated_priors_from_result(
        result=phase3.result
    )

    phase5 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[5]_light[parametric]_mass[total]_source[inversion]",
            n_live_points=100,
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=phase3.result.model.galaxies.lens.bulge,
                disk=phase3.result.model.galaxies.lens.disk,
                envelope=phase3.result.model.galaxies.lens.envelope,
                mass=mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=setup.redshift_source,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
    )

    return al.PipelineDataset(
        pipeline_name, path_prefix, None, phase1, phase2, phase3, phase4, phase5
    )
