from os import path
import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is modeled parametrically using one or more input `LightProfile`s.
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` and `ExternalShear`.
 - The source galaxy's light is modeled parametrically using one or more input `LightProfile`s.

The pipeline is three searches:

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

    Fit the lens mass with an `EllipticalIsothermal` (and optional shear) and source with the parametric profiles input
    into `SetupSourceParametric` (e.g. the `bulge_prior_model`, `disk_prior_model`, etc). The default is :
    
    - `SetupSourceParametric.bulge_prior_model=EllipticalSersic`, 
    - `SetupSourceParametric.disk_prior_model=EllipticalExponential`
    - `SetupSourceParametric.align_bulge_disk_centre=True` (meaning the two profiles above have aligned centre.
    
    Lens Light: Parametric model of search 1.
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: SetupSourceParametric.bulge_prior_model + SetupSourceParametric.disk_prior_model + others
    Prior Passing: Lens Light (instance -> search 1).
    Notes: Uses the lens subtracted image from search 1.

Search 3:

    Fit the `SetupMassTotal.mass_prior_model` (default=`EllipticalPowerLaw`) model, using priors from the  
    `EllipticalIsothermal` mass model of search 1 and the parametric lens light and source model with priors from 
    searches 1 & 2.
    
    Lens Light: SetupLightParametric.bulge_prior_model + SetupLightParametric.disk_prior_model + others
    Lens Mass: SetupMassTotal.mass_prior_model + ExternalShear
    Source Light: SetupSourceParametric.bulge_prior_model + SetupSourceParametric.disk_prior_model + others
    Prior Passing: Lens light (model -> search 1), lens mass and source light (model -> search 2).
    Notes: None
"""


def make_pipeline(setup, settings):

    pipeline_name = "pipeline_light[parametric]_mass[total]_source[parametric]"

    """
    This pipeline is tagged according to whether:

        1) The bulge model, disk model, etc. and whether their centres / elliptical_comps are aligned.
        2) The lens galaxy mass model includes an `ExternalShear`.
    """

    path_prefix = path.join(setup.path_prefix, pipeline_name, setup.tag)

    """
    Search 1: Fit only the lens galaxy's light, where we:

        1) Use the light model determined from `SetupLightParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.).
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

        1) Fix the foreground lens light subtraction to the lens galaxy light model from search 1.
        2) Use the source model determined from `SetupSourceParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.)
    """

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_light[fixed]_mass[sie]_source[parametric]", n_live_points=60
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
                al.Galaxy,
                redshift=setup.redshift_source,
                bulge=setup.setup_source.bulge_prior_model,
                disk=setup.setup_source.disk_prior_model,
                envelope=setup.setup_source.envelope_prior_model,
            ),
        ),
        settings=settings,
    )

    """
    Search 3: Fit simultaneously the lens and source galaxies, where we:

        1) Set the lens's light, mass, and source's light using the results of searches 1 and 2.
    """

    mass = setup.setup_mass.mass_prior_model_with_updated_priors_from_result(
        result=phase2.result
    )

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_light[parametric]_mass[total]_source[parametric]",
            n_live_points=100,
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=phase1.result.model.galaxies.lens.bulge,
                disk=phase1.result.model.galaxies.lens.disk,
                envelope=phase1.result.model.galaxies.lens.envelope,
                mass=mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=setup.redshift_source,
                bulge=phase2.result.model.galaxies.source.bulge,
                disk=phase2.result.model.galaxies.source.disk,
                envelope=phase2.result.model.galaxies.source.envelope,
            ),
        ),
        settings=settings,
    )

    return al.PipelineDataset(pipeline_name, path_prefix, None, phase1, phase2, phase3)
