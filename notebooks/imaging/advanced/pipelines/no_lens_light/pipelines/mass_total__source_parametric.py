from os import path
import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an input total `MassProfile` (default=`EllipticalPowerLaw`).
 - The source galaxy's is modeled parametrically using one or more input `LightProfile`s.
.
The pipeline is four searches:

Search 1:

Fit the lens mass with an `EllipticalIsothermal` (and optional shear) and source with the parametric profiles input
into `SetupSourceParametric` (e.g. the `bulge_prior_model`, `disk_prior_model`, etc). The default is :
    
- `SetupSourceParametric.bulge_prior_model=EllipticalSersic`, 
- `SetupSourceParametric.disk_prior_model=EllipticalExponential`
- `SetupSourceParametric.align_bulge_disk_centre=True` (meaning the two profiles above have aligned centres).
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: SetupSourceParametric.bulge_prior_model + SetupSourceParametric.disk_prior_model + others
    Prior Passing: None.
    Notes: The parametric source model depends on inputs in SetupSourceParametric such as `align_bulge_disk_centre`.

Search 2:

Fit the `SetupMassTotal.mass_prior_model` (default=`EllipticalPowerLaw`) model, using priors from the  
`EllipticalIsothermal` mass model of search 1 and the parametric source model with priors from phase `.
    
    Lens Mass: SetupMassTotal.mass_prior_model + ExternalShear
    Source Light: SetupSourceParametric.bulge_prior_model + SetupSourceParametric.disk_prior_model + others
    Prior Passing: Lens Mass (model -> search 1), Source `LightProfile`'s (model -> search 1)
    Notes: All parameters free and vary
"""


def make_pipeline(setup, settings):

    pipeline_name = "pipeline_mass[total]_source[parametric]"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an `ExternalShear`.
        2) `Model`'s that make up the `bulge`, `disk`, etc in the `SetupSourceParametric` as well as options that
           customize this model, like the alignement of centres, etc.
    """

    path_prefix = path.join(setup.path_prefix, pipeline_name, setup.tag)

    """
    Search 1: Fit the lens's `MassProfile`'s and source `LightProfile`, where we:

        1) Use an `EllipticalIsothermal` for the lens's mass irrespective of the final mass model that is fitted by 
           the pipeline.
        2) Use the source model determined from `SetupSourceParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.)
        3) include an `ExternalShear` in the mass model if `SetupMass.with_shear=True`.
   """

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_mass[sie]_source[parametric]", n_live_points=50
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
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
    Search 2: Fit the lens's `MassProfile`'s with the input `SetupMassTotal.mass_prior_model` using the parametric 
    source  model of the phase above, where we:

        1) Use the results of search 1 to initialize priors on the source `LightProfile`'s (`bulge`, `disk`, etc.).
        2) Set priors on the lens galaxy mass using the `EllipticalIsothermal` (and `ExternalShear`) of search 1.
    """

    """
    The method below passes priors for the `mass` from the fit above, irrespective of what `MassProfile` is used.
    """

    mass = setup.setup_mass.mass_prior_model_with_updated_priors_from_result(
        result=phase1.result
    )

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_mass[total]_source[parametric]", n_live_points=20
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                mass=mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=setup.redshift_source,
                bulge=phase1.result.model.galaxies.source.bulge,
                disk=phase1.result.model.galaxies.source.disk,
                envelope=phase1.result.model.galaxies.source.envelope,
            ),
        ),
        settings=settings,
    )

    return al.PipelineDataset(pipeline_name, path_prefix, None, phase1, phase2)
