from os import path
import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an input total `MassProfile` (default=`EllipticalPowerLaw`).
 - The source galaxy's surface-brightness is an `Inversion`.
.
The pipeline is four searches:

Search 1:

    Fit the lens mass with an `EllipticalIsothermal` (and optional shear) and source bulge with an `EllipticalSersic`.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None.
    Notes: None.

Search 2:

    Fit the source `Inversion` using the lens `EllipticalIsothermal` (and optional shear) inferred in search 1. The
    `Pixelization` uses `SetupSourceInversion.pixelization_prior_model` (default=`Rectangular`) and 
    `Regulaization` uses `SetupSourceInversion.regularization_prior_model` (default=`Constant`).
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: SetupSourceInversion.pixelization_prior_model + SetupSourceInversion.regularization_prior_model
    Prior Passing: Lens Mass (instance -> phase1).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Search 3:

    Refines the lens `EllipticalIsothermal` mass models using the source `Inversion` of search 2.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: `Pixelization` (default=VoronoiMagnification) + `Regularization` (default=Constant)
    Prior Passing: Lens Mass (model -> search 1), Source `Inversion` (instance -> search 2)
    Notes: Lens mass varies, source `Inversion` parameters fixed.

Search 4:

    Fit the `SetupMassTotal.mass_prior_model` (default=`EllipticalPowerLaw`) model, using priors from the  
    `EllipticalIsothermal` mass model of search 3 and the source `Inversion` of search 2.
    
    Lens Mass: SetupMassTotal.mass_prior_model + ExternalShear
    Source Light: SetupSourceInversion.pixelization_prior_model + SetupSourceInversion.regularization_prior_model
    Prior Passing: Lens Mass (model -> search 3), Source `Inversion` (instance -> search 3)
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings):

    pipeline_name = "pipeline_mass[total]_source[inversion]"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an `ExternalShear`.
        2) The `Pixelization` and `Regularization` scheme of the pipeline (fitted in searches 3 & 4).
    """

    path_prefix = path.join(setup.path_prefix, pipeline_name, setup.tag)

    """
    Search 1: Fit the lens's `MassProfile`'s and source `LightProfile`, where we:

        1) Use an `EllipticalIsothermal` for the lens's mass and `EllipticalSersic`for the source's bulge, 
           irrespective of the final model that is fitted by the pipeline.
        2) include an `ExternalShear` in the mass model if `SetupMass.with_shear=True`.
    """

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_mass[sie]_source[bulge]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=al.mp.EllipticalIsothermal,
                shear=setup.setup_mass.shear_prior_model,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source, bulge=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
    )

    """
    Search 2: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens's `MassProfile`'s to the results of search 1.
        2) Use the `Pixelization` input into `SetupSourceInversion.pixelization_prior_model`.
        3) Use the `Regularization` input into `SetupSourceInversion.regularization_prior_model`.
    """

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_mass[sie]_source[inversion_initialization]", n_live_points=20
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=setup.setup_source.pixelization_prior_model,
                regularization=setup.setup_source.regularization_prior_model,
            ),
        ),
        settings=settings,
    )

    """
    Search 3: Refit the lens's mass (and shear) using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Pixelization` and `Regularization` to the results of search 2.
        2) Set priors on the lens galaxy `MassProfile`'s using the results of search 1.
    """

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_mass[sie]_source[inversion]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
    )

    """
    We now `extend` search 1 with an additional `hyper phase` which uses the maximum log likelihood mass model of 
    search 1 above to refine the `Inversion`, by fitting only the parameters of the `Pixelization` and `Regularization`
    (in this case, the shape of the `VoronoiMagnification` and `Regularization` coefficient of the `Constant`.

    The `hyper` phase results are accessible as attributes of the phase results and used in search 3 below.
    """
    phase3 = phase3.extend_with_hyper_phase(setup_hyper=al.SetupHyper())

    """
    Search 4: Fit the lens's `MassProfile`'s with the input `SetupMassTotal.mass_prior_model` using the source 
    `Inversion` of phase above, where we:

        1) Use the source `Pixelization` and `Regularization inferred in search 3`s extended `inversion_phase`.
        2) Set priors on the lens galaxy mass using the `EllipticalIsothermal` (and `ExternalShear`) of search 3.
    """

    """
    The method below passes priors for the `mass` from the fit above, irrespective of what `MassProfile` is used.
    """

    mass = setup.setup_mass.mass_prior_model_with_updated_priors_from_result(
        result=phase3.result
    )

    phase4 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[4]_mass[total]_source[inversion]", n_live_points=100
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=phase3.result.hyper.instance.galaxies.source.pixelization,
                regularization=phase3.result.hyper.instance.galaxies.source.regularization,
            ),
        ),
        hyper_image_sky=af.last.hyper.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    return al.PipelineDataset(
        pipeline_name, path_prefix, None, phase1, phase2, phase3, phase4
    )
