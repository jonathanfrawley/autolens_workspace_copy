from os import path
import autofit as af
import autolens as al

"""
This pipeline performs a source `Inversion` analysis which fits an image with lens mass model and source galaxy.

searches 1 & 2 use a magnification based `Pixelization` and constant `Regularization` scheme to reconstruct the source
(as opposed to immediately using the `Pixelization` & `Regularization` input via the pipeline slam). This ensures
that if the input `Pixelization` or `Regularization` scheme use hyper-images, they are initialized using
a pixelized source-plane, which is key for lens's with multiple or irregular sources.

The pipeline uses 4 searches:

Search 1:

    Fit the inversion`s `Pixelization` and `Regularization`, using a magnification
    based pixel-grid and the previous lens mass model.

    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: source/parametric/mass_sie__source_parametric.py
    Prior Passing: Lens Mass (instance -> previous pipeline).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Search 2:

    Refine the lens mass model using the source `Inversion`.
    
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: source/parametric/mass_sie__source_parametric.py
    Prior Passing: Lens Mass (model -> previous pipeline), source `Inversion` (instance -> search 1).
    Notes: Lens mass varies, source `Inversion` parameters fixed.

Search 3:

    Fit the inversion`s `Pixelization` and `Regularization`, using the input pixelization,
    `Regularization` and the previous lens mass model.
    
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: slam.source.pixelization + slam.source.regularization
    Previous Pipelines: None
    Prior Passing: Lens Mass (instance -> search 2).
    Notes:  Lens mass fixed, source `Inversion` parameters vary.

Search 4:

    Refine the lens mass model using the `Inversion`.
    
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: slam.source.pixelization + slam.source.regularization
    Previous Pipelines: source/parametric/mass_sie__source_parametric.py
    Prior Passing: Lens Mass (model -> search 2), source `Inversion` (instance -> search 3).
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(slam, settings, real_space_mask, source_parametric_results):

    pipeline_name = "pipeline_source[inversion]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
        3) The `Pixelization` and `Regularization` scheme of the pipeline (fitted in searches 3 & 4).
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix, pipeline_name, slam.source_inversion_tag
    )

    """
    Search 1: Fit the `Pixelization` and `Regularization`, where we:

        1) Fix the lens mass model to the `MassProfile`'s inferred by the previous pipeline.
    """

    phase1 = al.PhaseInterferometer(
        search=af.DynestyStatic(
            name="phase[1]_mass[fixed]_source[inversion_magnification_initialization]",
            n_live_points=30,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=af.last.instance.galaxies.lens.mass,
                shear=af.last.instance.galaxies.lens.shear,
                hyper_galaxy=slam.setup_hyper.hyper_galaxy_lens_from_result(
                    result=source_parametric_results.last
                ),
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
                hyper_galaxy=slam.setup_hyper.hyper_galaxy_source_from_result(
                    result=source_parametric_results.last
                ),
            ),
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=source_parametric_results.last
        ),
        settings=settings,
        real_space_mask=real_space_mask,
    )

    """
    Search 2: Fit the lens's mass and source galaxy using the magnification `Inversion`, where we:

        1) Fix the source `Inversion` parameters to the results of search 1.
        2) Set priors on the lens galaxy `MassProfile`'s using the results of the previous pipeline.
    """

    phase2 = al.PhaseInterferometer(
        search=af.DynestyStatic(
            name="phase[2]_mass[total]_source[fixed]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=af.last[-1].model.galaxies.lens.mass,
                shear=af.last[-1].model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase1.result.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_background_noise=phase1.result.instance.hyper_background_noise,
        settings=settings,
        real_space_mask=real_space_mask,
        use_as_hyper_dataset=True,
    )

    """
    Search 3: fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens `MassProfile` to the result of search 2.
    """

    phase3 = al.PhaseInterferometer(
        search=af.DynestyStatic(
            name="phase[3]_mass[fixed]_source[inversion_initialization]",
            n_live_points=30,
            evidence_tolerance=slam.setup_hyper.evidence_tolerance,
            sample="rstagger",
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=phase2.result.instance.galaxies.lens.mass,
                shear=phase2.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=slam.pipeline_source_inversion.setup_source.pixelization_prior_model,
                regularization=slam.pipeline_source_inversion.setup_source.regularization_prior_model,
                hyper_galaxy=phase2.result.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_background_noise=phase2.result.instance.hyper_background_noise,
        settings=settings,
        real_space_mask=real_space_mask,
        use_as_hyper_dataset=True,
    )

    """
    Search 4: fit the lens's mass using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Inversion` parameters to the results of search 3.
        2) Set priors on the lens galaxy `MassProfile`'s using the results of search 2.
    """

    mass = slam.pipeline_source_parametric.setup_mass.mass_prior_model_with_updated_priors_from_result(
        result=phase2.result, unfix_mass_centre=True
    )

    phase4 = al.PhaseInterferometer(
        search=af.DynestyStatic(
            name="phase[4]_mass[total]_source[fixed]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=phase3.result.instance.galaxies.source.pixelization,
                regularization=phase3.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase3.result.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_background_noise=phase3.result.instance.hyper_background_noise,
        settings=settings,
        real_space_mask=real_space_mask,
    )

    phase4 = phase4.extend_with_hyper_phase(setup_hyper=slam.setup_hyper)

    return al.PipelineDataset(
        pipeline_name,
        path_prefix,
        source_parametric_results,
        phase1,
        phase2,
        phase3,
        phase4,
    )
