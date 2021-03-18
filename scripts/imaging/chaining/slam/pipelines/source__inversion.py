from os import path
import autofit as af
import autolens as al
from . import slam_util


def source__inversion__no_lens_light(
    path_prefix: str,
    analysis,
    setup_hyper: al.SetupHyper,
    source_parametric_results,
    pixelization: af.PriorModel(al.pix.Pixelization) = af.PriorModel(
        al.pix.VoronoiBrightnessImage
    ),
    regularization: af.PriorModel(al.reg.Regularization) = af.PriorModel(
        al.reg.Constant
    ),
):
    """
    The SlaM SOURCE INVERSION PIPELINE for fitting imaging data without a lens light component.

    Parameters
    ----------
    path_prefix : str or None
        The prefix of folders between the output path and the search folders.
    analysis : al.AnalysisImaging
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    pixelization : af.PriorModel(pix.Pixelization)
        The pixelization used by the `Inversion` which fits the source light.
    regularization : af.PriorModel(reg.Regularization)
        The regularization used by the `Inversion` which fits the source light.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the SOURCE INVERSION PIPELINE we fit a lens model where:

     - The lens galaxy is modeled using a total mass distribution [parameters fixed to result of SOURCE PARAMETRIC PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme.

    This search aims to quickly estimate values for the pixelization resolution and regularization coefficient.
    """
    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=source_parametric_results.last.instance.galaxies.lens.redshift,
                mass=source_parametric_results.last.instance.galaxies.lens.mass,
                shear=source_parametric_results.last.instance.galaxies.lens.shear,
                hyper_galaxy=setup_hyper.hyper_galaxy_lens_from_result(
                    result=source_parametric_results.last
                ),
            ),
            source=al.GalaxyModel(
                redshift=source_parametric_results.last.instance.galaxies.source.redshift,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
                hyper_galaxy=setup_hyper.hyper_galaxy_source_from_result(
                    result=source_parametric_results.last
                ),
            ),
        ),
        hyper_image_sky=setup_hyper.hyper_image_sky_from_result(
            result=source_parametric_results.last, as_model=False
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from_result(
            result=source_parametric_results.last
        ),
    )

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="source_inversion[1]_mass[fixed]_source[inversion_magnification_initialization]",
        n_live_points=30,
    )

    result_1 = search.fit(model=model, analysis=analysis)

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    In search 2 of the SOURCE INVERSION PIPELINE we fit a lens model where:

     - The lens galaxy is modeled using a total mass distribution [parameters initialized from the results of the 
     SOURCE PARAMETRIC PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme 
     [parameters are fixed to the result of search 1].

    This search aims to improve the lens mass model using the search 1 `Inversion`.
    """
    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=result_1.instance.galaxies.lens.redshift,
                mass=source_parametric_results.last.model.galaxies.lens.mass,
                shear=source_parametric_results.last.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=result_1.instance.galaxies.source.redshift,
                pixelization=result_1.instance.galaxies.source.pixelization,
                regularization=result_1.instance.galaxies.source.regularization,
                hyper_galaxy=result_1.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_1.instance.hyper_image_sky,
        hyper_background_noise=result_1.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="source_inversion[2]_mass[total]_source[fixed]",
        n_live_points=50,
    )

    result_2 = search.fit(model=model, analysis=analysis)

    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    In search 3 of the SOURCE INVERSION PIPELINE we fit a lens model where:

     - The lens galaxy is modeled using a total mass distribution [parameters fixed to result of search 2].
     - The source galaxy's light is the input pixelization and regularization.

    This search aims to estimate values for the pixelization and regularization scheme.
    """

    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=result_2.instance.galaxies.lens.redshift,
                mass=result_2.instance.galaxies.lens.mass,
                shear=result_2.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=result_2.instance.galaxies.source.redshift,
                pixelization=pixelization,
                regularization=regularization,
                hyper_galaxy=result_2.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_2.instance.hyper_image_sky,
        hyper_background_noise=result_2.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="source_inversion[3]_mass[fixed]_source[inversion_initialization]",
        n_live_points=30,
        evidence_tolerance=setup_hyper.evidence_tolerance,
        sample="rstagger",
    )

    analysis.set_hyper_dataset(result=result_2)

    result_3 = search.fit(model=model, analysis=analysis)
    result_3.use_as_hyper_dataset = True

    """
    __Model + Search + Analysis + Model-Fit (Search 4)__

    In search 4 of the SOURCE INVERSION PIPELINE we fit a lens model where:

     - The lens galaxy is modeled using a total mass distribution [parameters initialized from the results of the 
     search 2].
     - The source galaxy's light is the input pixelization and regularization scheme [parameters fixed to the result 
     of search 3].

    This search aims to improve the lens mass model using the input `Inversion`.
    """
    mass = slam_util.mass__from_result(
        mass=result_2.model.galaxies.lens.mass,
        result=source_parametric_results.last,
        unfix_mass_centre=True,
    )

    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=result_3.instance.galaxies.lens.redshift,
                mass=mass,
                shear=result_2.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=result_3.instance.galaxies.source.redshift,
                pixelization=result_3.instance.galaxies.source.pixelization,
                regularization=result_3.instance.galaxies.source.regularization,
                hyper_galaxy=result_3.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_3.instance.hyper_image_sky,
        hyper_background_noise=result_3.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="source_inversion[4]_mass[total]_source[fixed]",
        n_live_points=50,
    )

    analysis.set_hyper_dataset(result=result_3)

    result_4 = search.fit(model=model, analysis=analysis)

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:

     - The source is using an `Inversion`.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky input`.
     - The background noise is included via the `hyper_background_noise`.
    """
    result_4 = al.util.model.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_4,
        search=search,
        analysis=analysis,
        include_hyper_image_sky=True,
    )

    results = af.ResultsCollection()
    results.add(search.paths.name, result_1)
    results.add(search.paths.name, result_2)
    results.add(search.paths.name, result_3)
    results.add(search.paths.name, result_4)

    return results


def source__inversion__with_lens_light(slam, settings, source_parametric_results):

    pipeline_name = "pipeline_source[inversion]"

    """
    This pipeline is tagged according to whether:
    
        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The `Pixelization` and `Regularization` scheme of the pipeline (fitted in phases 3 & 4).
        3) The lens galaxy mass model includes an  `ExternalShear`.
        4) The lens light model used in the previous pipeline.
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix, pipeline_name, slam.source_inversion_tag
    )

    """
    Search 1: fit the `Pixelization` and `Regularization`, where we:

        1) Fix the lens light & mass model to the `LightProile`'s and `MassProfile`'s inferred by the previous pipeline.
    """

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_light[fixed]_mass[fixed]_source[inversion_magnification_initialization]",
            n_live_points=20,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=source_parametric_results.last.instance.galaxies.lens.bulge,
                disk=source_parametric_results.last.instance.galaxies.lens.disk,
                mass=source_parametric_results.last.instance.galaxies.lens.mass,
                shear=source_parametric_results.last.instance.galaxies.lens.shear,
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
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=source_parametric_results.last, as_model=False
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=source_parametric_results.last
        ),
        settings=settings,
    )

    """
    Search 2: Fit the lens`s mass and source galaxy using the magnification `Inversion`, where we:

        1) Fix the source `Inversion` parameters to the results of search 1.
        2) Fix the lens light model to the results of the previous pipeline.
        3) Set priors on the lens galaxy mass from the previous pipeline.
    """

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_light[fixed]_mass[total]_source[inversion_magnification]",
            n_live_points=50,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=source_parametric_results.last.instance.galaxies.lens.bulge,
                disk=source_parametric_results.last.instance.galaxies.lens.disk,
                mass=source_parametric_results.last.model.galaxies.lens.mass,
                shear=source_parametric_results.last.model.galaxies.lens.shear,
                hyper_galaxy=phase1.result.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase1.result.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.instance.hyper_image_sky,
        hyper_background_noise=phase1.result.instance.hyper_background_noise,
        settings=settings,
        use_as_hyper_dataset=True,
    )

    """
    Search 3: Fit the input pipeline `Pixelization` & `Regularization`, where we:
    
        1) Fix the lens `LightPofile`'s to the results of the previous pipeline.
        2) Fix the lens `MassProfile` to the result of search 2.
    """

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_light[fixed]_mass[fixed]_source[inversion_initialization]",
            n_live_points=30,
            evidence_tolerance=slam.setup_hyper.evidence_tolerance,
            sample="rstagger",
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=phase2.result.instance.galaxies.lens.bulge,
                disk=phase2.result.instance.galaxies.lens.disk,
                mass=phase2.result.instance.galaxies.lens.mass,
                shear=phase2.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase2.result.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=slam.pipeline_source_inversion.setup_source.pixelization_prior_model,
                regularization=slam.pipeline_source_inversion.setup_source.regularization_prior_model,
                hyper_galaxy=phase2.result.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.instance.hyper_image_sky,
        hyper_background_noise=phase2.result.instance.hyper_background_noise,
        settings=settings,
        use_as_hyper_dataset=True,
    )

    """
    Search 4: Fit the lens`s mass using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Inversion` parameters to the results of search 3.
        2) Fix the lens `LightProfile`'s to the results of the previous pipeline.
        3) Set priors on the lens galaxy `MassProfile`'s using the results of search 2.
    """

    mass = slam.pipeline_source_parametric.setup_mass.mass_prior_model_with_updated_priors_from_result(
        result=source_parametric_results.last, unfix_mass_centre=True
    )

    phase4 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[4]_light[fixed]_mass[total]_source[inversion]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=phase2.result.instance.galaxies.lens.bulge,
                disk=phase2.result.instance.galaxies.lens.disk,
                mass=mass,
                shear=phase2.result.model.galaxies.lens.shear,
                hyper_galaxy=phase3.result.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=phase3.result.instance.galaxies.source.pixelization,
                regularization=phase3.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase3.result.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.instance.hyper_image_sky,
        hyper_background_noise=phase3.result.instance.hyper_background_noise,
        settings=settings,
        use_as_hyper_dataset=True,
    )

    phase4 = phase4.extend_with_hyper_phase(
        setup_hyper=slam.setup_hyper, include_hyper_image_sky=True
    )

    return al.PipelineDataset(
        pipeline_name,
        path_prefix,
        source_parametric_results,
        phase1,
        phase2,
        phase3,
        phase4,
    )
