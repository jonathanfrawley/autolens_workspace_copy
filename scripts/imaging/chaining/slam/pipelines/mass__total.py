import autofit as af
import autolens as al
from . import slam_util


def mass__total__no_lens_light(
    path_prefix: str,
    analysis,
    setup_hyper: al.SetupHyper,
    source_results,
    mass: af.PriorModel(al.mp.MassProfile) = af.PriorModel(al.mp.EllipticalIsothermal),
    mass_centre: (float, float) = None,
    end_stochastic=False,
):
    """

    Parameters
    ----------
    path_prefix : str or None
        The prefix of folders between the output path and the search folders.
    analysis : al.AnalysisImaging
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    mass : af.PriorModel(mp.MassProfile)
        The `MassProfile` fitted by this pipeline.
    mass_centre : (float, float)
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    end_stochastic : bool
        If True, the pipeline ends with a stochastic phase for reliable error estimation.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the MASS TOTAL PIPELINE we fit a lens model where:

     - The lens galaxy is modeled using a total mass distribution [Priors initialized from SOURCE PIPELINE].
     - The source galaxy's light is parametric or an inversion depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE
    """
    mass = slam_util.mass__from_result(
        mass=mass, result=source_results[-1], unfix_mass_centre=True
    )

    if mass_centre is not None:
        mass.centre = mass_centre

    source = slam_util.source__from_result_model_if_parametric(
        result=source_results[-1], setup_hyper=setup_hyper
    )

    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=source_results[-1].instance.galaxies.lens.redshift,
                mass=mass,
                shear=source_results[-1].model.galaxies.lens.shear,
            ),
            source=source,
        )
    )

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="mass_total[1]_mass[total]_source",
        n_live_points=100,
    )

    result_1 = search.fit(model=model, analysis=analysis)

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:

    This hyper-search runs if:

     - The source is using an `Inversion`.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky input`.
     - The background noise is included via the `hyper_background_noise`.
     
    __Stochastic Extension__
    
    If `end_stochastic=True`, the pipeline ends with a stochastic phase.
    """
    if end_stochastic:

        result_1 = al.util.model.stochastic_fit(
            setup_hyper=setup_hyper, result=result_1, search=search, analysis=analysis
        )

    else:

        if not setup_hyper.hyper_fixed_after_source:

            result_1 = al.util.model.hyper_fit(
                setup_hyper=setup_hyper,
                result=result_1,
                search=search,
                analysis=analysis,
                include_hyper_image_sky=True,
            )

    results = af.ResultsCollection()
    results.add(search.paths.name, result_1)

    return results


def mass__total__with_lens_light(
    slam, settings, source_results, light_results, end_stochastic=False
):

    pipeline_name = "pipeline_mass[total]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
        3) The lens`s light model is fixed or variable.
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix,
        pipeline_name,
        slam.source_tag,
        slam.light_parametric_tag,
        slam.mass_tag,
    )

    """
    Search 1: Fit the lens `Galaxy`'s light and mass and one source galaxy, where we:

        1) Use the source galaxy of the `source` pipeline.
        2) Use the lens galaxy light of the `light` pipeline.
        3) Set priors on the lens galaxy `MassProfile`'s using the `EllipticalIsothermal` and `ExternalShear` of 
           previous pipelines.
    """

    mass = slam.pipeline_mass.setup_mass.mass_prior_model_with_updated_priors_from_result(
        result=source_results[-2], unfix_mass_centre=True
    )

    """SLaM: Set whether shear is included in the mass model using the `ExternalShear` model of the Source pipeline."""

    shear = slam.pipeline_mass.shear_from_result(result=source_results[-2])

    """SLaM: Use the source and lens light models from the previous *Source* and *Light* pipelines."""

    lens = slam.lens_for_mass_pipeline_from_result(
        result=light_results[-1], mass=mass, shear=shear
    )

    source = slam.source_from_result_model_if_parametric(result=source_results[-2])

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_light[parametric]_mass[total]_source", n_live_points=100
        ),
        galaxies=af.CollectionPriorModel(lens=lens, source=source),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=light_results[-1], as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=light_results[-1]
        ),
        settings=settings,
    )

    if end_stochastic:

        phase1 = phase1.extend_with_stochastic_phase(
            stochastic_search=af.DynestyStatic(n_live_points=100),
            include_lens_light=slam.pipeline_mass.light_is_model,
        )

    else:

        if not slam.setup_hyper.hyper_fixed_after_source:

            phase1 = phase1.extend_with_hyper_phase(setup_hyper=slam.setup_hyper)

    return al.PipelineDataset(pipeline_name, path_prefix, light_results, phase1)
