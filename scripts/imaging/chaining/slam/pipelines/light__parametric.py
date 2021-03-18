from os import path
import autofit as af
import autolens as al


def light__parametric__with_lens_light(slam, settings, source_results):

    pipeline_name = "pipeline_light[parametric]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The bulge model, disk model, etc. and whether their centres / elliptical_comps are aligned.
        3) The lens galaxy mass model includes an  `ExternalShear`.
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix, pipeline_name, slam.source_tag, slam.light_parametric_tag
    )

    """
    Search 1: Fit the lens `Galaxy`'s light, where we:

        1) Fix the lens `Galaxy`'s mass and source galaxy to the results of the previous pipeline.
        2) Vary the lens galaxy hyper noise factor if hyper-galaxies noise scaling is on.
    """

    """SlaM:  If hyper-galaxy noise scaling is on, it may over-scale the noise making this new `LightProfile` 
    fit the data less well. This can be circumvented by including the noise scaling as a free parameter."""

    hyper_galaxy = slam.setup_hyper.hyper_galaxy_lens_from_result(
        result=source_results.last, noise_factor_is_model=True
    )

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=slam.pipeline_light.setup_light.bulge_prior_model,
        disk=slam.pipeline_light.setup_light.disk_prior_model,
        envelope=slam.pipeline_light.setup_light.envelope_prior_model,
        mass=source_results.last.instance.galaxies.lens.mass,
        shear=source_results.last.instance.galaxies.lens.shear,
        hyper_galaxy=hyper_galaxy,
    )

    """
    SLaM: Use the Source pipeline source as an instance (whether its parametric or an Inversion).
    """

    source = slam.source_from_result(result=source_results.last, source_is_model=False)

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_light[parametric]_mass[fixed]_source[fixed]",
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(lens=lens, source=source),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=source_results.last, as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=source_results.last
        ),
        settings=settings,
    )

    if slam.pipeline_source_inversion is not None:
        phase1.preload_inversion = True

    if not slam.setup_hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_hyper_phase(setup_hyper=slam.setup_hyper)

    return al.PipelineDataset(pipeline_name, path_prefix, source_results, phase1)
