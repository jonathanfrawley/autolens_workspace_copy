from os import path
import autofit as af
import autolens as al

"""
This pipeline performs a parametric source analysis which fits an image with a lens mass model and
source galaxy.

This pipeline uses 1 phase:

Search 1:

    Fit the lens mass model and source `LightProfile`.
    
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: None
    Notes: None
"""


def make_pipeline(slam, settings):

    pipeline_name = "pipeline_source[parametric]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an `ExternalShear`.
        3) The source model determined from `SetupSourceParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.)
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix, pipeline_name, slam.source_parametric_tag
    )

    """
    Search 1: Fit the lens`s `MassProfile`'s and source galaxy.
    """

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_mass[total]_source[parametric]", n_live_points=200, walks=10
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=slam.redshift_lens,
                mass=slam.pipeline_source_parametric.setup_mass.mass_prior_model,
                shear=slam.pipeline_source_parametric.setup_mass.shear_prior_model,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=slam.redshift_source,
                bulge=slam.pipeline_source_parametric.setup_source.bulge_prior_model,
                disk=slam.pipeline_source_parametric.setup_source.disk_prior_model,
                envelope=slam.pipeline_source_parametric.setup_source.envelope_prior_model,
            ),
        ),
        settings=settings,
    )

    phase1 = phase1.extend_with_hyper_phase(
        setup_hyper=slam.setup_hyper, include_hyper_image_sky=True
    )

    return al.PipelineDataset(pipeline_name, path_prefix, None, phase1)
