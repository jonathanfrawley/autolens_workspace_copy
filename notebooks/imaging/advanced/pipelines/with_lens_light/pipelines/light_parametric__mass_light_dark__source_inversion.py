from os import path
import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` with a strong lens model where:

 - The lens galaxy's light is modeled parametrically using one or more input `LightProfile`s.
 - The lens galaxy's light matter mass distribution is the  `LightProfile`'s of the 
      lens's light, where they are converted to a stellar mass distribution via constant mass-to-light ratios.
 - The lens galaxy's dark matter mass distribution is a _SphericalNFW_.
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

    Fit the lens mass model and source `LightProfile`, where the `LightProfile` parameters of the lens's 
    `LightMassProfile` are fixed to the results of search 1.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: EllipticalSersic + EllipticalSersic + EllipticalNFWMCRLudlow + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens Light (instance -> search 1).
    Notes: Uses the lens subtracted image from search 1.

Search 3:

    Refine the lens light and mass models and source light model using priors initialized from searches 1 and 2.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + EllipticalNFWMCRLudlow + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> search 1), lens mass and source light (model -> search 2).
    Notes: None

Search 4:

    Fit the source `Inversion` using the lens light and `MassProfile`'s inferred in search 3.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + EllipticalNFWMCRLudlow + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Light & Mass (instance -> phase3).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Search 5:

    Refines the lens light and mass models using the source `Inversion` of search 4.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + EllipticalNFWMCRLudlow + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Light & Mass (model -> search 3), Source `Inversion` (instance -> search 4)
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings):

    pipeline_name = "pipeline_light[parametric]_mass[light_dark]_source[inversion]"

    """
    This pipeline is tagged according to whether:

        1) The bulge model, disk model, etc. and whether their centres / elliptical_comps are aligned.
        2) If the bulge and disk share the same mass-to-light ratio or each is fitted independently.
        3) The lens galaxy mass model includes an `ExternalShear`.
    """

    path_prefix = path.join(setup.path_prefix, pipeline_name, setup.tag)

    """
    Search 1: Fit only the lens galaxy's light, where we:

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
        2) Set priors on the centre of the lens galaxy's dark matter mass distribution by chaining them to those inferred 
           for the `LightProfile` in search 1.
        3) Use a `EllipticalNFWMCRLudlow` model for the dark matter which sets its scale radius via a mass-concentration
           relation and the lens and source redshifts.
    """

    bulge, disk, envelope = setup.setup_mass.light_and_mass_prior_models_with_updated_priors(
        result=phase1.result, as_instance=True
    )

    dark = setup.setup_mass.dark_prior_model
    dark.mass_at_200 = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e15)
    dark.redshift_object = setup.redshift_lens
    dark.redshift_source = setup.redshift_source

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_light[fixed]_mass[light_dark]_source[bulge]",
            n_live_points=60,
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=bulge,
                disk=disk,
                envelope=envelope,
                dark=dark,
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

        1) Set the lens's bulge, disk, dark, and source's light using the results of searches 1 and 2.
    """

    bulge, disk, envelope = setup.setup_mass.light_and_mass_prior_models_with_updated_priors(
        result=phase1.result, as_instance=False
    )

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_light[parametric]_mass[light_dark]_source[bulge]",
            n_live_points=100,
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=bulge,
                disk=disk,
                envelope=envelope,
                dark=phase2.result.model.galaxies.lens.dark,
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
                dark=phase3.result.instance.galaxies.lens.dark,
                shear=phase3.result.instance.galaxies.lens.shear,
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

    phase5 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[5]_light[parametric]_mass[light_dark]_source[inversion]",
            n_live_points=100,
        ),
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=setup.redshift_lens,
                bulge=phase3.result.model.galaxies.lens.bulge,
                disk=phase3.result.model.galaxies.lens.disk,
                envelope=phase3.result.model.galaxies.lens.envelope,
                dark=phase3.result.model.galaxies.lens.dark,
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
