import autofit as af
import autolens as al
from autofit.non_linear.grid import sensitivity as s


def subhalo__detection_single_plane__no_lens_light(
    slam, settings, mass_results, end_stochastic=False
):
    pipeline_name = "pipeline_subhalo"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix,
        pipeline_name,
        slam.source_tag,
        slam.mass_tag,
        slam.setup_subhalo.tag,
    )

    """
    Phase1 : Refit the lens`s `MassProfile`'s and source, where we:

        1) Use the source galaxy model of the `source` pipeline.
        2) Fit this source as a model if it is parametric and as an instance if it is an `Inversion`.
    """

    """
    SLaM: Setup the source passing them from the previous pipelines.
    """

    source = slam.source_from_result_model_if_parametric(result=mass_results.last)

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(name="phase[1]_mass[total_refine]", n_live_points=100),
        galaxies=af.CollectionPriorModel(
            lens=mass_results.last.model.galaxies.lens, source=source
        ),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=mass_results.last
        ),
        settings=settings,
    )

    """
    This `GridPhase` is used for all 3 subhalo detection phases, specifying that the subhalo (y,x) coordinates 
    are fitted for on a grid of non-linear searches.
    """

    class GridPhase(
        af.as_grid_search(
            phase_class=al.PhaseImaging,
            number_of_cores=slam.setup_subhalo.number_of_cores,
            number_of_steps=slam.setup_subhalo.number_of_steps,
        )
    ):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    """
    Phase Lens Plane: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift is fixed to that of the lens galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        5) For an `Inversion`, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          `LightProfile` they are varied (this is customized using source_is_model).
    """

    subhalo = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )

    subhalo.mass.redshift_object = slam.redshift_lens
    subhalo.mass.redshift_source = slam.redshift_source

    """
    SLaM: Setup the source model, which uses the the phase1 result is a model or instance depending on the 
    *source_is_model* parameter of `SetupSubhalo`.
    """

    source = slam.source_for_subhalo_pipeline_from_result(result=mass_results.last)

    phase2 = GridPhase(
        search=af.DynestyStatic(
            name="phase[2]_mass[total]_source_subhalo[search_lens_plane]",
            n_live_points=50,
            walks=5,
            facc=0.2,
        ),
        galaxies=af.CollectionPriorModel(
            lens=mass_results.last.model.galaxies.lens, subhalo=subhalo, source=source
        ),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=mass_results.last
        ),
        settings=settings,
    )

    subhalo = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo.mass.mass_at_200 = phase2.result.model.galaxies.subhalo.mass.mass_at_200
    subhalo.mass.centre = phase2.result.model.galaxies.subhalo.mass.centre

    subhalo.mass.redshift_object = slam.redshift_lens
    subhalo.mass.redshift_source = slam.redshift_source

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_subhalo[single_plane_refine]",
            path_prefix=path_prefix,
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(
            lens=phase2.result.model.galaxies.lens,
            subhalo=subhalo,
            source=phase2.result.model.galaxies.source,
        ),
        hyper_image_sky=phase2.result.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    if end_stochastic:
        phase3 = phase3.extend_with_stochastic_phase(
            stochastic_search=af.DynestyStatic(n_live_points=100)
        )

    return al.PipelineDataset(
        pipeline_name, path_prefix, mass_results, phase1, phase2, phase3
    )


def subhalo__detection_multi_plane__no_lens_light(
    slam, settings, mass_results, end_stochastic=False
):
    pipeline_name = "pipeline_subhalo"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix,
        pipeline_name,
        slam.source_tag,
        slam.mass_tag,
        slam.setup_subhalo.tag,
    )

    """
    Phase1 : Refit the lens`s `MassProfile`'s and source, where we:

        1) Use the source galaxy model of the `source` pipeline.
        2) Fit this source as a model if it is parametric and as an instance if it is an `Inversion`.
    """

    """
    SLaM: Setup the source passing them from the previous pipelines.
    """

    source = slam.source_from_result_model_if_parametric(result=mass_results.last)

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(name="phase[1]_mass[total_refine]", n_live_points=100),
        galaxies=af.CollectionPriorModel(
            lens=mass_results.last.model.galaxies.lens, source=source
        ),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=mass_results.last
        ),
        settings=settings,
    )

    """
    This `GridPhase` is used for all 3 subhalo detection phases, specifying that the subhalo (y,x) coordinates 
    are fitted for on a grid of non-linear searches.
    """

    class GridPhase(
        af.as_grid_search(
            phase_class=al.PhaseImaging,
            number_of_cores=slam.setup_subhalo.number_of_cores,
            number_of_steps=slam.setup_subhalo.number_of_steps,
        )
    ):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    """
    Phase Multi: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift has a UniformPrior between Earth and the source galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        4) For an `Inversion`, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          `LightProfile` they are varied (this is customized using source_is_model).
    """

    """
    The subhalo redshift is free to vary between 0.0 and the lens galaxy redshift.
    """

    subhalo_z_multi = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo_z_multi.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1.0e6, upper_limit=1.0e11
    )
    subhalo_z_multi.mass.centre_0 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    subhalo_z_multi.mass.centre_1 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    subhalo_z_multi.mass.redshift_source = slam.redshift_source
    subhalo_z_multi.mass.redshift_object = af.UniformPrior(
        lower_limit=0.0, upper_limit=slam.redshift_source
    )

    phase2 = GridPhase(
        search=af.DynestyStatic(
            name="phase[2]_mass[total]_subhalo[search_multi_plane]",
            n_live_points=50,
            walks=5,
            facc=0.2,
        ),
        galaxies=af.CollectionPriorModel(
            lens=mass_results.last.model.galaxies.lens,
            subhalo=subhalo_z_multi,
            source=source,
        ),
        hyper_image_sky=phase1.result.instance.optional.hyper_image_sky,
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=mass_results.last
        ),
        settings=settings,
    )

    subhalo = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo.mass.mass_at_200 = phase2.result.model.galaxies.subhalo.mass.mass_at_200
    subhalo.mass.centre = phase2.result.model.galaxies.subhalo.mass.centre
    subhalo.mass.redshift_object = slam.redshift_lens

    subhalo.mass.redshift_source = slam.redshift_source

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]__subhalo[multi_plane_refine]",
            path_prefix=path_prefix,
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(
            lens=phase2.result.model.galaxies.lens,
            subhalo=subhalo,
            source=phase2.result.model.galaxies.source,
        ),
        hyper_image_sky=phase2.result.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    if end_stochastic:
        phase3 = phase3.extend_with_stochastic_phase(
            stochastic_search=af.DynestyStatic(n_live_points=100)
        )

    return al.PipelineDataset(
        pipeline_name, path_prefix, mass_results, phase1, phase2, phase3
    )


def subhalo__sensitivity_mapping__no_lens_light(
    slam, mask, psf, mass_results, analysis_cls
):
    """
    To begin, we define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is f
    itted to every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model
    which includes one!).
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix,
        "sensitivity",
        slam.source_tag,
        slam.mass_tag,
        slam.setup_subhalo.tag,
    )

    """
    We now define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is fitted to 
    every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model which 
    includes one!). 

    For this model, we can use the result of fitting this model to the dataset before sensitivity mapping via the 
    mass pipeline. This ensures the priors associated with each parameter are initialized so as to speed up
    each non-linear search performed during sensitivity mapping.
    """
    base_model = mass_results.last.model

    """
    We now define the `perturbation_model`, which is the model component whose parameters we iterate over to perform 
    sensitivity mapping. In this case, this model is a `SphericalNFWMCRLudlow` model and we will iterate over its
    `centre` and `mass_at_200`. We set it up as a `GalaxyModel` so it has an associated redshift and can be directly
    passed to the tracer in the simulate function below.

    Many instances of the `perturbation_model` are created and used to simulate the many strong lens datasets that we fit. 
    However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter 
    subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to 
    determine if the subhalo was detectable.

    By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every 
    dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark mattter 
    subhalo the model-fit including a subhalo provide higher values of Bayesian evidence than the simpler model-fit (and
    therefore when it is detectable!).
    """
    perturbation_model = al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalNFWMCRLudlow)

    """
    Sensitivity mapping is typically performed over a large range of parameters. However, to make this demonstration quick
    and clear we are going to fix the `centre` of the subhalo to a value near the Einstein ring of (1.6, 0.0). We will 
    iterate over just two `mass_at_200` values corresponding to subhalos of mass 1e6 and 1e11, of which only the latter
    will be shown to be detectable.
    """
    perturbation_model.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1e6, upper_limit=1e11
    )
    perturbation_model.mass.centre.centre_0 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    perturbation_model.mass.centre.centre_1 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    perturbation_model.mass.redshift_object = slam.redshift_lens
    perturbation_model.mass.redshift_source = slam.redshift_source

    """
    We are performing sensitivity mapping to determine when a subhalo is detectable. Eery simulated dataset must 
    be simulated with a lens model, called the `simulation_instance`. We use the maximum likelihood model of the mass pipeline
    for this.

    This includes the lens light and mass and source galaxy light.
    """
    simulation_instance = mass_results.last.instance

    """
    We now write the `simulate_function`, which takes the `simulation_instance` of our model (defined above) and uses it to 
    simulate a dataset which is subsequently fitted.

    Note that when this dataset is simulated, the quantity `instance.perturbation` is used in the `simulate_function`.
    This is an instance of the `SphericalNFWMCRLudlow`, and it is different every time the `simulate_function` is called
    based on the value of sensitivity being computed. 

    In this example, this `instance.perturbation` corresponds to two different subhalos with values of `mass_at_200` of 
    1e6 MSun and 1e11 MSun.
    """

    def simulate_function(instance):
        """
        Set up the `Tracer` which is used to simulate the strong lens imaging, which may include the subhalo in
        addition to the lens and source galaxy.
        """
        tracer = al.Tracer.from_galaxies(
            galaxies=[
                instance.galaxies.lens,
                instance.perturbation,
                instance.galaxies.source,
            ]
        )

        """
        Set up the grid, PSF and simulator settings used to simulate imaging of the strong lens. These should be tuned to
        match the S/N and noise properties of the observed data you are performing sensitivity mapping on.
        """
        grid = al.Grid2DIterate.uniform(
            shape_native=mask.shape_native,
            pixel_scales=mask.pixel_scales,
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16, 24],
        )

        simulator = al.SimulatorImaging(
            exposure_time=300.0,
            psf=psf,
            background_sky_level=0.1,
            add_poisson_noise=True,
        )

        simulated_imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

        """
        The data generated by the simulate function is that which is fitted, so we should apply the mask for the analysis 
        here before we return the simulated data.
        """
        return al.MaskedImaging(imaging=simulated_imaging, mask=mask)

    """
    We next specify the search used to perform each model fit by the sensitivity mapper.
    """
    search = af.DynestyStatic(path_prefix=path_prefix, n_live_points=50)

    """
    We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
    object below are:

    - `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this 
    example it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform 
    sensitivity mapping on.

    - `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. 
    In this example is composed of an `EllipticalIsothermal` lens and `EllipticalSersic` source.

    - `perturbation_model`: This is the extra model component that alongside the `base_model` is fitted to every 
    simulated dataset. In this example it is a `SphericalNFWMCRLudlow` dark matter subhalo.

    - `simulate_function`: This is the function that uses the `simulation_instance` and many instances of the 
    `perturbation_model` to simulate many datasets that are fitted with the `base_model` 
    and `base_model` + `perturbation_model`.

    - `analysis_class`: The wrapper `Analysis` class that passes each simulated dataset to the `Analysis` class that 
    fits the data.

    - `number_of_steps`: The number of steps over which the parameters in the `perturbation_model` are iterated. In 
    this example, `mass_at_200` has a `LogUniformPrior` with lower limit 1e6 and upper limit 1e11, therefore 
    the `number_of_steps` of 2 will simulate and fit just 2 datasets where the `mass_at_200` is between 1e6 and 1e11.

    - `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel 
    processing if set above 1.
    """
    return s.Sensitivity(
        search=search,
        simulation_instance=simulation_instance,
        base_model=base_model,
        perturbation_model=perturbation_model,
        simulate_function=simulate_function,
        analysis_class=analysis_cls,
        number_of_steps=slam.setup_subhalo.number_of_steps,
        number_of_cores=slam.setup_subhalo.number_of_cores,
    )


def subhalo__detection_single_plane__with_lens_light(
    slam, settings, mass_results, end_stochastic=False
):

    pipeline_name = "pipeline_subhalo"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix,
        pipeline_name,
        slam.source_tag,
        slam.light_parametric_tag,
        slam.mass_tag,
        slam.setup_subhalo.tag,
    )

    """
    Phase1 : Refit the lens`s `MassProfile`'s and source, where we:

        1) Use the source galaxy model of the `source` pipeline.
        2) Fit this source as a model if it is parametric and as an instance if it is an `Inversion`.
    """

    """
    SLaM: Setup the lens and source passing them from the previous pipelines.
    """
    lens = mass_results.last.model.galaxies.lens
    lens.hyper_galaxy = slam.setup_hyper.hyper_galaxy_lens_from_result(
        result=mass_results.last
    )
    source = slam.source_from_result_model_if_parametric(result=mass_results.last)

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(name="phase[1]_mass[refine]", n_live_points=100),
        galaxies=af.CollectionPriorModel(lens=lens, source=source),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=mass_results.last
        ),
        settings=settings,
    )

    """
    This `GridPhase` is used for all 3 subhalo detection phases, specifying that the subhalo (y,x) coordinates 
    are fitted for on a grid of non-linear searches.
    """

    class GridPhase(
        af.as_grid_search(
            phase_class=al.PhaseImaging,
            number_of_cores=slam.setup_subhalo.number_of_cores,
            number_of_steps=slam.setup_subhalo.number_of_steps,
        )
    ):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    """
    Phase Lens Plane: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift is fixed to that of the lens galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        5) For an `Inversion`, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          `LightProfile` they are varied (this is customized using source_is_model).
    """
    subhalo = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )

    subhalo.mass.redshift_object = slam.redshift_lens
    subhalo.mass.redshift_source = slam.redshift_source

    """
    SLaM: Setup the source model, which uses the the phase1 result is a model or instance depending on the 
    *source_is_model* parameter of `SetupSubhalo`.
    """
    source = slam.source_for_subhalo_pipeline_from_result(result=mass_results.last)

    phase2 = GridPhase(
        search=af.DynestyStatic(
            name="phase[2]_mass_source_subhalo[search_lens_plane]",
            n_live_points=50,
            walks=5,
            facc=0.2,
        ),
        galaxies=af.CollectionPriorModel(lens=lens, subhalo=subhalo, source=source),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=mass_results.last
        ),
        settings=settings,
    )

    subhalo = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo.mass.mass_at_200 = phase2.result.model.galaxies.subhalo.mass.mass_at_200
    subhalo.mass.centre = phase2.result.model.galaxies.subhalo.mass.centre

    subhalo.mass.redshift_object = slam.redshift_lens
    subhalo.mass.redshift_source = slam.redshift_source

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_subhalo[single_plane_refine]",
            path_prefix=path_prefix,
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(
            lens=phase2.result.model.galaxies.lens,
            subhalo=subhalo,
            source=phase2.result.model.galaxies.source,
        ),
        hyper_image_sky=phase2.result.model.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    if end_stochastic:

        phase3 = phase3.extend_with_stochastic_phase(
            stochastic_search=af.DynestyStatic(n_live_points=100),
            include_lens_light=slam.pipeline_mass.light_is_model,
        )

    return al.PipelineDataset(
        pipeline_name, path_prefix, mass_results, phase1, phase2, phase3
    )


def subhalo__detection_multi_plane__with_lens_light(
    slam, settings, mass_results, end_stochastic=False
):

    pipeline_name = "pipeline_subhalo"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
    """
    path_prefix = slam.path_prefix_from(
        slam.path_prefix,
        pipeline_name,
        slam.source_tag,
        slam.mass_tag,
        slam.setup_subhalo.tag,
    )

    """
    Phase1 : Refit the lens`s `MassProfile`'s and source, where we:

        1) Use the source galaxy model of the `source` pipeline.
        2) Fit this source as a model if it is parametric and as an instance if it is an `Inversion`.
    """

    """
    SLaM: Setup the lens and source passing them from the previous pipelines.
    """
    lens = mass_results.last.model.galaxies.lens
    lens.hyper_galaxy = slam.setup_hyper.hyper_galaxy_lens_from_result(
        result=mass_results.last
    )
    source = slam.source_from_result_model_if_parametric(result=mass_results.last)

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(name="phase[1]_mass[refine]", n_live_points=100),
        galaxies=af.CollectionPriorModel(lens=lens, source=source),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=mass_results.last
        ),
        settings=settings,
    )

    """
    This `GridPhase` is used for all 3 subhalo detection phases, specifying that the subhalo (y,x) coordinates 
    are fitted for on a grid of non-linear searches.
    """

    class GridPhase(
        af.as_grid_search(
            phase_class=al.PhaseImaging,
            number_of_cores=slam.setup_subhalo.number_of_cores,
            number_of_steps=slam.setup_subhalo.number_of_steps,
        )
    ):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    """
    Phase Multi: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift has a UniformPrior between Earth and the source galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        4) For an `Inversion`, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          `LightProfile` they are varied (this is customized using source_is_model).
    """

    """
    The subhalo redshift is free to vary between 0.0 and the lens galaxy redshift.
    """
    subhalo_z_multi = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo_z_multi.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1.0e6, upper_limit=1.0e11
    )
    subhalo_z_multi.mass.centre_0 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    subhalo_z_multi.mass.centre_1 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    subhalo_z_multi.mass.redshift_source = slam.redshift_source
    subhalo_z_multi.mass.redshift_object = af.UniformPrior(
        lower_limit=0.0, upper_limit=slam.redshift_source
    )

    phase2 = GridPhase(
        search=af.DynestyStatic(
            name="phase[2]_mass_subhalo[search_multi_plane]",
            n_live_points=50,
            walks=5,
            facc=0.2,
        ),
        galaxies=af.CollectionPriorModel(
            lens=lens, subhalo=subhalo_z_multi, source=source
        ),
        hyper_image_sky=slam.setup_hyper.hyper_image_sky_from_result(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=slam.setup_hyper.hyper_background_noise_from_result(
            result=mass_results.last
        ),
        settings=settings,
    )

    subhalo = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo.mass.mass_at_200 = phase2.result.model.galaxies.subhalo.mass.mass_at_200
    subhalo.mass.centre = phase2.result.model.galaxies.subhalo.mass.centre
    subhalo.mass.redshift_object = slam.redshift_lens

    subhalo.mass.redshift_source = slam.redshift_source

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]__subhalo[multi_plane_refine]",
            path_prefix=path_prefix,
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(
            lens=phase2.result.model.galaxies.lens,
            subhalo=subhalo,
            source=phase2.result.model.galaxies.source,
        ),
        hyper_image_sky=phase2.result.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    if end_stochastic:

        phase3 = phase3.extend_with_stochastic_phase(
            stochastic_search=af.DynestyStatic(n_live_points=100),
            include_lens_light=slam.pipeline_mass.light_is_model,
        )

    return al.PipelineDataset(
        pipeline_name, path_prefix, mass_results, phase1, phase2, phase3
    )


def subhalo__sensitivity_mapping__with_lens_light(
    slam, mask, psf, mass_results, analysis_cls
):
    """
    To begin, we define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is f
    itted to every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model
    which includes one!).
    """

    path_prefix = slam.path_prefix_from(
        slam.path_prefix,
        "sensitivity",
        slam.source_tag,
        slam.mass_tag,
        slam.setup_subhalo.tag,
    )

    """
    We now define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is fitted to 
    every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model which 
    includes one!). 

    For this model, we can use the result of fitting this model to the dataset before sensitivity mapping via the 
    mass pipeline. This ensures the priors associated with each parameter are initialized so as to speed up
    each non-linear search performed during sensitivity mapping.
    """
    base_model = mass_results.last.model

    """
    We now define the `perturbation_model`, which is the model component whose parameters we iterate over to perform 
    sensitivity mapping. In this case, this model is a `SphericalNFWMCRLudlow` model and we will iterate over its
    `centre` and `mass_at_200`. We set it up as a `GalaxyModel` so it has an associated redshift and can be directly
    passed to the tracer in the simulate function below.

    Many instances of the `perturbation_model` are created and used to simulate the many strong lens datasets that we fit. 
    However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter 
    subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to 
    determine if the subhalo was detectable.

    By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every 
    dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark mattter 
    subhalo the model-fit including a subhalo provide higher values of Bayesian evidence than the simpler model-fit (and
    therefore when it is detectable!).
    """
    perturbation_model = al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalNFWMCRLudlow)

    """
    Sensitivity mapping is typically performed over a large range of parameters. However, to make this demonstration quick
    and clear we are going to fix the `centre` of the subhalo to a value near the Einstein ring of (1.6, 0.0). We will 
    iterate over just two `mass_at_200` values corresponding to subhalos of mass 1e6 and 1e11, of which only the latter
    will be shown to be detectable.
    """
    perturbation_model.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1e6, upper_limit=1e11
    )
    perturbation_model.mass.centre.centre_0 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    perturbation_model.mass.centre.centre_1 = af.UniformPrior(
        lower_limit=-slam.setup_subhalo.grid_dimensions_arcsec,
        upper_limit=slam.setup_subhalo.grid_dimensions_arcsec,
    )
    perturbation_model.mass.redshift_object = slam.redshift_lens
    perturbation_model.mass.redshift_source = slam.redshift_source

    """
    We are performing sensitivity mapping to determine when a subhalo is detectable. Eery simulated dataset must 
    be simulated with a lens model, called the `simulation_instance`. We use the maximum likelihood model of the 
    mass pipeline for this.

    This includes the lens light and mass and source galaxy light.
    """
    simulation_instance = mass_results.last.instance

    """
    We now write the `simulate_function`, which takes the `simulation_instance` of our model (defined above) and uses it to 
    simulate a dataset which is subsequently fitted.

    Note that when this dataset is simulated, the quantity `instance.perturbation` is used in the `simulate_function`.
    This is an instance of the `SphericalNFWMCRLudlow`, and it is different every time the `simulate_function` is called
    based on the value of sensitivity being computed. 

    In this example, this `instance.perturbation` corresponds to two different subhalos with values of `mass_at_200` of 
    1e6 MSun and 1e11 MSun.
    """

    def simulate_function(instance):
        """
        Set up the `Tracer` which is used to simulate the strong lens imaging, which may include the subhalo in
        addition to the lens and source galaxy.
        """
        tracer = al.Tracer.from_galaxies(
            galaxies=[
                instance.galaxies.lens,
                instance.perturbation,
                instance.galaxies.source,
            ]
        )

        """
        Set up the grid, PSF and simulator settings used to simulate imaging of the strong lens. These should be tuned to
        match the S/N and noise properties of the observed data you are performing sensitivity mapping on.
        """
        grid = al.Grid2DIterate.uniform(
            shape_native=mask.shape_native,
            pixel_scales=mask.pixel_scales,
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16, 24],
        )

        simulator = al.SimulatorImaging(
            exposure_time=300.0,
            psf=psf,
            background_sky_level=0.1,
            add_poisson_noise=True,
        )

        simulated_imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

        """
        The data generated by the simulate function is that which is fitted, so we should apply the mask for the analysis 
        here before we return the simulated data.
        """
        return al.MaskedImaging(imaging=simulated_imaging, mask=mask)

    """
    We next specify the search used to perform each model fit by the sensitivity mapper.
    """
    search = af.DynestyStatic(path_prefix=path_prefix, n_live_points=50)

    """
    We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
    object below are:

    - `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this 
    example it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform 
    sensitivity mapping on.

    - `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. 
    In this example is composed of an `EllipticalIsothermal` lens and `EllipticalSersic` source.

    - `perturbation_model`: This is the extra model component that alongside the `base_model` is fitted to every 
    simulated dataset. In this example it is a `SphericalNFWMCRLudlow` dark matter subhalo.

    - `simulate_function`: This is the function that uses the `simulation_instance` and many instances of the 
    `perturbation_model` to simulate many datasets that are fitted with the `base_model` 
    and `base_model` + `perturbation_model`.

    - `analysis_class`: The wrapper `Analysis` class that passes each simulated dataset to the `Analysis` class that 
    fits the data.

    - `number_of_steps`: The number of steps over which the parameters in the `perturbation_model` are iterated. In 
    this example, `mass_at_200` has a `LogUniformPrior` with lower limit 1e6 and upper limit 1e11, therefore 
    the `number_of_steps` of 2 will simulate and fit just 2 datasets where the `mass_at_200` is between 1e6 and 1e11.

    - `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel 
    processing if set above 1.
    """
    return s.Sensitivity(
        search=search,
        simulation_instance=simulation_instance,
        base_model=base_model,
        perturbation_model=perturbation_model,
        simulate_function=simulate_function,
        analysis_class=analysis_cls,
        number_of_steps=slam.setup_subhalo.number_of_steps,
        number_of_cores=slam.setup_subhalo.number_of_cores,
    )
