import autofit as af
import autolens as al


def mass__from_result(mass, result: af.Result, unfix_mass_centre=False):
    """
    Returns an updated mass `PriorModel` whose priors are initialized from previous results in a pipeline.

    It includes an option to unfix the input `mass_centre` used in the SOURCE PIPELINE, such that if the `mass_centre`
    were fixed (e.g. to (0.0", 0.0")) it becomes a free parameter in this pipeline.

    This function generically links any `MassProfile` to any `MassProfile`, pairing parameters which share the
    same path.

    Parameters
    ----------
    results : af.Result
        The result of a previous SOURCE PARAMETRIC PIPELINE or SOURCE INVERSION PIPELINE.

    Returns
    -------
    af.PriorModel(mp.MassProfile)
        The total mass profile whose priors are initialized from a previous result.
    """

    mass.take_attributes(source=result.model.galaxies.lens.mass)

    if unfix_mass_centre and isinstance(mass.centre, tuple):

        centre_tuple = mass.centre

        mass.centre = af.PriorModel(mass.cls).centre

        mass.centre.centre_0 = af.GaussianPrior(mean=centre_tuple[0], sigma=0.05)
        mass.centre.centre_1 = af.GaussianPrior(mean=centre_tuple[1], sigma=0.05)

    return mass


def source__from_result(
    result: af.Result, setup_hyper, source_is_model: bool = False
) -> al.GalaxyModel:
    """
    Setup the source model using the previous pipeline and phase results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source can be returned as an `instance` or `model`, depending on the optional input. The default SLaM
    pipelines return parametric sources as a model (give they must be updated to properly compute a new mass
    model) and return inversions as an instance (as they have sufficient flexibility to typically not required
    updating). They use the *source_from_pevious_pipeline* method of the SLaM class to do this.

    Parameters
    ----------
    result : af.Result
        The result of the previous source pipeline.
    source_is_model : bool
        If `True` the source is returned as a *model* where the parameters are fitted for using priors of the
        phase result it is loaded from. If `False`, it is an instance of that phase's result.
    """

    hyper_galaxy = setup_hyper.hyper_galaxy_source_from_result(result=result)

    if result.instance.galaxies.source.pixelization is None:

        if source_is_model:

            return al.GalaxyModel(
                redshift=result.instance.galaxies.source.redshift,
                bulge=result.model.galaxies.source.bulge,
                disk=result.model.galaxies.source.disk,
                envelope=result.model.galaxies.source.envelope,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return al.GalaxyModel(
                redshift=result.instance.galaxies.source.redshift,
                bulge=result.instance.galaxies.source.bulge,
                disk=result.instance.galaxies.source.disk,
                envelope=result.instance.galaxies.source.envelope,
                hyper_galaxy=hyper_galaxy,
            )

    if not setup_hyper.hyper_fixed_after_source:

        if source_is_model:

            return al.GalaxyModel(
                redshift=result.instance.galaxies.source.redshift,
                pixelization=result.hyper.instance.galaxies.source.pixelization,
                regularization=result.hyper.model.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return al.GalaxyModel(
                redshift=result.instance.galaxies.source.redshift,
                pixelization=result.hyper.instance.galaxies.source.pixelization,
                regularization=result.hyper.instance.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )

    else:

        if source_is_model:

            return al.GalaxyModel(
                redshift=result.instance.galaxies.source.redshift,
                pixelization=result.instance.galaxies.source.pixelization,
                regularization=result.model.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return al.GalaxyModel(
                redshift=result.instance.galaxies.source.redshift,
                pixelization=result.instance.galaxies.source.pixelization,
                regularization=result.instance.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )


def source__from_result_model_if_parametric(result: af.Result, setup_hyper):
    """
    Setup the source model for a MASS PIPELINE using the previous SOURCE PIPELINE results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source is returned as a model if it is parametric (given its parameters must be fitted for to properly compute
    a new mass model) whereas inversions are returned as an instance (as they have sufficient flexibility to not
    require updating). This behaviour can be customized in SLaM pipelines by replacing this method with the
    `source__from_result` method.

    Parameters
    ----------
    result : af.Result
        The result of the previous source pipeline.
    """
    if result.instance.galaxies.source.pixelization is None:
        return source__from_result(
            result=result, setup_hyper=setup_hyper, source_is_model=True
        )
    return source__from_result(
        result=result, setup_hyper=setup_hyper, source_is_model=False
    )
