import autofit as af
import autolens as al


def hyper_fit(
    setup_hyper: al.SetupHyper,
    result: af.Result,
    analysis,
    include_hyper_image_sky: bool = False,
):
    """
    Perform a hyper-fit, which extends a model-fit with an additional fit which fixes the non-hyper components of the
    model (e.g., `LightProfile`'s, `MassProfile`) to the `Result`'s maximum likelihood fit. The hyper-fit then treats
    only the hyper-model components as free parameters, which are any of the following model components:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
    data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `Inversion` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    hyper_model : CollectionPriorModel
        The hyper model used by the hyper-fit, which models hyper-components like a `Pixelization` or `HyperGalaxy`'s.
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result : af.Result
        The result of a previous `Analysis` phase whose maximum log likelihood model forms the basis of the hyper model.
    analysis : Analysis
        An analysis class used to fit imaging or interferometer data with a model.

    Returns
    -------
    af.Result
        The result of the hyper model-fit, which has a new attribute `result.hyper` that contains updated parameter
        values for the hyper-model components for passing to later model-fits.
    """

    if not setup_hyper.hyper_fixed_after_source:

        hyper_model = al.util.model.hyper_model_from(
            setup_hyper=setup_hyper,
            result=result,
            include_hyper_image_sky=include_hyper_image_sky,
        )

        return al.util.model.hyper_fit(
            hyper_model=hyper_model,
            setup_hyper=setup_hyper,
            result=result,
            analysis=analysis,
        )

    return result


def stochastic_fit(result, analysis):
    """
    Extend a model-fit with a stochastic model-fit, which refits a model but introduces a log likelihood cap whereby
    all model-samples with a likelihood above this cap are rounded down to the value of the cap.

    This `log_likelihood_cap` is determined by sampling ~250 log likelihood values from the original model's maximum
    log likelihood model. However, the pixelization used to reconstruct the source of each model evaluation uses a
    different KMeans seed, such that each reconstruction uses a unique pixel-grid. The model must therefore use a
    pixelization which uses the KMeans method to construct the pixel-grid, for example the `VoronoiBrightnessImage`.

    The cap is computed as the mean of these ~250 values and it is introduced to avoid underestimated errors due
    to artificial likelihood boosts.

    Parameters
    ----------
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result : af.Result
        The result of a previous `Analysis` phase whose maximum log likelihood model forms the basis of the hyper model.
    """

    stochastic_model = al.util.model.stochastic_model_from(result=result)

    return al.util.model.stochastic_fit(
        stochastic_model=stochastic_model, result=result, analysis=analysis
    )
