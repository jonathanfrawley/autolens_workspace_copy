import autofit as af
import autolens as al
from . import extensions

from typing import Union

def no_lens_light(
    path_prefix: str,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    mass: af.Model(al.mp.MassProfile) = af.Model(al.mp.EllipticalIsothermal),
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear),
    source_bulge: af.Model(al.lp.LightProfile) = af.Model(al.lp.EllipticalSersic),
    source_disk: af.Model(al.lp.LightProfile) = None,
    source_envelope: af.Model(al.lp.LightProfile) = None,
    redshift_lens: float = 0.5,
    redshift_source: float = 1.0,
    mass_centre: (float, float) = None,
):
    """
    The SlaM SOURCE PARAMETRIC PIPELINE for fitting imaging data without a lens light component.

    Parameters
    ----------
    path_prefix : str or None
        The prefix of folders between the output path and the search folders.
    analysis : al.AnalysisImaging
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    mass : af.Model(mp.MassProfile)
        The `MassProfile` fitted by this pipeline.
    shear : af.Model(mp.MassSheet)
        The model used to represent the external shear in the mass model (set to None to turn off shear).
                bulge_prior_model : af.Model(lp.LightProfile)
    source_bulge : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the source galaxy's bulge (set to
        None to omit a bulge).
    source_disk : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the source galaxy's disk (set to
        None to omit a disk).
    source_envelope : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the source galaxy's envelope (set to
        None to omit an envelope).
    redshift_lens : float
        The redshift of the lens galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    redshift_source : float
        The redshift of the source galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    mass_centre : (float, float)
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the SOURCE PARAMETRIC PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a parametric bulge + disk + envelope [no prior initialization].

    This search aims to accurately estimate the lens mass model and source model.
    """
    if mass_centre is not None:
        mass.centre = mass_centre

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=redshift_lens, mass=mass, shear=shear),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
                disk=source_disk,
                envelope=source_envelope,
            ),
        )
    )

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="source_parametric[1]_mass[total]_source[parametric]",
        n_live_points=200,
        walks=10,
    )

    result_1 = search.fit(model=model, analysis=analysis)

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:
    
     - The background sky is included via `hyper_image_sky`.
     - The background noise is included via the `hyper_background_noise`.
     - The source galaxy includes a `HyperGalaxy` for scaling the noise.
    """
    result_1 = extensions.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_1,
        analysis=analysis,
        include_hyper_image_sky=True,
    )

    return af.ResultsCollection([result_1])


def with_lens_light(
    path_prefix: str,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    lens_bulge: af.Model(al.lp.LightProfile) = af.Model(al.lp.EllipticalSersic),
    lens_disk: af.Model(al.lp.LightProfile) = af.Model(al.lp.EllipticalExponential),
    lens_envelope: af.Model(al.lp.LightProfile) = None,
    mass: af.Model(al.mp.MassProfile) = af.Model(al.mp.EllipticalIsothermal),
    shear: af.Model(al.mp.ExternalShear) = af.Model(al.mp.ExternalShear),
    source_bulge: af.Model(al.lp.LightProfile) = af.Model(al.lp.EllipticalSersic),
    source_disk: af.Model(al.lp.LightProfile) = None,
    source_envelope: af.Model(al.lp.LightProfile) = None,
    redshift_lens: float = 0.5,
    redshift_source: float = 1.0,
    mass_centre: (float, float) = None,
):
    """
    The SlaM SOURCE PARAMETRIC PIPELINE for fitting imaging data with a lens light component.

    Parameters
    ----------
    path_prefix : str or None
        The prefix of folders between the output path and the search folders.
    analysis : al.AnalysisImaging
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    lens_bulge : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_envelope : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the lens galaxy's envelope (set to
        None to omit an envelope).        
    mass : af.Model(mp.MassProfile)
        The `MassProfile` fitted by this pipeline.
    shear : af.Model(mp.MassSheet)
        The model used to represent the external shear in the mass model (set to None to turn off shear).
                bulge_prior_model : af.Model(lp.LightProfile)
    source_bulge : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the source galaxy's bulge (set to
        None to omit a bulge).
    source_disk : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the source galaxy's disk (set to
        None to omit a disk).
    source_envelope : af.Model(lp.LightProfile)
        The `LightProfile` `Model` used to represent the light distribution of the source galaxy's envelope (set to
        None to omit an envelope).
    redshift_lens : float
        The redshift of the lens galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    redshift_source : float
        The redshift of the source galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
        solMass, etc.
    light_centre : (float, float) or None
       If input, a fixed (y,x) centre of the galaxy is used for every light model centre which are not treated as a
        free parameter by the non-linear search.
    light_centre_gaussian_prior_values : (float, float) or None
       If input, the mean and sigma of every light model centre is set using these values as (mean, sigma).
    mass_centre : (float, float)
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the SOURCE PARAMETRIC PIPELINE we fit a lens model where:

     - The lens galaxy light is modeled using parametric bulge + disk + envelope [no prior initialization].
     - The lens's mass and source galaxy are omitted from the fit.

    This search aims to produce a somewhat accurate lens light subtracted image for the next search which fits the 
    the lens mass model and source model.
    """
    lens = af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        bulge=lens_bulge,
        disk=lens_disk,
        envelope=lens_envelope,
    )

    model = af.Collection(galaxies=af.Collection(lens=lens))

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="source_parametric[1]_light[parametric]",
        n_live_points=75,
    )

    result_1 = search.fit(model=model, analysis=analysis)

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    In search 2 of the SOURCE PARAMETRIC PIPELINE we fit a lens model where:

     - The lens galaxy light is modeled using a parametric bulge + disk + envelope [fixed to result of Search 1].
     - The lens galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a parametric bulge + disk + envelope [no prior initialization].

    This search aims to accurately estimate the lens mass model and source model.
    """

    if mass_centre is not None:
        mass.centre = mass_centre

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=result_1.instance.galaxies.lens.bulge,
                disk=result_1.instance.galaxies.lens.disk,
                envelope=result_1.instance.galaxies.lens.envelope,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
                disk=source_disk,
                envelope=source_envelope,
            ),
        )
    )

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="source_parametric[2]_light[fixed]_mass[total]_source[parametric]",
        n_live_points=200,
        walks=10,
    )

    result_2 = search.fit(model=model, analysis=analysis)

    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    In search 2 of the SOURCE PARAMETRIC PIPELINE we fit a lens model where:

     - The lens galaxy light is modeled using a parametric bulge + disk + envelope [priors are not initialized from 
     previous searches].
     - The lens galaxy mass is modeled using a total mass distribution [priors initialized from search 2].
     - The source galaxy's light is a parametric bulge + disk + envelope [priors initialized from search 2].

    This search aims to accurately estimate the lens light model, mass model and source model.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=lens_disk,
                envelope=lens_envelope,
                mass=result_2.model.galaxies.lens.mass,
                shear=result_2.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=result_2.model.galaxies.source.bulge,
                disk=result_2.model.galaxies.source.disk,
                envelope=result_2.model.galaxies.source.envelope,
            ),
        )
    )

    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name="source_parametric[3]_light[parametric]_mass[total]_source[parametric]",
        n_live_points=100,
    )

    result_3 = search.fit(model=model, analysis=analysis)
    result_3.use_as_hyper_dataset = True

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:

     - The background sky is included via `hyper_image_sky`.
     - The background noise is included via the `hyper_background_noise`.
     - The source galaxy includes a `HyperGalaxy` for scaling the noise.
    """
    result_3 = extensions.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_3,
        analysis=analysis,
        include_hyper_image_sky=True,
    )

    return af.ResultsCollection([result_1, result_2, result_3])
