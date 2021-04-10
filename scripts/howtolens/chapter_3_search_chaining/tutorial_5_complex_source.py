"""
Tutorial 5: Complex Source
==========================

Up to now, we've not paid much attention to the source galaxy's morphology. We've assumed its a single-component
exponential profile, which is a fairly crude assumption. A quick look at any image of a real galaxy reveals a
wealth of different structures that could be present: bulges, disks, bars, star-forming knots and so on. Furthermore,
there could be more than one source-galaxy!

In this example, we'll explore how far we get fitting a complex source using a pipeline. Fitting complex source's is
an exercise in diminishing returns. Each light profile we add to our source model brings with it an extra 5-7,
parameters. If there are 4 components, or multiple galaxies, we are quickly entering the somewhat nasty regime of
30-40+ parameters in our non-linear search. Even with a pipeline, that is a lot of parameters to fit!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

we'll use new strong lensing data, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's `LightProfile` is four `EllSersic`.
"""
dataset_name = "mass_sie__source_sersic_x4"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Paths__

All four searches will use the same `path_prefix`, so we write it here to avoid repetition.
"""
path_prefix = path.join("howtolens", "chapter_3", "tutorial_5_complex_source")

"""
__Search Chaining Approach__

The source is clearly complex, with more than 4 peaks of light. Through visual inspection of this image, we cannot state
with confidence how many sources of light there truly is! The data also omits he lens galaxy's light. This keep the 
number of parameters down and therefore makes the searches faster, however we would not get such a luxury for a real 
galaxy.

To fit this lens with a complex source model, our approach is simply to fit the the lens galaxy mass and source using
one light profile in the first search, and then add an additional light profile to each search. The mass model and
light profiles inferred in the previous search are then used to pass priors.

__Model + Search + Analysis + Model-Fit (Search 1)__

 - The lens galaxy's total mass distribution is an `EllIsothermal` [5 parameters].
 
 - The source galaxy's light is a parametric `EllSersic` [7 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge_0=al.lp.EllSersic),
    )
)

analysis = al.AnalysisImaging(dataset=imaging)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[1]__mass[sie]__source_x1[bulge]",
    nlive=40,
    dlogz=5.0,
)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

 - The lens galaxy's total mass distribution is an `EllIsothermal` [5 parameters: priors initialized from 
 search 1].

 - The source galaxy's light is two parametric `EllSersic` [14 parameters: first Sersic initialized from 
 search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=result_1.model.galaxies.lens.mass),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge_0=result_1.model.galaxies.source.bulge_0,
            bulge_1=al.lp.EllSersic,
        ),
    )
)

analysis = al.AnalysisImaging(dataset=imaging)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]_mass[sie]_source_x2[bulge]",
    nlive=40,
    dlogz=5.0,
)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

 - The lens galaxy's total mass distribution is an `EllIsothermal` [5 parameters: priors initialized from 
 search 2].

 - The source galaxy's light is three parametric `EllSersic` [21 parameters: first two Sersic's initialized from 
 search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=result_2.model.galaxies.lens.mass),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge_0=result_2.model.galaxies.source.bulge_0,
            bulge_1=result_2.model.galaxies.source.bulge_1,
            bulge_2=al.lp.EllSersic,
        ),
    )
)

analysis = al.AnalysisImaging(dataset=imaging)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[3]_mass[sie]_source_x3[bulge]",
    nlive=50,
    dlogz=5.0,
)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

 - The lens galaxy's total mass distribution is an `EllIsothermal` [5 parameters: priors initialized from 
 search 4].

 - The source galaxy's light is four parametric `EllSersic` [28 parameters: first three Sersic's initialized from 
 search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=26.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=result_3.model.galaxies.lens.mass),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge_0=result_3.model.galaxies.source.bulge_0,
            bulge_1=result_3.model.galaxies.source.bulge_1,
            bulge_2=result_3.model.galaxies.source.bulge_2,
            bulge_3=al.lp.EllSersic,
        ),
    )
)

analysis = al.AnalysisImaging(dataset=imaging)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[4]_mass[sie]_source_x4[bulge]",
    nlive=50,
    dlogz=0.3,
)

result_4 = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

With four light profiles, we were still unable to produce a fit to the source that did not leave residuals. However, I 
actually simulated the lens using a source with four light profiles. A `perfect fit` was therefore somewhere in 
parameter space, but our search unfortunately was unable to locate this.

Lets confirm this, by manually fitting the imaging data with the true input model.

We cannot apply a mask to a dataset that was already masked, so we first reload the imaging from .fits.
"""
imaging = imaging.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
    )
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light_0=al.lp.EllSersic(
        centre=(0.1, 0.1),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.1,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
    light_1=al.lp.EllSersic(
        centre=(0.8, 0.6),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.5, angle=30.0),
        intensity=0.2,
        effective_radius=0.3,
        sersic_index=3.0,
    ),
    light_2=al.lp.EllSersic(
        centre=(-0.3, 0.6),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.3, angle=120.0),
        intensity=0.6,
        effective_radius=0.5,
        sersic_index=1.5,
    ),
    light_3=al.lp.EllSersic(
        centre=(-0.3, -0.3),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=85.0),
        intensity=0.4,
        effective_radius=0.1,
        sersic_index=2.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

true_fit = al.FitImaging(imaging=imaging, tracer=tracer)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=true_fit)
fit_imaging_plotter.subplot_fit_imaging()
fit_imaging_plotter.subplot_of_planes(plane_index=1)

"""
And indeed, we see an improved residual-map, chi-squared-map, and so forth.

If the source morphology is complex, there is no way we chain searches to fit it perfectly. The non-linear parameter 
space simply becomes too complex. For this tutorial, this was true even though our source model could actually fit 
the data perfectly. For  real lenses, the source may be *even more complex* giving us even less hope of getting a 
good fit.

But fear not, **PyAutoLens** has you covered. In chapter 4, we'll introduce a completely new way to model the source 
galaxy, which addresses the problem faced here.
"""
