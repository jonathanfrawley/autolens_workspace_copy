"""
Tutorial 3: Lens and Source
===========================

In this tutorial, we demonstrate search chaining using three searches to fit strong lens `Imaging` which includes the
lens galaxy's light.

The crucial point to note is that for many lenses the lens galaxy's light can be fitted and subtracted reasonably 
well before we attempt to fit the source galaxy. This makes sense, as fitting the lens's light (which is an elliptical
blob of light in the centre of the imaging) looks nothing like the source's light (which is a ring of light)! Formally,
we would say that these two model components (the lens's light and source's light) are not covariate.

So, as a newly trained lens modeler, what does the lack of covariance between these parameters make you think?
Hopefully, you're thinking, why should I bother fitting the lens and source galaxy simultaneously? Surely we can
find the right regions of non-linear parameter space by fitting each separately first? This is what we're going to do
in this tutorial, using a pipeline composed of a modest 3 searches:

 1) Fit the lens galaxy's light, ignoring the source.
 2) Fit the source-galaxy's light (and therefore lens galaxy's mass), ignoring the len`s light.
 3) Fit both simultaneously, using these results to initialize our starting location in parameter space.

Of course, given that we do not care for the errors in searches 1 and 2, we will set up our non-linear search to
perform sampling as fast as possible!
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

we'll use strong lensing data, where:

 - The lens galaxy's light is an `EllSersic`.
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's `LightProfile` is an `EllExponential`.
 
This image was fitted throughout chapter 2.
"""
dataset_name = "light_sersic__mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

"""
__Paths__

All three searches will use the same `path_prefix`, so we write it here to avoid repetition.
"""
path_prefix = path.join("howtolens", "chapter_3", "tutorial_3_lens_and_source")

"""
__Masking (Search 1)__

We need to choose our mask for the analysis. Given we are only fitting the lens light we have two options: 

 - A circular mask that does not remove the source's light from the fit, assuming the lens light model will still be 
 sufficiently accurate to reveal the source in the second search.
 - An 'anti-annular' mask that removes the source's ring of light.

In this example, we will use the anti-annular mask to demonstrate that we can change the mask used by each search in a 
chain of non-linear searches.
"""
mask = al.Mask2D.circular_anti_annular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    inner_radius=0.8,
    outer_radius=2.2,
    outer_radius_2=3.0,
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In search 1 we fit a lens model where:

 - The lens galaxy's light is a parametric `EllSersic` bulge [7 parameters].
 
 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, bulge=al.lp.EllSersic)
    )
)

analysis = al.AnalysisImaging(dataset=imaging)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[1]_light[bulge]",
    nlive=30,
    dlogz=5.0,
)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Masking (Search 2)__

In search 2 we are only fitting the source's light, thus we can apply an annular mask that removes regions of the
image that contained only the lens's light.
"""
mask = al.Mask2D.circular_annular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    inner_radius=1.2,
    outer_radius=1.8,
)

imaging = imaging.apply_mask(mask=mask)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

In search 2 we fit a lens model where:

 - The lens galaxy's light is an `EllSersic` bulge [Parameters fixed to results of search 1].
 
 - The lens galaxy's total mass distribution is an `EllIsothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric `EllSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.

In search 2, we fit the source-`galaxy's light and fix the lens light model to the model inferred in search 1, 
ensuring the image we has the foreground lens subtracted. We do this below by passing the lens light as an `instance` 
object.

By passing an `instance`, we are telling **PyAutoLens** that we want it to pass the maximum log likelihood result of 
that search and use those parameters as fixed values in the model. The model parameters passed as an `instance` are not 
free parameters fitted for by the non-linear search, thus this reduces the dimensionality of the non-linear search 
making model-fitting faster and more reliable. 
     
Thus, search 2 includes the lens light model from search 1, but it is completely fixed during the model-fit!

We also use the centre of the `bulge` to initialize the priors on the lens's `mass`.
"""
mass = af.Model(al.mp.EllIsothermal)
mass.centre_0 = result_1.model.galaxies.lens.bulge.centre_0
mass.centre_1 = result_1.model.galaxies.lens.bulge.centre_1

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=result_1.instance.galaxies.lens.bulge,
            mass=mass,
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]_mass[sie]_source[bulge]",
    nlive=50,
    dlogz=5.0,
)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Masking (Search 3)__

In search 3 we fit the lens and source, therefore we will use a large circular mask.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

In search 3 we fit a lens model where:

 - The lens galaxy's light is an `EllSersic` bulge [7 Parameters: priors initialized from search 1].
 
 - The lens galaxy's total mass distribution is an `EllIsothermal` with `ExternalShear` [7 parameters: priors
 initalized from search 2].
 
 - The source galaxy's light is a parametric `EllSersic` [7 parameters: priors initalized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=25.

There isn't a huge amount to say about this search, we have initialized the priors on all of our models parameters
and the only thing that is left to do is fit for all model components simultaneously, with slower Dynesty settings
that will give us more accurate parameter values and errors.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=result_1.model.galaxies.lens.bulge,
            mass=result_2.model.galaxies.lens.mass,
        ),
        source=af.Model(
            al.Galaxy, redshift=1.0, bulge=result_2.model.galaxies.source.bulge
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[3]_light[bulge]_mass[sie]_source[bulge]",
    nlive=100,
)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

And there we have it, a sequence of searches that breaks modeling the lens and source galaxy into 3 simple searches. 
This approach is much faster than fitting the lens and source simultaneously from the beginning. Instead of asking you 
questions at the end of this chapter`s tutorials, I'm going to ask questions which I then answer. This will hopefully 
get you thinking about how to approach pipeline writing.

 1) Can this pipeline really be generalized to any lens? Surely the radii of the masks depends on the lens and source 
 galaxies?

Whilst this is true, we chose mask radii above that are `excessive` and masks out a lot more of the image than just 
the source (which, in terms of run-time, is desirable). Thus, provided you know the Einstein radius distribution of 
your lens sample, you can choose mask radii that will masks out every source in your sample adequately (and even if 
some of the source is still there, who cares? The fit to the lens galaxy will be okay).

However, the template pipelines provided on the `autolens_workspace` simply use circular masks for every search and do
not attempt to use different masks for the lens light fit and source fit. This is to keep things simple (at the expense
of slower run times). It is up to you if you want to adapt these scripts to try and use more specific masking strategies.
"""
