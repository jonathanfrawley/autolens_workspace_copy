"""
Tutorial 4: Two Lens Galaxies
=============================

Up to now, all the images we've fitted had one lens galaxy. However, we saw in chapter 1 that our lens plane can
consist of multiple galaxies which each contribute to the strong lensing. Multi-galaxy systems are challenging to
model, because they add an extra 5-10 parameters to the non-linear search and, more problematically, the degeneracies
between the `MassProfile`'s of the two galaxies can be severe.

However, we can still break their analysis down using muiltiple searches and give ourselves a shot at getting a good
lens model. Here, we're going to fit a double lens system, fitting as much about each individual lens galaxy before
fitting them simultaneously.

Up to now, I've put a focus on an analysis being general. The script we write in this example is going to be the
opposite, specific to the image we're modeling. Fitting multiple lens galaxies is really difficult and writing a
pipeline that we can generalize to many lenses isn't currently possible with **PyAutoLens**.
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
we'll use new strong lensing data, where:

 - There are two lens galaxy's whose `LightProfile`'s are both `EllipticalSersic``..
 - There are two lens galaxy's whose `MassProfile`'s are both `EllipticalIsothermal``..
 - The source galaxy's `LightProfile` is an `EllipticalExponential`.
"""
dataset_name = "light_sersic_x2__mass_sie_x2__source_sersic"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

"""
We need to choose our mask for the analysis. Given the lens light is present in the image we'll need to include all 
of its light in the central regions of the image, so lets use a circular mask.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Search Chaining Approach__

Looking at the image, there are clearly two blobs of light corresponding to our two lens galaxies. The source's 
light is also pretty complex and the arcs don't posses the rotational symmetry we're used to seeing up to now. 
Multi-galaxy ray-tracing is just a lot more complicated, which means so is modeling it!

So, how can we break the lens modeling up? As follows:

 1) Fit and subtract the light of each lens galaxy individually.
 2) Use these results to initialize each lens galaxy's total mass distribution.

So, with this in mind, we'll perform an analysis using searches:

 1) Fit the `LightProfile` of the lens galaxy on the left of the image, at coordinates (0.0", -1.0").
 2) Fit the `LightProfile` of the lens galaxy on the right of the image, at coordinates (0.0", 1.0").
 3) Use this lens-subtracted image to fit the source's light. The `MassProfile`'s of the two lens 
 galaxies are fixed to (0.0", -1.0") and (0.0", 1.0").
 4) Fit all relevant parameters simultaneously, using priors from searches 1, 2 and 3.
"""

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In search 1 we fit a lens model where:

 - The left lens galaxy's light is a parametric `EllipticalSersic` bulge [7 parameters].

 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

We fix the centre of its light to (0.0, -1.0), the pixel we know the left galaxy's light centre peaks.
"""
left_lens = af.Model(al.Galaxy, redshift=0.5, bulge=al.lp.EllipticalSersic)
left_lens.bulge.centre_0 = 0.0
left_lens.bulge.centre_1 = -1.0

model = af.Model(af.Collection(galaxies=af.Collection(left_lens=left_lens)))

analysis = al.AnalysisImaging(dataset=masked_imaging)

search = af.DynestyStatic(
    name="search[1]__left_lens_light[bulge]", n_live_points=30, evidence_tolerance=5.0
)

result_1 = search.fit(model=model, analysis=analysis)


"""
__Model + Search + Analysis + Model-Fit (Search 2)__

In search 2 we fit a lens model where:

 - The left lens galaxy's light is a parametric `EllipticalSersic` bulge [0 parameters: fixed from search 1].

 - The right lens galaxy's light is a parametric `EllipticalSersic` bulge [7 parameters].

 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=7.

We fix the centre of the right lens's light to (0.0, 1.0), the pixel we know the left galaxy's light centre peaks.

We also pass the result of the `left_lens` from search ` as an `instance`, which should improve the fitting of the
right lens.
"""
right_lens = af.Model(al.Galaxy, redshift=0.5, bulge=al.lp.EllipticalSersic)
right_lens.bulge.centre_0 = 0.0
right_lens.bulge.centre_1 = 1.0

model = af.Model(
    af.Collection(
        galaxies=af.Collection(
            left_lens=result_1.instance.galaxies.left_lens, right_lens=right_lens
        )
    )
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

search = af.DynestyStatic(
    name="search[2]__right_lens_light[bulge]", n_live_points=30, evidence_tolerance=5.0
)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

In search 3 we fit a lens model where:

 - The left lens galaxy's light is a parametric `EllipticalSersic` bulge [0 parameters: fixed from search 1].

 - The right lens galaxy's light is a parametric `EllipticalSersic` bulge [0 parameters: fixed from search 2].

 - The lens galaxy's mass is modeled using two `EllipticalIsothermal` profiles whose centres are fixed to (0.0, -1.0)
  and (0.0, 1.0) [6 parameters].
  
 - The source galaxy's light is a parametric `EllipticalExponential` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.
"""
left_lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=result_1.instance.galaxies.left_lens.bulge,
    mass=al.mp.EllipticalIsothermal,
)

right_lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=result_2.instance.galaxies.right_lens.bulge,
    mass=al.mp.EllipticalIsothermal,
)

left_lens.mass.centre_0 = 0.0
left_lens.mass.centre_1 = -1.0
right_lens.mass.centre_0 = 0.0
right_lens.mass.centre_1 = 1.0

model = af.Model(
    af.Collection(
        galaxies=af.Collection(
            left_lens=left_lens,
            right_lens=right_lens,
            source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllipticalExponential),
        )
    )
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

search = af.DynestyStatic(
    name="search[3]__mass_x2[sie]__source[exp]",
    n_live_points=50,
    evidence_tolerance=5.0,
)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

In search43 we fit a lens model where:

 - The left lens galaxy's light is a parametric `EllipticalSersic` bulge [7 parameters: priors initialized from search 1].

 - The right lens galaxy's light is a parametric `EllipticalSersic` bulge [7 parameters: priors initialized from search 2].

 - The lens galaxy's mass is modeled using two `EllipticalIsothermal` profiles whose centres are fixed to (0.0, -1.0)
  and (0.0, 1.0) [6 parameters: priors initialized from search 3].

 - The source galaxy's light is a parametric `EllipticalSersic` [7 parameters: priors initialized from search 3].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=27.
"""
left_lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=result_1.model.galaxies.left_lens.bulge,
    mass=result_3.model.galaxies.left_lens.mass,
)

right_lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=result_2.model.galaxies.right_lens.bulge,
    mass=result_3.model.galaxies.right_lens.mass,
)

source_bulge = af.Model(al.lp.EllipticalSersic)

source_bulge.take_attributes(result_3.model.galaxies.source.bulge)

model = af.Model(
    af.Collection(
        galaxies=af.Collection(
            left_lens=left_lens,
            right_lens=right_lens,
            source=af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge),
        )
    )
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

search = af.DynestyStatic(
    name="search[4]_light_x2[bulge]_mass_x2[sie]_source[exp]",
    n_live_points=60,
    evidence_tolerance=0.3,
)

result_4 = search.fit(model=model, analysis=analysis)

"""
And, we're done. This pipeline takes a while to run, as is the nature of multi-galaxy modeling. Nevertheless, 
the techniques we've learnt above can be applied to systems with even more galaxies, albeit the increases in parameters 
will slow down the non-linear search. Here are some more Q&A`s

 1) This system had two very similar lens galaxy's with comparable amounts of light and mass. How common is this? 
 Does it make it harder to model them?

Typically, a 2 galaxy system has 1 massive galaxy (that makes up some 80%-90% of the overall light and mass), 
accompanied by a smaller satellite. The satellite can`t be ignored, it impacts the ray-tracing in a measureable way, 
but this a lot less degenerate with the main lens galaxy. This means we can often model the satellite with much 
simpler profiles (e.g. spherical profiles). So yes, multi-galaxy systems can often be easier to model.

 2) It got pretty confusing passing all those priors towards the end of the pipeline there, didn`t it?

It does get confusing, I won't lie. This is why we made galaxies named objects, so that we could call them the 
`left_lens` and `right_lens`. It still requires caution when writing the pipeline, but goes to show that if you name 
your galaxies sensibly you should be able to avoid errors, or spot them quickly when you make them.
"""
