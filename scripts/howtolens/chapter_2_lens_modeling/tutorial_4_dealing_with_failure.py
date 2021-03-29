"""
Tutorial 4: Dealing With Failure
================================

In the last tutorial we showed how our non-linear search can potentially fail, and infer a local maxima solution.
In this tutorial, we're going to learn how to stop this and infer the globally maximal lens model.

In the previous tutorial, when we inferred a local maxima we knew that we had done so. For modeling a real lens,
we do not know the *true* lens model and it is often  unclear if a solution is a global or local maximma. The tricks
we learn in this tutorial are therefore equally important for verifying that a solution does indeed appear to be the
global maxima.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autolens as al
import autolens.plot as aplt
import autofit as af

"""
we'll use the same strong lensing data as the previous tutorial, where:

 - The lens galaxy's light is an `EllipticalSersic`.
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` and `ExternalShear`.
 - The source galaxy's `LightProfile` is an `EllipticalExponential`.
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
we'll create and use a smaller 2.0" `Mask2D` again.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=2.6
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

"""
When plotted, the lens light`s is clearly visible in the centre of the image.
"""
imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Approach 1: Prior Tuning__

First, we're going to try giving our non-linear search a helping hand. Lets think about our priors, they`re telling 
the non-linear search where to look in parameter space. If we tell it to look in the right place (that is, 
*tune* our priors), it might find the best-fit lens model.

We've already seen that we can fully customize priors in **PyAutoLens**, so lets do it. I've set up a custom search 
below and specified a new set of priors that'll give the non-linear search a much better chance if inferring the 
correct model. I've also let you know what we're changing the priors from (as initially specified by the 
`config/priors/default` config files.)

we'll call our lens and source galaxies `lens` and `source` this time, for shorter more readable code.

In a later tutorial, we'll cover non-linear search approaches that are different to Dynesty, where one provides the
non-linear search with a *starting point* where it samples parameter space. In a similar fashion to prior tuning,
giving these searches a good starting point will increase the chances of us finding the global maxima.

In the previous tutorial, we customized the priors of a model by creating a `Galaxy` as a `Model` and customizing each
prior:
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.SphericalIsothermal)
lens.mass.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

"""
We can alternative create the light and mass profiles as a `Model` and customize their parameters, before then
creating the overall model. These two approaches are equivalent, but this does provide a more concise and readable API.
We will therefore switch to this API in this tutorial, but may swap back and forth between the two depending on what is
more readable for the specific example.
"""
bulge = af.Model(al.lp.EllipticalSersic)
mass = af.Model(al.mp.EllipticalIsothermal)

"""
By default, the prior on the $(y,x)$ coordinates of a `LightProfile` / `MassProfile` is a GaussianPrior with mean 0.0" and 
sigma "1.0. However, visual inspection of our strong lens image tells us that its clearly around x = 0.0" and y = 0.0", 
so lets reduce where non-linear search looks for these parameters.
"""
bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)

"""
By default, the elliptical components of the of our lens galaxy's elliptical `LightProfile` are `UniformPriors` 
between -1.0 and 1.0, corresponding to the full range of possible ellipses with axis-ratio from 0.0 to 1.0 and 
position angles from 0.0 to 180.0 degrees.

However, looking close to the image it is clear that the lens galaxy's light is elliptical and oriented around 
45.0 degrees counter-clockwise from the x-axis. We can update the priors on our elliptical components to reflect this.
"""
bulge.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(
    mean=0.333333, sigma=0.1, lower_limit=-1.0, upper_limit=1.0
)
bulge.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(
    mean=0.0, sigma=0.1, lower_limit=-1.0, upper_limit=1.0
)

"""
Lets additionally assume that the `LightProfile`'s ellipticity informs us of the `MassProfile`'s ellipticity. Because 
this may not strictly be true (e.g. because of dark matter) we'll use a wider prior.
"""
mass.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(
    mean=0.333333, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
)
mass.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(
    mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
)

"""
The effective radius of a `LightProfile` is its `half-light` radius, the radius at which 50% of its total luminosity 
is internal to a circle defined within that radius. **PyAutoLens** assumes a UniformPrior on this quantity between 0.0" and 
4.0", but inspection of the image (again, using a colormap scaling) shows the lens's light doesn`t extend anywhere near 
4.0", so lets reduce it.
"""
bulge.effective_radius = af.GaussianPrior(
    mean=1.0, sigma=0.8, lower_limit=0.0, upper_limit=np.inf
)

"""
Typically, we have knowledge of our lens galaxy's morphology. Most strong lenses are massive elliptical galaxies which 
have Sersic indexes near 4. So lets change our Sersic index from a UniformPrior between 0.8 and 8.0 to reflect this.
"""
bulge.sersic_index = af.GaussianPrior(
    mean=4.0, sigma=1.0, lower_limit=0.0, upper_limit=np.inf
)

"""
Finally, the `ring` that the lensed source forms clearly has a radius of about 1.6". This is its Einstein radius, so 
lets change the prior from a UniformPrior between 0.0" and 4.0".
"""
mass.einstein_radius = af.GaussianPrior(
    mean=1.6, sigma=0.2, lower_limit=0.0, upper_limit=np.inf
)

"""
We now compose the overall model, where the lens galaxy model uses the `Model` components above which had their
priors customizes.

In this exercise, I'm not going to change any priors on the source galaxy. Whilst lens modeling experts can look at a 
strong lens and often tell you roughly where the source-galaxy is located (in the source-plane), it is something of art 
form. Furthermore, the source's morphology can be pretty complex, making it difficult to come up with a good source prior!
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp.EllipticalSersic,
    mass=al.mp.EllipticalIsothermal,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllipticalExponential)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
We can now create this custom search and run it. Our non-linear search will now start by sampling higher likelihood 
regions of parameter space, given our improved and more informed priors.
"""
search = af.DynestyStatic(
    path_prefix="howtolens", name="tutorial_4_custom_priors", n_live_points=50
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

print(
    "Dynesty has begun running - checkout the workspace/output/tutorial_4_custom_priors"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once Dynesty has completed - this could take some time!"
)

result_custom_priors = search.fit(model=model, analysis=analysis)

print("Dynesty has finished run - you may now continue the notebook.")

"""
Bam! We get a good model, which indeed corresponds to the global maxima. By giving our non-linear search a helping hand
and informing it of where to sample parameter space, we can increase the odds that we find the global maxima solution.
"""
fit_imaging_plotter = aplt.FitImagingPlotter(
    fit=result_custom_priors.max_log_likelihood_fit
)
fit_imaging_plotter.subplot_fit_imaging()

"""
By tuning our priors to the lens we're fitting we can increase our chance of inferring the global maxima lens model. 
Before moving onto the next approach, lets think about the advantages and disadvantages of prior tuning:

Advantages: 

 - We find the maximum log likelihood solution in parameter space.
 - The search took less time to run because the non-linear search explored less of parameter space.

Disadvantages: 

 - If we specified a prior incorrectly the non-linear search would begin and therefore end at an incorrect solution.
 - Our search was tailored to this specific strong lens. If we want to fit a large sample of lenses we`d 
 have to write a custom search for every single one - this would take up a lot of our time!

__Approach 2: Reducing Complexity__

Previously, Our non-linear searched failed because we made the lens model more complex. Can we can make it less complex, 
whilst still keeping it fairly realistic? Maybe there are some assumptions we can make to reduce the number of 
lens model parameters and therefore dimensionality of non-linear parameter space?

Well, we can *always* make assumptions. Below, I'm going to create a search that assumes that light-traces-mass. That 
is, that our `LightProfile`'s centre, and elliptical components are perfectly aligned with its mass. This may, or may 
not, be a reasonable assumption, but it`ll remove 4 parameters from the lens model (the `MassProfile`'s y, x, and 
elliptical components), so its worth trying!
"""
bulge = af.Model(al.lp.EllipticalSersic)
mass = af.Model(al.mp.EllipticalIsothermal)

"""
In the pass priors function we can `pair` any two parameters by setting them equal to one another. This removes the 
parameter on the left-hand side of the pairing from the lens model such that is always assumes the same value as the 
parameter on the right-hand side.
"""
mass.centre = bulge.centre

"""
Lets do this with the elliptical components of the light and mass profiles.
"""
mass.elliptical_comps = bulge.elliptical_comps

"""
Compose the model using the bulge and mass `Model` components that now have their centre and elliptical components
tied together.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp.EllipticalSersic,
    mass=al.mp.EllipticalIsothermal,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllipticalExponential)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
Again, we create this search and run it. The non-linear search now has a less complex parameter space to search.
"""
search = af.DynestyStatic(
    path_prefix="howtolens", name="tutorial_4_light_traces_mass", n_live_points=40
)

print(
    "Dynesty has begun running - checkout the workspace/output/tutorial_4_dealing_with_failure"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once Dynesty has completed - this could take some time!"
)

result_light_trace_mass = search.fit(model=model, analysis=analysis)

print("Dynesty has finished run - you may now continue the notebook.")

fit_imaging_plotter = aplt.FitImagingPlotter(
    fit=result_light_trace_mass.max_log_likelihood_fit
)
fit_imaging_plotter.subplot_fit_imaging()

"""
The results look pretty good. Our source galaxy fits the data pretty well and we've clearly inferred a model that 
looks similar to the one above. However, inspection of the residuals shows that the fit wasn`t quite as good as the 
custom-search above.

It turns out that when I simulated this image light didn`t perfectly trace mass. The `LightProfile`'s elliptical 
components were (0.333333, 0.0) whereas the `MassProfile`'s were (0.25, 0.0). The quality of the fit has suffered as a 
result and the log likelihood we inferred is lower.

Herein lies the pitfalls of making assumptions - they may make your model less realistic and your fits worse! 

Again, lets consider the advantages and disadvantages of this approach:

Advantages:

 - By reducing parameter space`s complexity we inferred a global maximum log likelihood.
 - The search is not specific to one lens - we could run it on many strong lens images.
    
Disadvantages:

 - Our model was less realistic and our fit suffered as a result.

__Approach 3: Look Harder__

In approaches 1 and 2 we extended our non-linear search an olive branch and helped it find the highest log likelihood 
regions of parameter space. In approach 3 ,we're going to tell it to just `look harder`.

Basically, every non-linear search has a set of parameters that govern how thoroughly it searches parameter 
space. The more thoroughly it looks, the more likely it is that it`ll find the global maximum lens model. However, 
the search will also take longer - and we don't want it to take too long to get us a result!

In tutorial 7, we'll discuss non-linear searches in more detail, so we'll defer a detailed discussion of setting up
the non-linear searches until then.

lets list the advantages and disadvantages of simply adjusting the non-linear search:

Advantages:

 - Its easy to setup, we simply change parameter of the non-linear search like n_live_points.
    
 - It generalizes to any strong lens.
 
 - We didn`t have to make our model less realistic.

Disadvantage:
 
 - Its potentially expensive. Very expensive. For very complex models, the run times can begin to take hours, days, 
 weeks or, dare I say it, months!

So, we can now fit strong lenses with **PyAutoLens**. And when it fails, we know how to get it to work. I hope you're 
feeling pretty smug. You might even be thinking `why should I bother with the rest of these tutorials, if I can fit 
strong a lens already`.

Well, my friend, I want you to think about the last disadvantage listed above. If modeling a single lens could really 
take as long as a month, are you really willing to spend your valuable time waiting for this? I'm not, which is why I 
developed **PyAutoLens**. 

In chapter 3 of **HowToLens** we'll introduce a concept called `search chaining`, which will give us the best of all 3
approaches demonstrated in this tutorial: realistic, complex lens model that take mere hours to fit! However, to round 
out chapter 2, we'll look at more detail how a non-linear search works, considering how to approach masking our
dataset as well as consider what the results of lens model-fit actually tell us.
"""
