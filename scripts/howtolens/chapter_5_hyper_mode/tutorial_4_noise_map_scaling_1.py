"""
Tutorial 4: Noise-Map Scaling 1
===============================

In tutorial 1, we discussed how when an inversion did not fit a compact source well we had skewed and undesirable
chi-squared distribution. A small subset of the lensed source's brightest pixels were fitted poorly, contributing
to the majority of our chi-squared signal. In terms of lens modeling, this meant that we would over-fit these regions
of the image. We would prefer that our lens model provides a global fit to the entire lensed source galaxy.

With our adaptive pixelization and regularization we are now able to fit the data to the noise-limit and remove this
skewed chi-squared distribution. So, why do we need to introduce noise-map scaling? Well, we achieve a good fit when
our lens's mass model is accurate (in the previous tutorials we used the *correct* lens mass model). But, what if our
lens mass model isn't accurate? We'll have residuals which will cause the same problem as before; a skewed chi-squared
distribution and an inability to fit the data to the noise level.

So, lets simulate an image and fit it with a slightly incorrect mass model.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

we'll use the same strong lensing data as the previous tutorial, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's `LightProfile` is an `EllSersic`.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=2,
    radius=3.0,
)

imaging = imaging.apply_mask(mask=mask)

"""
Next, we're going to fit the image using our magnification based grid. To perform the fit, we'll use a convenience 
function to fit the lens data we simulated above.

In this fitting function, we have changed the lens galaxy's einstein radius to 1.5 from the `true` simulated value of 
1.6. Thus, we are going to fit the data with an *incorrect* mass model.
"""


def fit_imaging_with_source_galaxy(imaging, source_galaxy):

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllIsothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.5,
            elliptical_comps=al.convert.elliptical_comps_from(
                axis_ratio=0.9, angle=45.0
            ),
        ),
        shear=al.mp.ExternalShear(elliptical_comps=(0.05, 0.05)),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(imaging=imaging, tracer=tracer)


"""
And now, we'll use the same magnification based source to fit this data.
"""
source_magnification = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=3.3),
)

fit = fit_imaging_with_source_galaxy(
    imaging=imaging, source_galaxy=source_magnification
)

include_2d = aplt.Include2D(mapper_data_pixelization_grid=True, mask=True)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()

inversion_plotter = fit_imaging_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d(reconstruction=True)

"""
__Hyper Image__

The fit isn't great. The main structure of the lensed source is reconstructed, but there are residuals. These 
residuals are worse than we saw in the previous tutorials (when source's compact central structure was the problem). 
So, the obvious question is can our adaptive pixelization and regularization schemes address the problem?

Lets find out, using this solution as our hyper-image. In this case, our hyper-image isn't a perfect fit to the data. 
This should not be too problematic, as the solution still captures the source's overall structure. The pixelization and 
regularization hyper parameters have enough flexibility in how they use this image to adapt themselves, thus hyper-image 
doesn`t *need* to be perfect.
"""
hyper_image = fit.model_image.binned.slim

"""
Note again that the source galaxy receives two types of hyper-images, a `hyper_galaxy_image` and a `hyper_model_image`. 
I'll discuss why in this tutorial.
"""
source_adaptive = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy_image=hyper_image,
    hyper_model_image=hyper_image,
)

fit = fit_imaging_with_source_galaxy(imaging=imaging, source_galaxy=source_adaptive)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()

inversion_plotter = fit_imaging_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d(reconstruction=True)

print("Evidence = ", fit.log_evidence)

"""
__Noise Scaling__

The solution is better, but far from perfect. Furthermore, this solution maximizes the Bayesian log evidence, meaning 
there is no reasonable way to change our source pixelization or regularization to better fit the data. The problem 
is with the lens's mass model!

This poses a major problem for model-fitting. A small subset of our data has such large chi-squared values the 
non-linear search is going to seek solutions which reduce only these chi-squared values. For the image above, a 
small subset of our data (e.g. < 5% of pixels) contributes to the majority of our log_likelihood (e.g. > 95% of the 
overall chi-squared). This is *not* what we want, as it means that instead of using the entire surface brightness 
profile of the lensed source galaxy to constrain our lens model, we end up using only a small subset of its brightest 
pixels.

This is even more problematic when we try and use the Bayesian evidence to objectively quantify the quality of the 
fit, as it cannot obtain a solution that provides a reduced chi-squared of 1 (e.g. that leaves only the Gaussian noise
in the image).

So, you're probably wondering, why can`t we just change the mass model to fit the data better? Surely if we 
actually modeled this image with **PyAutoLens** it wouldn't go to this solution anyway but instead infer the correct 
Einstein radius of 1.6? That`s true.

However, for *real* strong gravitational lenses, there is no such thing as a `correct mass model`. Real galaxies are 
not elliptical isothermal mass profiles, or power-laws, or NFW`s, or any of the symmetric and smooth analytic profiles 
we assume to model their mass. For real strong lenses our mass model will pretty much always lead to source 
reconstruction residuals, producing these skewed chi-squared distributions.

This is where noise-map scaling comes in. If we have no alternative, the best way to get a Gaussian distribution 
(e.g. more uniform) chi-squared fit is to increase the variances of image pixels with high chi-squared values. So, 
that`s what we're going to do, by making our source galaxy a `hyper-galaxy`, a galaxy which use`s its hyper-image to 
increase the noise in pixels where it has a large chi-squared value.
"""
source_hyper_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy=al.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.5, noise_power=1.0
    ),
    hyper_galaxy_image=hyper_image,
    hyper_model_image=hyper_image,
)

fit = fit_imaging_with_source_galaxy(imaging=imaging, source_galaxy=source_hyper_galaxy)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()

"""
As expected, the chi-squared distribution looks *alot* better. The chi-squareds have reduced from the 200's to the 
50's, because the variances were increased. This is what we want, so lets make sure we see an appropriate increase in 
Bayesian log evidence
"""
print("Evidence using baseline variances = ", 3885.2797)

print("Evidence using variances scaling by hyper-galaxy = ", fit.log_evidence)

"""
Yep, a huge increase in the 1000's! Clearly, if our model doesn't fit the data well we *need* to increase the noise 
wherever the fit is poor to ensure that our use of the Bayesian log evidence is well defined.

__How does the HyperGalaxy that we attached to the source-galaxy above actually scale the noise?__

First, it creates a `contribution_map` from the hyper-galaxy-image of the lensed source galaxy. This uses the 
`hyper_model_image`, which is the overall model-image of the best-fit lens model. In this tutorial, because our 
strong lens imaging only has a source galaxy emitting light, the `hyper_galaxy_image` of the source galaxy is the same 
as the `hyper_model_image`. However, In the next tutorial, we'll introduce the lens galaxy's light, such that each 
hyper-galaxy image is different to the hyper-galaxy model image!

We compute the contribution map as follows:

 1) Add the `contribution_factor` hyper-parameter value to the `hyper_model_image`.
  
 2) Divide the `hyper_galaxy_image` by the image created in step 1).
    
 3) Divide the image created in step 2) by its maximum value, such that all pixels range between 0.0 and 1.0.

Lets look at a few contribution maps, generated using hyper-galaxy's with different contribution factors.
"""
source_contribution_factor_1 = al.Galaxy(
    redshift=1.0,
    hyper_galaxy=al.HyperGalaxy(contribution_factor=1.0),
    hyper_galaxy_image=hyper_image,
    hyper_model_image=hyper_image,
)

contribution_map = source_contribution_factor_1.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image
)

mat_plot_2d = aplt.MatPlot2D(title=aplt.Title(label="Contribution Map"))

array_plotter = aplt.Array2DPlotter(array=contribution_map, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

source_contribution_factor_3 = al.Galaxy(
    redshift=1.0,
    hyper_galaxy=al.HyperGalaxy(contribution_factor=3.0),
    hyper_galaxy_image=hyper_image,
    hyper_model_image=hyper_image,
)

contribution_map = source_contribution_factor_3.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image
)

array_plotter = aplt.Array2DPlotter(array=contribution_map, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

source_hyper_galaxy = al.Galaxy(
    redshift=1.0,
    hyper_galaxy=al.HyperGalaxy(contribution_factor=5.0),
    hyper_galaxy_image=hyper_image,
    hyper_model_image=hyper_image,
)

contribution_map = source_hyper_galaxy.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image
)

array_plotter = aplt.Array2DPlotter(array=contribution_map, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

"""
By increasing the contribution factor we allocate more pixels with higher contributions (e.g. values closer to 1.0) 
than pixels with lower values. This is all the `contribution_factor` does; it scales how we allocate contributions to 
the source galaxy. Now, we're going to use this contribution map to scale the noise-map, as follows:

 1) Multiply the baseline (e.g. unscaled) noise-map of the image-data by the contribution map made in step 3) above. 
 This means that only noise-map values where the contribution map has large values (e.g. near 1.0) are going to 
 remain in this image, with the majority of values multiplied by contribution map values near 0.0.
    
 2) Raise the noise-map generated in step 1) above to the power of the hyper-parameter `noise_power`. Thus, for 
 large values of noise_power, the largest noise-map values will be increased even more, raising their noise the most.
    
 3) Multiply the noise-map values generated in step 2) by the hyper-parameter `noise_factor`. Again, this is a
 means by which **PyAutoLens** is able to scale the noise-map values.

Lets compare two fits, one where a hyper-galaxy scales the noise-map, and one where it does not.
"""
source_no_hyper_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy_image=hyper_image,
)

fit = fit_imaging_with_source_galaxy(
    imaging=imaging, source_galaxy=source_no_hyper_galaxy
)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()


print("Evidence using baseline variances = ", 3885.2797)

source_hyper_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy=al.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.5, noise_power=1.0
    ),
    hyper_galaxy_image=hyper_image,
    hyper_model_image=hyper_image,
)

fit = fit_imaging_with_source_galaxy(imaging=imaging, source_galaxy=source_hyper_galaxy)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()


print("Evidence using variances scaling by hyper-galaxy = ", fit.log_evidence)

"""
__Wrap Up__

Feel free to play around with the `noise_factor` and `noise_power` hyper-parameters above. It should be fairly 
clear what they do; they simply change the amount by which the noise is increased.

And with that, we've completed the first of two tutorials on noise-map scaling. To end, I want you to have a quick 
think, is there anything else that you can think of that would mean we need to scale the noise? In this tutorial, 
it was the inadequacy of our mass-model that lead to significant residuals and a skewed chi-squared distribution. 
What else might cause residuals? I'll give you a couple below;

 1) A mismatch between our model of the imaging data's Point Spread Function (PSF) and the true PSF of the telescope 
 optics of the data.
    
 2) Unaccounted for effects in our data-reduction of the image, in particular the correlated signals and noise arising
 during the data reduction.
    
 3) A sub-optimal background sky subtraction of the image, which can leave large levels of signal in the outskirts of 
 the image that are not due to the strong lens system itself.

Oh, there is on more thing that can cause much worse residuals than all the effects above. That'll be the topic of 
the next tutorial.
"""
