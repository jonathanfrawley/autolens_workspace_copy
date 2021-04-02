"""
Tutorial 8: Fitting
===================

In this example, we'll fit the `Imaging` data we simulated in the previous exercise. we'll do this using model images
generated via a `Tracer`, and by comparing to the simulated image we'll get diagnostics about the quality of the fit.
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
The `dataset_path` specifies where the data was output in the last tutorial, which is the directory 
`autolens_workspace/dataset/imaging/no_lens_light/howtolens/`.
"""
dataset_path = path.join("dataset", "imaging", "no_lens_light", "howtolens")

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

"""
The `imaging` is an `Imaging` object, which is a package of all components of the dataset, in particular:

 1) The image.
 2) The Point Spread Function (PSF).
 3) Its noise-map.
    
Which are all stored as `Array2D` objects.
"""
print("Image:")
print(imaging.image)
print("Noise-Map:")
print(imaging.noise_map)
print("PSF:")
print(imaging.psf)

"""
To fit an image, we first specify a `Mask2D`, which describes the sections of the image that we fit.

Typically, we want to mask regions of the image where the lens and source galaxies are not visible, for example at 
the edges where the signal is entirely background sky and noise.

For the image we simulated, a 3" circular `Mask2D` will do the job.

A `Mask2D` also takes the `sub_size` parameter we are used to giving a grid. This does what it does for a `Grid2D`, 
defining the sub-grid used to calculate lensing quantities.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=1,
    radius=3.0,
)

print(mask)  # 1 = True, which means the pixel is masked. Edge pixels are indeed masked.
print(mask[48:53, 48:53])  # Whereas central pixels are `False` and therefore unmasked.

"""
We can use an `ImagingPlotter`.to compare the mask and the image - this is useful if we really want to `tailor` a 
mask to the lensed source's light (which in this example, we won't).

However, the mask is not an attribute of the `Imaging` object. Thus, we cannot use `Include2D(mask=True)` to plot it, 
as the `Imaging` doesn't know what the mask is!

To manually plot an object over the figure of another object, we can pass it to the `Visuals2D` object and then use
this in the `ImagingPlotter`. Note that the `Visuals2D` object can be used to customize the appearance of *any* figure
in PyAutoLens and is therefore a powerful means by which to create custom visuals!
"""
visuals_2d = aplt.Visuals2D(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging, visuals_2d=visuals_2d)
imaging_plotter.figures(image=True)

"""
To fit the data we create a `Imaging` object, which is a `package` of all parts of a data-set we need in order 
to fit it with a lens model:

 1) The imaging-data, including the image, PSF (so that when we compare a tracer`s image to the image instrument we 
 can include blurring due to the telescope optics) and noise-map (so our goodness-of-fit measure accounts for 
 noise in the observations).

 2) The mask, so that only the regions of the image with a signal are fitted.

 3) A `Grid2D` aligned to the `Imaging` data's pixels, so the tracer`s image is generated on the same (masked) `Grid2D` 
 as the image.
"""
masked_imaging = imaging.apply_mask(mask=mask)

"""
Note that because the `Mask2D` is now an attribute of the `Imaging` we can plot it using `Include2D`.

Because it is an attribute, the `mask` now also automatically `zooms` our plot around the masked region only. This 
means that if our image is very large, we focus-in on the lens and source galaxies.
"""
include_2d = aplt.Include2D(mask=True)

imaging_plotter = aplt.ImagingPlotter(imaging=masked_imaging, include_2d=include_2d)
imaging_plotter.figures(image=True)

"""
By printing its attributes, we can see that it does indeed contain the mask, masked image, masked noise-map, psf and so 
on.
"""
print("Mask2D")
print(masked_imaging.mask)
print()
print("Masked Image:")
print(masked_imaging.image)
print()
print("Masked Noise-Map:")
print(masked_imaging.noise_map)
print()
print("PSF:")
print(masked_imaging.psf)
print()

"""
This image and noise-map are again stored in 2D and 1D. 

However, the 1D array now corresponds only to the pixels that were not masked, whereas for the 2D array, all edge 
values are masked and are therefore zeros.
"""
print("The 2D Masked Image and 1D Image of unmasked entries")
print(masked_imaging.image.shape_native)
print(masked_imaging.image.shape_slim)
print(masked_imaging.image.native)
print(masked_imaging.image.slim)
print()
print("The 2D Masked Noise-Map and 1D Noise-Map of unmasked entries")
print(masked_imaging.noise_map.shape_native)
print(masked_imaging.noise_map.shape_slim)
print(masked_imaging.noise_map.native)
print(masked_imaging.noise_map.slim)

"""
The masked dataset also has a `Grid2D`, where only coordinates which are not masked are included (the masked 2D values 
are set to [0.0. 0.0]).
"""
print("Masked Grid2D")
print(masked_imaging.grid.native)
print(masked_imaging.grid.slim)

"""
To fit an image, create an image using a `Tracer`. Lets use the same `Tracer` we simulated the `Imaging` instrument 
with (thus, our fit is `perfect`).

Its worth noting that below, we use the `Imaging`'s `Grid2D` to setup the `Tracer`. This ensures that our 
image-plane image is the same resolution and alignment as our lens data's masked image.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=masked_imaging.grid)
tracer_plotter.figures(image=True)

"""
To fit the image, we pass the `Imaging` and `Tracer` to a `FitImaging` object. This performs the following:

 1) Blurs the tracer`s image with the lens data's PSF, ensuring the telescope optics are included in the fit. This 
 creates the fit`s `model_image`.

 2) Computes the difference between this model_image and the observed image-data, creating the fit`s `residual_map`.

 3) Divides the residual-map by the noise-map, creating the fit`s `normalized_residual_map`.

 4) Squares every value in the normalized residual-map, creating the fit`s `chi_squared_map`.

 5) Sums up these chi-squared values and converts them to a `log_likelihood`, which quantifies how good the tracer`s 
 fit to the data was (higher log_likelihood = better fit).
"""
fit = al.FitImaging(imaging=masked_imaging, tracer=tracer)

include_2d = aplt.Include2D(mask=True)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()

"""
We can print the fit`s attributes. As usual, we can choose whether to return the fits in slim or native format, with
the native data's edge values all zeros, as the edges were masked:
"""
print("Model-Image:")
print(fit.model_image.native)
print(fit.model_image.slim)
print()
print("Residual Maps:")
print(fit.residual_map.native)
print(fit.residual_map.slim)
print()
print("Chi-Squareds Maps:")
print(fit.chi_squared_map.native)
print(fit.chi_squared_map.slim)

"""
Of course, the central unmasked pixels have non-zero values.
"""
model_image = fit.model_image.native
print(model_image[48:53, 48:53])
print()

residual_map = fit.residual_map.native
print("Residuals Central Pixels:")
print(residual_map[48:53, 48:53])
print()

print("Chi-Squareds Central Pixels:")
chi_squared_map = fit.chi_squared_map.native
print(chi_squared_map[48:53, 48:53])

"""
The fit also gives a log likelihood, which is a single-figure estimate of how good the model image fitted the simulated 
image (in unmasked pixels only!).
"""
print("Likelihood:")
print(fit.log_likelihood)

"""
We can customize the `Imaging` we set up, using the `SettingsImaging` object. 

For example, we can: 

 - Specify the `Grid2D` used by the `Imaging` to fit the data, where we below increase it from its default 
 value of 2 to 4.
"""
settings_masked_imaging = al.SettingsImaging(grid_class=al.Grid2D, sub_size=4)

masked_imaging_custom = al.Imaging(
    imaging=imaging, mask=mask, settings=settings_masked_imaging
)

"""
The use of `Settings` objects is a core feature of the **PyAutoLens** API and will appear throughout the **HowToLens**
chapters for setting up many different aspects of a **PyAutoLens** fit, so take note!

We used the same `Tracer` to create and fit the image, giving an excellent fit. The residual-map and chi-squared-map, 
show no signs of the source-`Galaxy`'s light present, indicating a good fit. This solution will translate to one of the 
highest-log_likelihood solutions possible.

Lets change the `Tracer`, so that it`s near the correct solution, but slightly off. Below, we slightly offset the lens 
galaxy, by 0.005"
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.005, 0.005), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(imaging=masked_imaging, tracer=tracer)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()

"""
Residuals now appear at the locations of the source galaxy, increasing the chi-squared values (which determine 
our log_likelihood).

Lets compare the log likelihood to the value we computed above (which was 4372.90):
"""
print("Previous Likelihood:")
print(2967.0488)
print("New Likelihood:")
print(fit.log_likelihood)

"""
It decreases! As expected, this model is a worse fit to the data.

Lets change the `Tracer`, one more time, to a solution nowhere near the correct one.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.005, 0.005),
        einstein_radius=1.5,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, phi=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllipticalSersic(
        centre=(0.2, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=60.0),
        intensity=0.5,
        effective_radius=0.8,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(imaging=masked_imaging, tracer=tracer)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()

"""
Clearly, the model provides a terrible fit and this `Tracer` is not a plausible representation of the `Imaging` dataset
(of course, we already knew that, given that we simulated it!)

The log likelihood drops dramatically, as expected.
"""
print("Previous Likelihoods:")
print(2967.0488)
print(2687.4724)
print("New Likelihood:")
print(fit.log_likelihood)

"""
Congratulations, you`ve fitted your first strong lens with **PyAutoLens**! Perform the following exercises:

 1) In this example, we `knew` the correct solution, because we simulated the lens ourselves. In the real Universe, 
 we have no idea what the correct solution is. How would you go about finding the correct solution? Could you find a 
 solution that fits the data reasonable through trial and error?
"""
