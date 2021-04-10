"""
Tutorial 8: Pipeline
====================

To illustrate lens modeling using an inversion this tutorial revists revisit the complex source model-fit that we
performed in tutorial 6 of chapter 3. This time, as you have probably guessed, we will fit the complex source using
an inversion.

We will use search chaining to do this, first fitting the source with a light profile, thereby initialize the mass
model priors and avoiding the unphysical solutions discussed in tutorial 6. In the later searches we will switch to
an `Inversion`.
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
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.6
)


imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()


"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In search 1 we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric `EllSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic),
    )
)

search = af.DynestyStatic(
    path_prefix=path.join("howtolens", "chapter_4"),
    name="search[1]_mass[sie]_source[parametric]",
    n_live_points=50,
)

analysis = al.AnalysisImaging(dataset=imaging)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear` [Parameters fixed to 
 results of search 1].

 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [2 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This search allows us to very efficiently set up the resolution of the pixelization and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.

Also, note how we can pass the `SettingsPixelization` object to an analysis class to customize if the border relocation
is used.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=result_1.instance.galaxies.lens.mass,
            shear=result_1.instance.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            pixelization=al.pix.VoronoiMagnification,
            regularization=al.reg.Constant,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path.join("howtolens", "chapter_4"),
    name="search[2]_mass[sie]_source[inversion_initialization]",
    n_live_points=20,
)

analysis = al.AnalysisImaging(
    dataset=imaging, settings_pixelization=al.SettingsPixelization(use_border=True)
)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search (Search 3)__

We use the results of searches 1 and 2 to create the lens model fitted in search 3, where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear` [7 parameters: priors 
 initialized from search 1].

 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [parameters fixed to results of search 2].

 - This pixelization is regularized using a `Constant` scheme [parameters fixed to results of search 2]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This search allows us to very efficiently set up the resolution of the pixelization and regularization coefficient 
of the regularization scheme, before using these models to refit the lens mass model.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=result_1.model.galaxies.lens.mass,
            shear=result_1.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            pixelization=result_2.instance.galaxies.source.pixelization,
            regularization=result_2.instance.galaxies.source.regularization,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path.join("howtolens", "chapter_4"),
    name="search[3]_mass[sie]_source[inversion]",
    n_live_points=50,
)

"""
__Positions + Analysis + Model-Fit (Search 3)__

The unphysical solutions that can occur in an `Inversion` can be mitigated by using a positions threshold to resample
mass models where the source's brightest lensed pixels do not trace close to one another. With search chaining, we can
in fact use the model-fit of a previous search (in this example, search 1) to compute the positions that we use in a 
later search.

Below, we use the results of the first search to compute the lensed source positions that are input into search 2. The
code below uses the  maximum log likelihood model mass model and source galaxy centre, to determine where the source
positions are located in the image-plane. 

We also use this result to set the `position_threshold`, whereby the threshold value is based on how close these 
positions trace to one another in the source-plane (using the best-fit mass model again). This threshold is multiplied 
by a `factor` to ensure it is not too small (and thus does not remove plausible mass  models). If, after this 
multiplication, the threshold is below the `minimum_threshold`, it is rounded up to this minimum value.
"""
settings_lens = al.SettingsLens(
    positions_threshold=result_2.positions_threshold_from(
        factor=3.0, minimum_threshold=0.2
    )
)

analysis = al.AnalysisImaging(
    dataset=imaging,
    positions=result_2.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

And with that, we now have a pipeline to model strong lenses using an inversion! 

Checkout the example pipelines in the `autolens_workspace/notebooks/chaining` package for inversion pipelines that 
includes the lens light component.
"""
