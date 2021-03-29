"""
Chaining: Hyper Mode
====================

Non-linear search chaining is an advanced model-fitting approach in **PyAutoLens** which breaks the model-fitting
procedure down into multiple non-linear searches, using the results of the initial searches to initialization parameter
sampling in subsequent searches. This contrasts the `modeling` examples which each compose and fit a single lens
model-fit using one non-linear search.

An overview of search chaining is provided in the `autolens_workspace/notebooks/imaging/chaining/api.py` script, make
sure to read that before reading this script!

This script introduces **PyAutoLens**'s hyper-mode, which passes the the results of previous model-fits performed by
earlier searches to searches performed later in the chain. This script illustrates two uses of hyper mode:

 - Using the `VoronoiBrightnessImage` pixelization and `AdaptiveBrightness` regularization scheme to adapt the source
 reconstruction to the source galaxy's morphology (as opposed to schemes introduced previously which adapt to the mass
 model magnification or apply a constant regularization pattern).

 - Using `HyperGalaxy`'s to scale the noise map, so as to down weight the fit to regions of an image the model is unable
 to fit accurately.

This script illustrates the API used for hyper-mode, but does not go into the details of how it works. This is described
in chapter 5 of the **HowToLens** lectures.
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
__Dataset + Masking + Positions__ 

Load, plot and mask the `Imaging` data.
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
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

imaging_plotter = aplt.ImagingPlotter(
    imaging=masked_imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = (path.join("imaging", "chaining", "api"),)

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit. The following options are 
available:

 - `hyper_galaxies`: whether the lens and / or source galaxy are treated as a hyper-galaxy, meaning that the model-fit
 can increase the noise-map values in the regions of the lens or source if they are poorly fitted.
 
 - `hyper_image_sky`: The background sky subtraction may be included in the model-fitting.

 - `hyper_background_noise`: The background noise-level may be included in the model-fitting.

The pixelization and regularization schemes which use hyper-mode to adapt to the source's properties are not passed into
`SetupHyper`, but are used in this example script below.

In this example, we only include the hyper galaxies, and because we are only fitting an image with a lensed source we
only include the hyper source galaxy..
"""
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=True,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__Model (Search 1)__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In the first
search our lens model is:

 - The lens galaxy's total mass distribution is an `EllipticalIsothermal` with `ExternalShear` [7 parameters].
 
 - The source-galaxy's light uses a `VoronoiBrightness` pixelization with fixed resolution 30 x 30 pixels (0 parameters).

 - This pixelization is regularized using a `Constant` scheme which smooths every source pixel equally [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant,
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]__hyper", n_live_points=50
)

analysis = al.AnalysisImaging(dataset=masked_imaging)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 2)__

In search 2, our source model now uses the `VoronoiBrightnessImage` pixelization and `AdaptiveBrightness` regularization
scheme that adapt to the source's unlensed morphology. These use the model-images of search 1, which is passed to the
`Analysis` class below. 

The source also includes a `HyperGalaxy` which can scale its noise if the model fit is poor.

We also use the results of search 1 to create the lens `Model` that we fit in search 2. This is described in the 
`api.py` chaining example.
"""
lens = result_1.model.galaxies.lens
source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
    hyper_galaxy=al.HyperGalaxy,
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.

When we create the analysis, we pass it a `hyper_result`, which is the result of search 1. This is telling the 
`Analysis` class to use the model-images of this fit to aid the fitting of the `VoronoiBrightnessImage` pixelization, 
`AdaptiveBrightness` regularization and source `HyperGalaxy`.

If you inspect and compare the results of searches 1 and 2, you'll note how the model-fits of search 2 have a much
higher likelihood than search 1 and how the source reconstruction has congregated it pixels to the bright central
regions of the source. This indicates that a much better result has been achieved, the reasons for which are discussed
in chapter 5 of the **HowToLens** lectures.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]__hyper", n_live_points=30
)

analysis = al.AnalysisImaging(dataset=masked_imaging, hyper_result=result_1)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Hyper Mode__
"""
