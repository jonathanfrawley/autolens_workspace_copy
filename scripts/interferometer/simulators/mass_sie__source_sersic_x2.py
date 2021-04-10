"""
Simulator: SIE
==============

This script simulates `Interferometer` data of a strong lens where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's `LightProfile` is an `EllSersic`.
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
The `dataset_type` describes the type of data being simulated (in this case, `Interferometer` data) and `dataset_name` 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/psf.fits`.
"""
dataset_type = "interferometer"
dataset_name = "mass_sie__source_sersic_x2"

"""
The path where the dataset will be output, which in this case is
`/autolens_workspace/dataset/interferometer/mass_sie__source_sersic_x2`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
For simulating an image of a strong lens, we recommend using a Grid2DIterate object. This represents a grid of (y,x) 
coordinates like an ordinary Grid2D, but when the light-profile`s image is evaluated below (using the Tracer) the 
sub-size of the grid is iteratively increased (in steps of 2, 4, 8, 16, 24) until the input fractional accuracy of 
99.99% is met.

This ensures that the divergent and bright central regions of the source galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""
grid = al.Grid2DIterate.uniform(
    shape_native=(151, 151), pixel_scales=0.1, fractional_accuracy=0.9999
)

"""
To perform the Fourier transform we need the wavelengths of the baselines, which we'll load from the fits file below.

By default we use baselines from the Square Mile Array (SMA), which produces low resolution interferometer data that
can be fitted extremely efficiently. The `autolens_workspace` includes ALMA uv_wavelengths files for simulating
much high resolution datasets (which can be performed by replacing "sma.fits" below with "alma.fits").
"""
uv_wavelengths_path = path.join("dataset", dataset_type, "uv_wavelengths")
uv_wavelengths = al.util.array_1d.numpy_array_1d_from_fits(
    file_path=path.join(uv_wavelengths_path, "sma.fits"), hdu=0
)

"""
To simulate the interferometer dataset we first create a simulator, which defines the exposure time, noise levels 
and Fourier transform method used in the simulation.
"""
simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    background_sky_level=0.1,
    noise_sigma=0.1,
    transformer_class=al.TransformerNUFFT,
)

"""
Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.

For lens modeling, defining ellipticity in terms of the `elliptical_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the **PyAutoLens** `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.05, 0.05)),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllSersic(
        centre=(0.25, 0.15),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=0.7,
        effective_radius=0.7,
        sersic_index=1.0,
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllSersic(
        centre=(0.7, -0.5),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=60.0),
        intensity=0.2,
        effective_radius=1.6,
        sersic_index=3.0,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated interferometer dataset.
"""
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
)

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
interferometer dataset.
"""
interferometer = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

"""
Lets plot the simulated interferometer dataset before we output it to fits.
"""
interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.subplot_interferometer()

"""
Output the simulated dataset to the dataset path as .fits files.
"""
interferometer.output_to_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    overwrite=True,
)

"""
Output a subplot of the simulated dataset, the image and a subplot of the `Tracer`'s quantities to the dataset path 
as .png files.
"""
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.subplot_interferometer()

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)
tracer_plotter.subplot_tracer()

"""
Pickle the `Tracer` in the dataset folder, ensuring the true `Tracer` is safely stored and available if we need to 
check how the dataset was simulated in the future. 

This will also be accessible via the `Aggregator` if a model-fit is performed using the dataset.
"""
tracer.save(file_path=dataset_path, filename="true_tracer")

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/no_lens_light/mass_sie__source_sersic_x2`.
"""
