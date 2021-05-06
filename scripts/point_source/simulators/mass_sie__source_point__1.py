"""
Simulator: Point Source
=======================

This script simulates `Positions` data of a strong lens where:

 - The lens galaxy's total mass distribution is an `EllIsothermal`.
 - The source `Galaxy` is a `PointSource`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `Positions` data) and `dataset_name` 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/positions.json`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_label/dataset_name/noise_map.json`.
"""
dataset_type = "point_source"
dataset_name = "mass_sie__source_point__1"

"""
The path where the dataset will be output, which in this case is:
`/autolens_workspace/dataset/positions/mass_sie__source_point__1`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE) and source galaxy `PointSource` for this simulated lens. We include a 
faint dist in the source for purely visualization purposes to show where the multiple images appear.

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
        einstein_radius=1.2,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=90.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllExponential(
        centre=(0.05, 0.05), intensity=0.1, effective_radius=0.02
    ),
    point_0=al.ps.PointSource(centre=(0.05, 0.05)),
)

"""
Use these galaxies to setup a tracer, which will compute the multiple image positions of the simulated dataset.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
We will use a `PositionSolver` to locate the multiple images. 

We will use computationally slow but robust settings to ensure we accurately locate the image-plane positions.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PositionsSolver(
    grid=grid, use_upscaling=True, pixel_scale_precision=0.001, upscale_factor=2
)

"""
We now pass the `Tracer` to the solver. This will then find the image-plane coordinates that map directly to the
source-plane coordinate (0.0", 0.0").
"""
positions = solver.solve(
    lensing_obj=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

positions = al.Grid2DIrregular(
    grid=[
        positions.in_list[2],
        positions.in_list[9],
        positions.in_list[-3],
        positions.in_list[-1],
    ]
)

"""
Use the positions to compute the magnification of the `Tracer` at every position.
"""
magnifications = tracer.magnification_via_hessian_from_grid(grid=positions)

"""
We can now compute the observed fluxes of the `PointSource`, give we know how much each is magnified.
"""
flux = 1.0
fluxes = [flux * np.abs(magnification) for magnification in magnifications]
fluxes = al.ValuesIrregular(values=fluxes)

"""
We now output the image of this strong lens to `.fits` which can be used for visualize when performing point-source 
modeling and to `.png` for general inspection.
"""
visuals_2d = aplt.Visuals2D(multiple_images=positions)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, visuals_2d=visuals_2d)
tracer_plotter.figures_2d(image=True)

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=dataset_path, filename="image_2d", format="fits")
)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)
tracer_plotter.figures_2d(image=True)

mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)
tracer_plotter.subplot_tracer()

"""
Create a point-source dictionary data object and output this to a `.json` file, which is the format used to load and
analyse the dataset.
"""
point_source_dataset = al.PointSourceDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=positions.values_from_value(value=grid.pixel_scale),
    fluxes=fluxes,
    fluxes_noise_map=al.ValuesIrregular(values=[1.0, 1.0, 1.0, 1.0]),
)

point_source_dict = al.PointSourceDict(point_source_dataset_list=[point_source_dataset])

point_source_dict.output_to_json(
    file_path=path.join(dataset_path, "point_source_dict.json"), overwrite=True
)

"""
Pickle the `Tracer` in the dataset folder, ensuring the true `Tracer` is safely stored and available if we need to 
check how the dataset was simulated in the future. 

This will also be accessible via the `Aggregator` if a model-fit is performed using the dataset.
"""
tracer.save(file_path=dataset_path, filename="true_tracer")

"""
Finished.
"""
