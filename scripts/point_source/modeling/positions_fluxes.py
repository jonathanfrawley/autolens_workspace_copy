"""
Modeling: Point-Source Position + Fluxes
========================================

In this script, we fit a `PointSourceDataset` with a strong lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source `Galaxy` is a `PointSource`.
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
__Dataset__

Load the strong lens dataset `mass_sie__source_point`, which is the dataset we will use to perform lens modeling.

We begin by loading an image of the dataset. Although we are performing point-source modeling and will not use this
data in the model-fit, it is useful to load it for visualization. By passing this dataset to the model-fit at the
end of the script it will be used when visualizing the results. However, the use of an image in this way is entirely
optional, and if it were not included in the model-fit visualization would simple be performed using grids without
the image.
"""
dataset_name = "mass_sie__source_point"
dataset_path = path.join("dataset", "point_source", dataset_name)

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"), pixel_scales=0.05
)

"""
We now load the positions we will fit using point source modeling. We load them as a `Grid2DIrregular` data 
structure, which groups different sets of positions to a common source. This is used, for example, when there are 
multiple source galaxy's in the source plane. For this simple example, we assume there is just one source and just one 
group.
"""
positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

print(positions.in_list)

"""
We also load the observed fluxes of the point source at every one of these position. We load them as 
a `ValuesIrregular` data  structure, which groups different sets of positions to a common source. This is used, 
for example, when there are  multiple source galaxy's in the source plane. For this simple example, we assume there 
is just one source and just one group.
"""
fluxes = al.ValuesIrregular.from_file(file_path=path.join(dataset_path, "fluxes.json"))

print(fluxes.in_list)

"""
We can now plot our positions dataset over the observed image.
"""
visuals_2d = aplt.Visuals2D(positions=positions)

array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)
array_plotter.figure_2d()

"""
We can also just plot the positions, omitting the image.
"""
grid_plotter = aplt.Grid2DPlotter(grid=positions)
grid_plotter.figure_2d()

"""
For point-source modeling, we also need the noise of every measured position. This is simply the pixel-scale of our
observed dataset, which in this case is 0.05".

The `position_noise_map` should have the same structure as the `Grid2DIrregular`. In this example, the positions
are a single group of 4 (y,x) coordinates, therefore their noise map should be a single group of 4 floats. We can
make this noise-map by creating a `ValuesIrregular` structure from the `Grid2DIrregular`.

We also create the noise map of fluxes, which for simplicity here I have entered manually.
"""
positions_noise_map = positions.values_from_value(value=image.pixel_scale)

print(positions_noise_map)

fluxes_noise_map = al.ValuesIrregular(values=[1.0, 2.0, 3.0, 4.0])

"""
__PointSourceDataset__

We next create a `PointSourceDataset` which contains the positions, fluxes and their noise-maps. 

It also names the the dataset. This `name` pairs the dataset to the `PointSource` in the model below. Specifically, 
because we name the dataset `point_0`, there must be a corresponding `PointSource` in the model below with the name 
`point_0` for the model-fit to be possible.

In this example, where there is just one source, named pairing appears uncessary. However, point-source datasets may
have many source galaxies in them, and name pairing ensures every point source in the model is compared against its
point source dataset.
"""
point_source_dataset = al.PointSourceDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=positions_noise_map,
    fluxes=fluxes,
    fluxes_noise_map=fluxes_noise_map,
)

"""
We now create the `PointSourceDict`, which is a dictionary of every `PointSourceDataset`. Again, because we only have 
one dataset the use of this class seems unecessary, but it is important for model-fits containing many point sources.
"""
point_source_dict = al.PointSourceDict(point_source_dataset_list=[point_source_dataset])

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear` [7 parameters].
 - The source galaxy's light is a parametric `EllSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.

NOTE: 

**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/notebooks/preprocess`). 
 - Manually override the lens model priors (`autolens_workspace/notebooks/imaging/modeling/customize/priors.py`).
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal)
source = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.PointSourceFlux)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__PositionsSolver__

For point-source modeling we also need to define our `PositionsSolver`. This object determines the multiple-images of 
a mass model for a point source at location (y,x) in the source plane, by iteratively ray-tracing light rays to the 
source-plane. 

Checkout the script ? for a complete description of this object, we will use the default `PositionSolver` in this 
exampl
"""
grid = al.Grid2D.uniform(
    shape_native=image.shape_native, pixel_scales=image.pixel_scales
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.02)

"""
__Search__

The lens model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/).

The folder `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches` gives an overview of the 
non-linear searches **PyAutoLens** supports. If you are unclear of what a non-linear search is, checkout chapter 2 of 
the **HowToLens** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/imaging/mass_sie__source_sersic/mass[sie]_source[bulge]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("point_source", dataset_name),
    name="mass[sie]_source[point_flux]",
    n_live_points=50,
)

"""
__Analysis__

The `AnalysisPointSource` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `PointSourceDataset`.
"""
analysis = al.AnalysisPointSource(
    point_source_dict=point_source_dict, solver=positions_solver
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autolens_workspace/output/point_source/mass_sie__source_point/mass[sie]_source[point_flux]` for 
live outputs  of the results of the fit, including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The lens model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` object.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

"""
Checkout `autolens_workspace/notebooks/modeling/results.py` for a full description of the result object.
"""
