"""
Modeling: Point-Source Mass Total
=================================

In this script, we fit a `PointSourceDict` with a strong lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal`.
 - The source `Galaxy` is a point source `PointSource`.

The point-source dataset used in this example includes the flux of every point-source multiple image. However we omit
the fluxes from the fit and the lens model (by using a `PointSource` model instead of a `PointSourceFlux`). We make
this choice because most strong lens models of point sources it is common practise to omit flux information from the
model-fit. Changing the point source model `PointSourceFlux` will therefore use this flux information.

The `ExternalShear` is also not included in the mass model, where it is for the `imaging` and `interferometer` examples.
For a quadruply imaged point source (8 data points) there is insufficient information to fully constain a model with
an `EllIsothermal` and `ExternalShear` (9 parameters).
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
dataset_name = "mass_sie__source_point__0"
dataset_path = path.join("dataset", "point_source", dataset_name)

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"), pixel_scales=0.05
)

"""
We now load the point source dataset we will fit using point source modeling. We load this data as a `PointSourceDict`,
which is a Python dictionary containing the positions and fluxes of every point source. 

In this example there is just one point source, but point source model can be applied to datasets with any number 
of source's.
"""
point_source_dict = al.PointSourceDict.from_json(
    file_path=path.join(dataset_path, "point_source_dict.json")
)

"""
We can print the `positions` and `fluxes` of this dataset, as well as their noise-map values.
"""
print("Point Source Dataset Name:")
print(point_source_dict["point_0"].name)
print("Point Source Multiple Image (y,x) Arc-second Coordinates:")
print(point_source_dict["point_0"].positions.in_list)
print("Point Source Multiple Image Noise-map Values:")
print(point_source_dict["point_0"].positions_noise_map.in_list)
print("Point Source Flux Values:")
print(point_source_dict["point_0"].fluxes.in_list)
print("Point Source Flux Noise-map Values:")
print(point_source_dict["point_0"].fluxes_noise_map.in_list)

"""
We can plot our positions dataset over the observed image.
"""
visuals_2d = aplt.Visuals2D(positions=point_source_dict.positions_list)

array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)
array_plotter.figure_2d()

"""
We can also just plot the positions, omitting the image.
"""
grid_plotter = aplt.Grid2DPlotter(grid=point_source_dict["point_0"].positions)
grid_plotter.figure_2d()

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` [5 parameters].
 - The source galaxy's light is a point `PointSource` [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.

NOTE: 

Every point-source dataset in the `PointSourceDict` has a name. Its `name` pairs the dataset to the `PointSource` 
in the model below. Specifically, because the name of the dataset is `point_0`, there must be a corresponding
`PointSource` model component in the model below with the name `point_0` for the model-fit to be possible.

In this example, where there is just one source, named pairing appears unecessary. However, point-source datasets may
have many source galaxies in them, and name pairing ensures every point source in the model is compared against its
point source dataset.

**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/notebooks/preprocess`). 
 - Manually override the lens model priors (`autolens_workspace/notebooks/imaging/modeling/customize/priors.py`).
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal)
source = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.PointSource)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__PositionsSolver__

For point-source modeling we also need to define our `PositionsSolver`. This object determines the multiple-images of 
a mass model for a point source at location (y,x) in the source plane, by iteratively ray-tracing light rays to the 
source-plane. 

Checkout the script ? for a complete description of this object, we will use the default `PositionSolver` in this 
example with a `point_scale_precision` half the value of the position noise-map, which should be sufficiently good 
enough precision to fit the lens model accurately.
"""
grid = al.Grid2D.uniform(
    shape_native=image.shape_native, pixel_scales=image.pixel_scales
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.025)

"""
__Search__

The lens model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/).

The folder `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches` gives an overview of the 
non-linear searches **PyAutoLens** supports. If you are unclear of what a non-linear search is, checkout chapter 2 of 
the **HowToLens** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/ppoint_source/mass_sie__source_sersic/mass[sie]_source[point]/unique_identifier`.
 
__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model, search and dataset generates the same identifier, meaning that rerunning the
script will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset,
a new unique identifier will be generated, ensuring that the model-fit results are output into a separate folder. 
"""
search = af.DynestyStatic(
    path_prefix=path.join("point_source"),
    name="mass[sie]_source[point]",
    unique_tag=dataset_name,
    nlive=50,
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

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
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
