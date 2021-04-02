"""
Database: Introduction
======================

The default behaviour of **PyAutoLens** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check. For small model-fitting tasks this is sufficient, however many users 
have a need to perform many model fits to large sampels of lenses, making manual inspection of results time consuming.

PyAutoLens's database feature outputs all model-fitting results as a
sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database, such that all results
can be efficiently loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation. This
database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can be
loaded.

This script fits a sample of three simulated strong lenses using the same non-linear search. The results will be used
to illustrate the database in the database tutorials that follow.

The search fits each lens with:
 
 - An `EllipticalIsothermal` `MassProfile` for the lens galaxy's mass.
 - An `EllipticalSersic` `LightProfile` for the source galaxy's light.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
from os import path
import autofit as af
import autolens as al

"""
For each dataset we load it from hard-disc, set up its `Analysis` class and fit it with a non-linear search. 

The 3 datasets are in the `autolens_workspace/dataset/database` folder.

We want each results to be stored in the database with an entry specific to the dataset. We'll use the `Dataset`'s name 
string to do this, so lets create a list of the 3 dataset names.
"""
dataset_names = [
    "mass_sie__source_sersic__0",
    "mass_sie__source_sersic__1",
    "mass_sie__source_sersic__2",
]

"""
Specify the dataset type, label and name, which we use to determine the path we load the data from.
"""
pixel_scales = 0.1

for dataset_name in dataset_names:

    """
    __Paths__
    
    Set up the config and output paths.
    """
    dataset_path = path.join("dataset", "database", dataset_name)

    """
    __Dataset__
    
    Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files.
    
    This `Imaging` object will be available via the aggregator. Note also that we give the dataset a `name` via the
    command `name=dataset_name`. we'll use this name in the aggregator tutorials.
    """
    imaging = al.Imaging.from_fits(
        image_path=path.join(dataset_path, "image.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=pixel_scales,
        name=dataset_name,
    )

    """
    __Mask__
    
    The `Mask2D` we fit this data-set with, which will be available via the aggregator.

    The `SettingsImaging` (which customize the fit of the search`s fit), will also be available to the aggregator! 
    """
    mask = al.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    settings_imaging = al.SettingsImaging(grid_class=al.Grid2D, sub_size=2)

    masked_imaging = imaging.apply_mask(mask=mask, settings=settings_imaging)

    """
    __Info__

    Information about our model-fit that isn't part of the model-fit can be made accessible to the database, by 
    passing an `info` dictionary. 

    Below we load this info dictionary from an `info.json` file stored in each dataset's folder. This dictionary
    contains the (hypothetical) lens redshift, source redshift and lens velocity dispersion of every lens in our sample.
    """
    with open(path.join(dataset_path, "info.json")) as json_file:
        info = json.load(json_file)

    """
    __Pickle Files__

    We can pass strings specifying the path and filename of .pickle files stored on our hard-drive to the `search.fit()`
    method, which will make them accessible to the aggregator to aid interpretation of results. Our simulated strong
    lens datasets have a `true_tracer.pickle` file which we pass in below, which we use in the `Aggregator` tutorials 
    to check if the model-fit recovers its true input parameters.
    """
    pickle_files = [path.join(dataset_path, "true_tracer.pickle")]

    """
    Model-Fit:
    
    We perform the model-fit as per usual
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllipticalSersic),
        )
    )

    search = af.DynestyStatic(
        path_prefix=path.join("database", dataset_name),
        name="database_example",
        n_live_points=50,
    )

    analysis = al.AnalysisImaging(dataset=masked_imaging)

    search.fit(analysis=analysis, model=model, info=info, pickle_files=pickle_files)
