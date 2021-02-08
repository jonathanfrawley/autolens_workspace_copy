import autolens as al

"""
__Settings__

 - That a `Grid2DInterpolate` is used to fit create the model-image when fitting the data 
      (see `autolens_workspace/examples/grids.py` for a description of grids).
 - The pixel-scale of this interpolation grid.

The deflection angle calculation of the `EllipticalSersic` `MassProfile`.requires numerical integration and is
computationally more expensive than most mass profiles. For this reason, we use a `Grid2DInterpolate` grid instead of
the `Grid2D` we use in most other examples, which limits the deflection angle calculation to a grid of reduced resolution
and interpolates the results to the native-resolution grid. 

A description of the *GridIterpolate* object can be found in the script `autolens_workspace/examples/grids.py`.

Different `SettingsPhase` are used in different example model scripts and a full description of all `SettingsPhase` 
can be found in the example script `autolens/workspace/notebooks/imaging/modeling/customize/settings.py` and the following 
chain -> <chain>
"""
settings_masked_imaging = al.SettingsMaskedImaging(
    grid_class=al.Grid2DInterpolate, pixel_scales_interp=0.1
)

"""
Finish.
"""
