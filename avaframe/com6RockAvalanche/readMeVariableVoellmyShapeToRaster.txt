The VariableVoellmyShapeToRaster.py script allows the user to define spatially different values for the voellmy Parameters mu and xsi, with the use of polygon shapefiles. For the extent of a DEM raster, all the areas that are not covered by a polygon get assigned a default mu or xsi value. The script then converts this Information into a raster mu and a raster xsi file, which can then be used in Avaframe Simulation runs, using the "spatialVoellmy" friction model. 

First, set up the Config File and provide inputs:
•Config File:
	oIn the first step, the Config File needs to be configured and all input files have to be provided
		Main Config (avaframeCfg.ini):
			•Set the path to the avalanche directory 
oInputs:
	All the Input Files are automatically fetched through the set avalanche directory. It is not necessary to provide a file path.
	dem: DEM Raster that is later on used for the avaframe simulation. This is needed, because the mu and xsi output rasters need to be the exact same size. Has to lie in avadir/Inputs. 
	mu_shapefile: Mu shapefile, that is then converted to a raster file. Be aware, that the attribute has to be named “mu” and the file name has to end with “_mu”. Has to lie in avadir/Inputs/POLYGONS.
	xsi_shapefile: Xsi shapefile, that is then converted to a raster file. Be aware, that the attribute has to be named “xsi” and the file name has to end with “_xsi”. Has to lie in avadir/Inputs/POLYGONS.
oDefaults:
	default_mu: this is the default mu value, that gets assigned to all areas in the raster, that are not covered by shapefile-polygons
	default_xsi: this is the default xsi value, that gets assigned to all areas in the raster, that are not covered by shapefile-polygons
oOutputs:
	For the variable Voellmy calculations in the com1DFA algorithm to work, it is mandatory, that the files are stored in: avaframe\data\*yourAvalancheDir*\Inputs\RASTERS\
	mu_raster: Output for the generated mu raster file stored as *_mu.asc
	xsi_raster: Output for the generated xsi raster file stored as *_xi.asc

•RunScript:
	oOnce everything is set up, run the script “runVariableVoellmyShapeToRaster.py”
	oIf libraries are missing use: pip install *name of missing library
