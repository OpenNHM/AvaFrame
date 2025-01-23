 
import rasterio
import numpy as np
import pathlib
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from in2Trans.shpConversion import SHP2Array
from in1Data.getInput import getAndCheckInputFiles
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def generateMuXsiRasters(avadir, variableVoellmyCfg):
    """
    Generate raster files for \u03bc and \u03be based on input DEM and shapefiles.

    Parameters
    ----------
    avadir : str
        Path to the avalanche directory.
    variableVoellmyCfg : Config Parser Object
        variableVoellmyCfg Configuration File

    Returns
    -------
    None
    """
    avadir = pathlib.Path(avadir)
    
    config = variableVoellmyCfg  # Directly use the ConfigParser object
    
    inputDir = avadir / "Inputs"
    outputDir = avadir / "Inputs" # Output directory is Inputs, because Outputs of this Script will be used as Inputs for AvaFrame

    demPath, _ = getAndCheckInputFiles(inputDir, '', 'DEM', fileExt='asc')
    muShapefile, _ = getAndCheckInputFiles(inputDir, 'POLYGONS', '\u03bc Shapefile', fileExt='shp', fileSuffix='_mu')
    xsiShapefile, _ = getAndCheckInputFiles(inputDir, 'POLYGONS', '\u03be Shapefile', fileExt='shp', fileSuffix='_xsi')

    muOutputPath = outputDir / "RASTERS" / "raster_mu.asc"
    xsiOutputPath = outputDir / "RASTERS" /"raster_xi.asc"

    defaultMu = float(config['DEFAULTS']['default_mu'])
    defaultXsi = float(config['DEFAULTS']['default_xsi'])

    # Read DEM
    with rasterio.open(demPath) as demSrc:
        demData = demSrc.read(1)
        demTransform = demSrc.transform
        demCrs = demSrc.crs
        demShape = demData.shape

    def rasterizeShapefile(shapefilePath, defaultValue, attributeName):
        if not shapefilePath:
            return np.full(demShape, defaultValue, dtype=np.float32)

        shpData = SHP2Array(shapefilePath)
        shapes = []
        for i in range(shpData['nFeatures']):
            start = int(shpData['Start'][i])
            length = int(shpData['Length'][i])
            coords = [(shpData['x'][j], shpData['y'][j]) for j in range(start, start + length)]
            poly = shape({'type': 'Polygon', 'coordinates': [coords]})
            value = shpData['attributes'][i][attributeName]
            shapes.append((mapping(poly), value))

        return rasterize(shapes, out_shape=demShape, transform=demTransform, fill=defaultValue, all_touched=True, dtype=np.float32)

    log.info("Rasterizing \u03bc shapefile.")
    muRaster = rasterizeShapefile(muShapefile, defaultMu, "mu")

    log.info("Rasterizing \u03be shapefile.")
    xsiRaster = rasterizeShapefile(xsiShapefile, defaultXsi, "xsi")

    def saveRaster(outputPath, data):
        with rasterio.open(outputPath, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, crs=demCrs, transform=demTransform) as dst:
            dst.write(data, 1)

    log.info("Saving \u03bc raster to %s", muOutputPath)
    saveRaster(muOutputPath, muRaster)

    log.info("Saving \u03be raster to %s", xsiOutputPath)
    saveRaster(xsiOutputPath, xsiRaster)

    log.info("Raster generation completed.")
