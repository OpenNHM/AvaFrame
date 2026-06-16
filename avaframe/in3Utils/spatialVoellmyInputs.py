"""
Functions for generating spatial Voellmy friction raster inputs.
"""

import pathlib
import logging
import numpy as np
import shapefile
from rasterio.features import rasterize
from shapely.geometry import shape, mapping

from avaframe.in1Data.getInput import getAndCheckInputFiles, getDEMPath
from avaframe.in2Trans.rasterUtils import readRasterHeader, writeResultToRaster

log = logging.getLogger(__name__)


def generateMuXsiRasters(avaDir, cfg):
    """Generate mu and xi raster files from polygon shapefiles.

    Reads polygon shapefiles with "mu" and "xsi" attribute fields,
    rasterizes the attribute values onto a grid matching the DEM extent
    and resolution, and writes the rasters to Inputs/RASTERS/.

    Parameters
    ----------
    avaDir : pathlib.Path
        Path to avalanche directory containing Inputs/DEM and
        Inputs/POLYGONS/ with *_mu.shp and *_xsi.shp shapefiles.
    cfg : configparser.ConfigParser
        Configuration with [DEFAULTS] section containing
        default_mu and default_xsi values for uncovered areas.
    """
    avaDir = pathlib.Path(avaDir)
    inputDir = avaDir / "Inputs"
    outDir = inputDir / "RASTERS"
    outDir.mkdir(parents=True, exist_ok=True)

    # Find DEM
    demPath = getDEMPath(avaDir)
    demSuffix = demPath.suffix

    # Find shapefiles
    muShpPath, muAvailable, _ = getAndCheckInputFiles(
        inputDir, "POLYGONS", "mu shapefile", fileExt="shp", fileSuffix="_mu"
    )
    if muAvailable == "No":
        raise FileNotFoundError("No *_mu.shp found in %s/POLYGONS/" % inputDir)
    xsiShpPath, xsiAvailable, _ = getAndCheckInputFiles(
        inputDir, "POLYGONS", "xsi shapefile", fileExt="shp", fileSuffix="_xsi"
    )
    if xsiAvailable == "No":
        raise FileNotFoundError("No *_xsi.shp found in %s/POLYGONS/" % inputDir)

    # Read DEM header
    demHeader = readRasterHeader(demPath)
    demTransform = demHeader["transform"]
    demCrs = demHeader["crs"]
    demShape = (demHeader["nrows"], demHeader["ncols"])

    defaultMu = float(cfg["DEFAULTS"]["default_mu"])
    defaultXsi = float(cfg["DEFAULTS"]["default_xsi"])

    # Rasterize mu
    log.info("Rasterizing mu shapefile: %s", muShpPath)
    muRaster = _rasterizeShapefile(muShpPath, defaultMu, "mu", demShape, demTransform)

    # Rasterize xsi
    log.info("Rasterizing xsi shapefile: %s", xsiShpPath)
    xsiRaster = _rasterizeShapefile(xsiShpPath, defaultXsi, "xsi", demShape, demTransform)

    # Determine output driver
    if demSuffix == ".asc":
        driver = "AAIGrid"
    else:
        driver = "GTiff"

    # Write output
    outHeader = {
        "driver": driver,
        "crs": demCrs,
        "transform": demTransform,
        "nodata_value": None,
    }
    log.info("Writing mu raster")
    writeResultToRaster(outHeader, muRaster, outDir / "raster_mu")
    log.info("Writing xsi raster")
    writeResultToRaster(outHeader, xsiRaster, outDir / "raster_xi")
    log.info("Raster generation completed.")


def _rasterizeShapefile(shpPath, defaultValue, fieldName, demShape, demTransform):
    """Rasterize a polygon shapefile attribute field onto a DEM-matching grid.

    Parameters
    ----------
    shpPath : pathlib.Path
        Path to shapefile.
    defaultValue : float
        Fill value for cells not covered by any polygon.
    fieldName : str
        Attribute field name to extract from each feature.
    demShape : tuple
        (height, width) of the output raster.
    demTransform : affine.Affine
        Geotransform of the DEM.

    Returns
    -------
    raster : numpy.ndarray
        Rasterized array with shape demShape.
    """
    with shapefile.Reader(str(shpPath)) as sf:
        fieldNames = [f[0].lower() for f in sf.fields[1:]]
        if fieldName not in fieldNames:
            raise KeyError(
                "Field '%s' not found in %s. Available fields: %s" % (fieldName, shpPath.name, fieldNames)
            )
        fieldIdx = fieldNames.index(fieldName)

        shapes = []
        for rec in sf.shapeRecords():
            geom = rec.shape.__geo_interface__
            poly = shape(geom)
            value = rec.record[fieldIdx]
            shapes.append((mapping(poly), value))

    raster = rasterize(
        shapes,
        out_shape=demShape,
        transform=demTransform,
        fill=defaultValue,
        all_touched=True,
        dtype=np.float32,
    )
    return raster
