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
    """Generate mu and xi raster files from a polygon shapefile.

    Reads a polygon shapefile with "mu" and "xsi" attribute fields,
    rasterizes the attribute values onto a grid matching the DEM extent
    and resolution, and writes the rasters to Inputs/RASTERS/.

    Parameters
    ----------
    avaDir : pathlib.Path
        Path to avalanche directory containing Inputs/DEM and
        Inputs/POLYGONS/ with *_spatialVoellmy.shp shapefile.
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

    # Find shapefile
    shpPath, shpAvailable, _ = getAndCheckInputFiles(
        inputDir, "POLYGONS", "spatialVoellmy shapefile", fileExt="shp",
        fileSuffix="_spatialVoellmy"
    )
    if shpAvailable == "No":
        raise FileNotFoundError(
            "No *_spatialVoellmy.shp found in %s/POLYGONS/" % inputDir
        )

    # Read DEM header
    demHeader = readRasterHeader(demPath)
    demTransform = demHeader["transform"]
    demCrs = demHeader["crs"]
    demShape = (demHeader["nrows"], demHeader["ncols"])

    defaultMu = cfg["DEFAULTS"].getfloat("default_mu")
    defaultXsi = cfg["DEFAULTS"].getfloat("default_xsi")

    # Validate required fields
    with shapefile.Reader(str(shpPath)) as sf:
        fieldNames = [f[0].lower() for f in sf.fields[1:]]
    for field in ["mu", "xsi"]:
        if field not in fieldNames:
            raise KeyError(
                "Field '%s' not found in %s. Available fields: %s"
                % (field, shpPath.name, fieldNames)
            )

    # Rasterize mu and xsi from the same shapefile
    log.info("Rasterizing mu from: %s", shpPath)
    muRaster = _rasterizeShapefile(shpPath, defaultMu, "mu", demShape, demTransform)

    log.info("Rasterizing xsi from: %s", shpPath)
    xsiRaster = _rasterizeShapefile(shpPath, defaultXsi, "xsi", demShape, demTransform)

    # Determine output driver
    if demSuffix == ".asc":
        driver = "AAIGrid"
    else:
        driver = "GTiff"

    # Check if any mu or xi raster files already exist
    existing = sorted(p.name for p in outDir.glob("*_mu.*")) + sorted(p.name for p in outDir.glob("*_xi.*"))
    if existing:
        raise FileExistsError(
            "Output file(s) already exist in %s: %s" % (outDir, ", ".join(existing))
        )

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

    All cells not covered by any polygon are filled with defaultValue.

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
