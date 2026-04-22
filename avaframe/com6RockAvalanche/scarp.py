import rasterio
import numpy as np
import math
import logging
import pathlib
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from avaframe.in2Trans.shpConversion import SHP2Array
from avaframe.in1Data import getInput as gI
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.in2Trans.rasterUtils as IOf

# Configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def scarpAnalysisMain(cfg, baseDir):
    """Run the scarp analysis using parameters from an .ini cfguration file and input from a directory

    Parameters
    ----------
    cfg : cfgparser
        with all required fields in scarpCfg.ini
    baseDir: str
        path to directory of data to analyze

    Returns
    -------
    None
    """

    # Get input from baseDir and check if all files exist
    if cfg["INPUT"].getboolean("useShapefiles"):

        # Set directories for inputs, outputs and current work
        inputDir = pathlib.Path(baseDir, "Inputs")

        # get the path to the perimeter shapefile
        perimeterShapefilePath, periFile, _ = gI.getAndCheckInputFiles(
            inputDir, "POLYGONS", "scarp perimeter", fileExt="shp", fileSuffix="_perimeter"
        )
        if periFile:
            log.info("Perimeterfile is: %s" % perimeterShapefilePath)
        else:
            log.error("Perimeter shapefile not found in %s", inputDir)

        # get the path to the coordinates shapefile
        coordinatesShapefilePath, coordFile, _ = gI.getAndCheckInputFiles(
            inputDir, "POINTS", "scarp coordinates", fileExt="shp", fileSuffix="_coordinates"
        )
        if coordFile:
            log.info("Coordinate shapefile is: %s" % coordinatesShapefilePath)
        else:
            log.error("Coordinate shapefile not found in %s", inputDir)

    else:
        log.error(
            "Shapefile option not selected. Please set 'useShapefiles' to True in the configuration file. A "
            "raster version is currently not implemented."
        )

    # Initialize feature parameters
    planeFeatures = []
    ellipsoidFeatures = []

    # Read the DEM
    dem = gI.readDEM(baseDir)
    dem["rasterData"] = np.flipud(dem["rasterData"])

    n = dem["header"]["nrows"]
    m = dem["header"]["ncols"]

    periData = readPerimeterSHP(perimeterShapefilePath, dem["header"]["transform"], (n, m))

    SHPdata = SHP2Array(coordinatesShapefilePath)

    log.debug("Feature shapefile data loaded: %s", SHPdata)

    method = cfg["SETTINGS"]["method"].lower()

    if method == "plane":
        # Read required attributes directly from the shapefile's attribute table
        try:
            planesZseed = list(map(float, SHPdata['zseed']))
            planesDipDir = list(map(float, SHPdata['dipdir']))
            planesDipAngle = list(map(float, SHPdata['dipAngle']))
    
        except KeyError as e:
            raise ValueError(f"Required attribute '{e.args[0]}' not found in shapefile. Make sure 'zseed', 'dipdir', and 'dipangle' fields exist.")

        if not (len(planesZseed) == len(planesDipDir) == len(planesDipAngle) == SHPdata["nFeatures"]):
            raise ValueError("Mismatch between number of features and extracted plane attributes in the shapefile.")

        if not (len(planesZseed) == len(planesDipDir) == len(planesDipAngle) == SHPdata["nFeatures"]):
            raise ValueError(
                "Mismatch between number of shapefile features and plane parameters in the .ini file."
            )

        for i in range(SHPdata["nFeatures"]):
            xSeed = SHPdata["x"][int(SHPdata["Start"][i])]
            ySeed = SHPdata["y"][int(SHPdata["Start"][i])]
            zSeed = planesZseed[i]
            dip = planesDipDir[i]
            slopeAngle = planesDipAngle[i]
            planeFeatures.extend([xSeed, ySeed, zSeed, dip, slopeAngle])

        features = ",".join(map(str, planeFeatures))
        log.debug("Plane features extracted and combined: %s", features)

    elif method == "ellipsoid":
        try:
            ellipsoidsMaxDepth = list(map(float, SHPdata['maxdepth']))
            ellipsoidsSemiMajor = list(map(float, SHPdata['semimajor']))
            ellipsoidsSemiMinor = list(map(float, SHPdata['semiminor']))
            ellipsoidsDipAngle = list(map(float, SHPdata['dipAngle']))
            ellipsoidsDipDir = list(map(float, SHPdata['dipdir']))
            ellipsoidsOffset = list(map(float, SHPdata['offset']))
            ellipsoidsRotAngle = list(map(float, SHPdata['rotAngle']))
        except KeyError as e:
            raise ValueError(f"Required attribute '{e.args[0]}' not found in shapefile. Ensure the fields 'maxdepth', 'semimajor', 'semiminor', 'rotangle', 'dipdir', 'dipangle', and 'offset' exist.")

        if not all(len(lst) == SHPdata["nFeatures"] for lst in [ellipsoidsMaxDepth, ellipsoidsSemiMajor, ellipsoidsSemiMinor, ellipsoidsDipAngle, ellipsoidsDipDir, ellipsoidsOffset, ellipsoidsRotAngle]):
            raise ValueError("Mismatch between number of shapefile features and ellipsoid parameters.")

        for i in range(SHPdata["nFeatures"]):
            xCenter = SHPdata["x"][int(SHPdata["Start"][i])]
            yCenter = SHPdata["y"][int(SHPdata["Start"][i])]
            maxDepth = ellipsoidsMaxDepth[i]
            semiMajor = ellipsoidsSemiMajor[i]
            semiMinor = ellipsoidsSemiMinor[i]
            tilt = ellipsoidsDipAngle[i]
            direction = ellipsoidsDipDir[i]
            offset = ellipsoidsOffset[i]
            dip = ellipsoidsRotAngle[i]
            ellipsoidFeatures.extend([xCenter, yCenter, maxDepth, semiMajor, semiMinor, tilt, direction, offset, dip])

        features = ",".join(map(str, ellipsoidFeatures))

        log.debug("Ellipsoid features extracted and combined: %s", features)

    if method == 'plane':
        scarpData = calculateScarpWithPlanes(
            dem["rasterData"], periData, dem["header"]["transform"], features
        )
    elif method == 'ellipsoid':
        scarpData = calculateScarpWithEllipsoids(
            dem["rasterData"], periData, dem["header"]["transform"], features
        )
    else:
        raise ValueError("Unsupported method. Choose 'plane' or 'ellipsoid'.")
        

    hRelData = dem["rasterData"] - scarpData
    min_hrel = hRelData.min()
    if min_hrel < 0:
        log.warning("hRel: most negative clipped value = %.2f m", min_hrel)
    hRelData = np.maximum(0, hRelData)   #enforce non-negative release

    
    #Compute and log excavated volume
    cellArea = abs(dem["header"]["cellsize"] ** 2)
    volume = np.sum(hRelData[periData > 0]) * cellArea
    log.info(f"Excavated volume (within perimeter): {volume:.2f} m³")

    # create output directory and files
    outDir = pathlib.Path(baseDir)
    outDir = outDir / "Outputs" / "com6RockAvalanche" / "scarp"
    fU.makeADir(outDir)

    # Set file names and write
    elevationOut = outDir / "scarpElevation"
    hRelOut = outDir / "scarpHRel"
    IOf.writeResultToRaster(dem["header"], scarpData, elevationOut)
    IOf.writeResultToRaster(dem["header"], hRelData, hRelOut)

def readPerimeterSHP(perimeterShapefilePath, elevTransform, elevShape):
    """Read perimeter from shapefile and return as numpy array.

    Parameters
    ----------
    perimeterShapefilePath : str
        Path to the perimeter shapefile.
    elevTransform : Affine
        transformation of the elevation raster.
    elevShape : tuple
        Shape (height, width) of the elevation raster.

    Returns
    -------
    periData : np.ndarray
        2D array representing the perimeter mask.
    """

    log.debug("Processing perimeter shapefile: %s", perimeterShapefilePath)
    SHPdata = SHP2Array(perimeterShapefilePath)
    log.debug("Perimeter shapefile data loaded: %s", SHPdata)

    shapes = []
    for i in range(SHPdata["nFeatures"]):
        start = int(SHPdata["Start"][i])
        length = int(SHPdata["Length"][i])
        coords = [(SHPdata["x"][j], SHPdata["y"][j]) for j in range(start, start + length)]
        poly = shape({"type": "Polygon", "coordinates": [coords]})
        shapes.append(poly)

    periData = rasterize(
        [(mapping(poly), 1) for poly in shapes],
        out_shape=elevShape,
        transform=elevTransform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    log.debug("Perimeter shapefile rasterized.")

    return periData


def calculateScarpWithPlanes(elevData, periData, elevTransform, planes):
    """Calculate the scarp using sliding planes.

    Parameters
    ----------
    elevData : np.ndarray
        The elevation data as a 2D array.
    periData : np.ndarray
        The perimeter data as a 2D array, which defines the extent of the scarp.
    elevTransform : Affine
        The affine transformation matrix of the raster (used to convert pixel to geographic coordinates).
    planes : str
        Comma-separated string defining sliding planes (xseed, yseed, zseed, dip, slope).

    Returns
    -------
    np.ndarray
        A 2D array with the calculated scarp values.
    """
    n, m = elevData.shape
    scarpData = np.zeros_like(elevData, dtype=np.float32)

    planes = list(map(float, planes.split(',')))
    nPlanes = int(len(planes) / 5)

    xSeed = [planes[0]]
    ySeed = [planes[1]]
    zSeed = [planes[2]]
    dip = [planes[3]]
    slope = [planes[4]]

    slopeRad = math.radians(slope[0])
    dipRad   = math.radians(dip[0])
    betaX = [ math.tan(slopeRad) * math.sin(dipRad) ]
    betaY = [ math.tan(slopeRad) * math.cos(dipRad) ]

    min_clipped = 0.0

    for i in range(1, nPlanes):
        xSeed.append(planes[5 * i])
        ySeed.append(planes[5 * i + 1])
        zSeed.append(planes[5 * i + 2])
        dip.append(planes[5 * i + 3])
        slope.append(planes[5 * i + 4])
        
        slopeRad = math.radians(slope[i])
        dipRad   = math.radians(dip[i])
        betaX.append( math.tan(slopeRad) * math.sin(dipRad) )
        betaY.append( math.tan(slopeRad) * math.cos(dipRad) )

    for row in range(n):
        for col in range(m):
            west, north = rasterio.transform.xy(elevTransform, row, col, offset='center')

            scarpVal = zSeed[0] + (north - ySeed[0]) * betaY[0] - (west - xSeed[0]) * betaX[0]
            for k in range(1, nPlanes):
                scarpVal = max(scarpVal, zSeed[k] + (north - ySeed[k]) * betaY[k] - (west - xSeed[k]) * betaX[k])
                
            if periData[row, col] > 0:
                val = min(elevData[row, col], scarpVal)
                if val < 0:
                    min_clipped = min(min_clipped, val)
                scarpData[row, col] = max(0, val)
            else:
                scarpData[row, col] = elevData[row, col]
                
    if min_clipped < 0:
        log.warning("Plane scarp: most negative clipped value = %.2f m", min_clipped)

    return scarpData

def calculateScarpWithEllipsoids(elevData, periData, elevTransform, ellipsoids):
    """Calculate the scarp using tilted and offset ellipsoids.

    Parameters
    ----------
    elevData : np.ndarray
        The elevation data as a 2D array.
    periData : np.ndarray
        The perimeter data as a 2D array.
    elevTransform : Affine
        The affine transformation matrix of the raster.
    ellipsoids : str
        Comma-separated string defining ellipsoids with parameters:
        (x_center, y_center, max_depth, semi_major, semi_minor, tilt, dir, offset)

    Returns
    -------
    np.ndarray
        A 2D array with the calculated scarp values.
    """

    n, m = elevData.shape
    scarpData = np.zeros_like(elevData, dtype=np.float32)

    # Parse ellipsoid definitions
    ellipsoids = list(map(float, ellipsoids.split(',')))
    nEllipsoids = int(len(ellipsoids) / 9)

    xCenter, yCenter, maxDepth = [], [], []
    semiMajor, semiMinor = [], []
    tilt, tiltDir, offset, dip = [], [], [], []
    min_clipped = 0.0

    for i in range(nEllipsoids):
        xCenter.append(ellipsoids[9 * i])
        yCenter.append(ellipsoids[9 * i + 1])
        maxDepth.append(ellipsoids[9 * i + 2])
        semiMajor.append(ellipsoids[9 * i + 3])
        semiMinor.append(ellipsoids[9 * i + 4])
        tilt.append(ellipsoids[9 * i + 5])
        tiltDir.append(np.radians(ellipsoids[9 * i + 6]))  # tilt direction in radians
        offset.append(ellipsoids[9 * i + 7])
        dip.append(np.radians(ellipsoids[9 * i + 8]))      # rotation of base ellipse in radians

    # Compute slope direction and magnitude from DEM gradients
    grad_y, grad_x = np.gradient(elevData, abs(elevTransform[4]), elevTransform[0])
    slopeDir = np.arctan2(-grad_y, grad_x)  # Azimuth of steepest descent
    slopeMagnitude = np.sqrt(grad_x**2 + grad_y**2)

    for row in range(n):
        for col in range(m):
            west, north = rasterio.transform.xy(elevTransform, row, col, offset='center')
            scarpVal = elevData[row, col]

            for k in range(nEllipsoids):
                center_col, center_row = ~elevTransform * (xCenter[k], yCenter[k])
                center_row = int(round(center_row))
                center_col = int(round(center_col))

                if 0 <= center_row < n and 0 <= center_col < m:
                    slopeAzimuth = slopeDir[center_row, center_col]
                    slopeAzimuth = (slopeAzimuth + 2 * np.pi) % (2 * np.pi)
                    slopeMag = slopeMagnitude[center_row, center_col]

                    if not np.isnan(slopeAzimuth) and slopeMag > 0:
                        normal_dx = np.cos(slopeAzimuth)
                        normal_dy = np.sin(slopeAzimuth)
                        slopeAngle = math.atan(slopeMag)
                        dxOffset = -offset[k] * normal_dx * math.sin(slopeAngle)
                        dyOffset = -offset[k] * normal_dy * math.sin(slopeAngle)
                        dzOffset = -offset[k] * math.cos(slopeAngle)
                        #clamp z-offset
                        dzOffset = max(-maxDepth[k], min(dzOffset, maxDepth[k]))
                    else:
                        dxOffset = dyOffset = dzOffset = 0
                else:
                    dxOffset = dyOffset = dzOffset = 0

                x0 = xCenter[k] + dxOffset
                y0 = yCenter[k] + dyOffset
                z0 = dzOffset

                dxPos = west - x0
                dyPos = north - y0
                
                dxRot =  dxPos * np.sin(dip[k]) - dyPos * np.cos(dip[k])
                dyRot =  dxPos * np.cos(dip[k]) + dyPos * np.sin(dip[k])
                # Normalize to ellipsoid axes
                xNorm = dxRot / semiMajor[k]
                yNorm = dyRot / semiMinor[k]
                distance = xNorm**2 + yNorm**2

                if distance <= 1:
                    baseDepth = maxDepth[k] * (1 - distance)
                    tiltEffect = math.tan(math.radians(tilt[k])) * (
                        dxRot * np.cos(tiltDir[k]) + dyRot * np.sin(tiltDir[k])
                    )
                    totalDepth = baseDepth + tiltEffect + z0
                    scarpVal = min(scarpVal, elevData[row, col] - totalDepth)

            if periData[row, col] > 0:
                val = min(elevData[row, col], scarpVal)
                if val < 0:
                    min_clipped = min(min_clipped, val)
                scarpData[row, col] = max(0, val)
            else:
                scarpData[row, col] = elevData[row, col]

    if min_clipped < 0:
        log.warning("Ellipsoid scarp: most negative clipped value = %.2f m", min_clipped)            

    return scarpData
