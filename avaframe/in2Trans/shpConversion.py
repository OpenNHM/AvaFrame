"""
Conversion functions to read/ write Shape files
"""

import pathlib
import shapefile
import copy
import numpy as np
import logging

from shapely.geometry import shape

# create local logger
log = logging.getLogger(__name__)


def SHP2Array(infile, defname=None):
    """Read shapefile and convert it to a python dictionary

    The dictionary contains the name of the paths in the shape file, the np array with
    the coordinates of the feature points (all stacked in the same array)
    and information about the starting index and length of each feature

    Parameters
    ----------
    infile: str
        path to shape file
    defname: str
        name to give to the feature in the shape file

    Returns
    -------
    SHPdata : dict
        sks :
            projection information
        Name : str
            list of feature names
        x : 1D numpy array
            np array of the x coord of the points in the features
        y : 1D numpy array
            np array of the y coord of the points in the features
        z : 1D numpy array
            np array of zeros and same size as the x an y coordinates array
        Start : 1D numpy array
            np array with the starting index of each feature in the coordinates
            arrays (as many indexes as features)
        Length : 1D numpy array
            np array with the length of each feature in the coordinates
            arrays (as many indexes as features)
        thickness (optional) : 1D numpy array
            np array with the (release or entrainment) thickness of each feature (as many values as features)
        id : list
            list of oid as string for each feature
        ci95: list
            list of 95% confidence interval of thickness value
        nParts: list
            list of parts of polygon (added the total number of points as list item, so if multiple parts len>2)
        nFeatures: int
            number of features per line (parts)
        zseed
            np array with the height of each scarp plane-feature (as many values as features)
        slopeAngle
            np array with the slope angle of each scarp plane-feature (as many values as features)
        dip
            np array with the dip angle of each scarp plane-feature (as many values as features)
        semiminor
            np array with the semi-minor axis of each scarp ellipsoid-feature (as many values as features)
        maxdepth
            np array with the masimum depth of each scarp ellipsoid-feature (as many values as features)
        semimajor
            np array with the semi-major axis of each scarp ellipsoid-feature (as many values as features)
    """
    #  Input shapefile
    sf = shapefile.Reader(str(infile))
    infile = pathlib.Path(infile)
    # set defaults for variables
    layername = None
    thickness = None
    slope = None
    rho = None
    sks = None
    iso = None
    id = None
    ci95 = None
    layerN = None
    zseed_value = None
    dip_value = None
    slopeAngle_value = None
    semiminor_value = None
    maxdepth_value = None
    semimajor_value = None
    tilt_value = None
    direc_value = None
    offset_value = None

    # get coordinate system
    sks = getSHPProjection(infile)

    # Start reading the shapefile
    records = sf.shapeRecords()
    shps = sf.shapes()

    SHPdata = {}
    SHPdata["sks"] = sks
    Name = []
    thicknessList = []
    idList = []
    ci95List = []
    layerNameList = []
    zseedList = []
    dipList = []
    slopeList = []
    slopeAngleList = []
    semiminorList = []
    semimajorList = []
    maxdepthList = []
    tiltList = []
    direcList = []
    offsetList = []

    Length = np.empty(0)
    Start = np.empty(0)
    Coordx = np.empty(0)
    Coordy = np.empty(0)
    Coordz = np.empty(0)
    start = 0
    nParts = []

    # New: Create an empty list to store attributes
    attributes = []

    for n, (item, rec) in enumerate(zip(shps, sf.records())):
        pts = item.points
        # if feature has no points - ignore
        if len(pts) < 1:
            log.warning("Feature %d in %s no points found - ignored" % (n, infile.name))
            continue

        # is there a z coordinate?
        try:
            # for dams
            zs = item.z
        except AttributeError:
            zs = [0.0] * len(pts)
        # add info on number of parts
        nParts.append(list(item.parts) + [len(item.points)])

        # check if records are available and extract
        if records:
            # loop through fields
            # Extract attributes for the feature
            attr_dict = {}
            for (name, typ, size, deci), value in zip(sf.fields[1:], records[n].record):
                # get entity name
                name = name.lower()
                attr_dict[name] = value  # Store attributes in dictionary

                # Specific field handling (existing code)
                if name == "name":
                    layername = str(value)
                if (name == "thickness") or (name == "d0"):
                    thickness = value
                if name == "ci95":
                    ci95 = value
                if name == "slope":
                    # for dams
                    slope = value
                if name == "slopeangle":
                    # for dams
                    slopeAngle_value = value
                if name == "zseed":
                    zseed_value = value
                if name == "dip":
                    dip_value = value
                if name == "semiminor":
                    semiminor_value = value
                if name == "maxdepth":
                    maxdepth_value = value
                if name == "semimajor":
                    semimajor_value = value
                if name == "tilt":
                    tilt_value = value
                if name == "direc":
                    direc_value = value
                if name == "offset":
                    offset_value = value
                if name == "rho":
                    rho = value
                if name == "sks":
                    sks = value
                if name == "iso":
                    iso = value
                if name == "layer":
                    layerN = value               
            # if name is still empty go through file again and take Layer instead
            if (type(layername) is bytes) or (layername is None):
                for (name, typ, size, deci), value in zip(sf.fields[1:], records[n].record):
                    if name == "Layer":
                        layername = value

            attributes.append(attr_dict)  # Add the attribute dictionary to the list

        # if layer still not defined, use generic
        if layername is None:
            layername = defname

        Name.append(layername)
        log.debug("SHPConv: Found layer %s", layername)
        thicknessList.append(str(thickness))
        ci95List.append(str(ci95))
        layerNameList.append(layerN)
        idList.append(str(rec.oid))
        zseedList.append(zseed_value)
        dipList.append(dip_value)
        slopeList.append(slope)
        slopeAngleList.append(slopeAngle_value)
        semiminorList.append(semiminor_value)
        semimajorList.append(semimajor_value)
        maxdepthList.append(maxdepth_value)
        tiltList.append(tilt_value)
        direcList.append(direc_value)
        offsetList.append(offset_value)

        Start = np.append(Start, start)
        length = len(pts)
        Length = np.append(Length, length)
        start += length

        for pt, z in zip(pts, zs):
            Coordx = np.append(Coordx, pt[0])
            Coordy = np.append(Coordy, pt[1])
            Coordz = np.append(Coordz, z)

    SHPdata["Name"] = Name
    SHPdata["thickness"] = thicknessList
    SHPdata["slope"] = slope
    SHPdata["Start"] = Start
    SHPdata["Length"] = Length
    SHPdata["x"] = Coordx
    SHPdata["y"] = Coordy
    SHPdata["z"] = Coordz
    SHPdata["id"] = idList
    SHPdata["ci95"] = ci95List
    SHPdata["layerName"] = layerNameList
    SHPdata["nParts"] = nParts
    SHPdata["nFeatures"] = len(Start)
    SHPdata["slopeangle"] = slopeAngleList
    SHPdata["zseed"] = zseedList
    SHPdata["dip"] = dipList
    SHPdata["maxdepth"] = maxdepthList
    SHPdata["semimajor"] = semimajorList
    SHPdata["semiminor"] = semiminorList
    SHPdata["tilt"] = tiltList
    SHPdata["direc"] = direcList
    SHPdata["offset"] = offsetList
    SHPdata["attributes"] = attributes  # Add attributes to SHPdata

    sf.close()

    return SHPdata


def getSHPProjection(infile):
    """Fetch projection from shp file

    Parameters
    ----------
    infile: str
        path to shape file

    Returns
    -------
    sks: str
        projection string (if available, None if not)
    """
    # get coordinate system
    prjfile = infile.with_suffix(".prj")
    if prjfile.is_file():
        prjf = open(prjfile, "r")
        sks = prjf.readline()
        prjf.close()
        return sks
    else:
        message = "No projection layer for shp file %s" % infile
        log.warning(message)
        return None


def readThickness(infile, defname=None):
    """Read shapefile and fetch info on features' ids and thickness values

    Parameters
    ----------
    infile: str
        path to shape file
    defname: str
        name to give to the feature in the shape file

    Returns
    -------
    thickness: list
        list of strings with the (release or entrainment) thickness of each feature (as many values as features)
    id : list
        list of strings for oid of each feature in shp file
    ci95: list
        list of all ci95 values if provided

    """
    #  Input shapefile
    sf = shapefile.Reader(str(infile))

    thickness = None
    id = None
    ci95 = None

    # Start reading the shapefile
    records = sf.shapeRecords()
    shps = sf.shapes()

    thicknessList = []
    idList = []
    ci95List = []

    for n, item in enumerate(shps):
        pts = item.points
        zs = [0.0] * len(pts)

        # check if records are available and extract
        if records:
            # loop through fields
            for (name, typ, size, deci), value in zip(sf.fields[1:], records[n].record):
                # get entity name
                name = name.lower()
                if (name == "thickness") or (name == "d0"):
                    thickness = value
                if name == "ci95":
                    ci95 = value

        thicknessList.append(str(thickness))
        ci95List.append(str(ci95))

    # get unique ID of features in shapefile
    for rec in sf.records():
        id = rec.oid
        idList.append(str(id))

    sf.close()

    return thicknessList, idList, ci95List


def readLine(fname, defname, dem):
    """Read line from  .shp
    Use SHP2Array to read the shape file.
    Check if the lines are laying inside the dem extend

    Parameters
    ----------
    fname: str
        path to shape file
    defname: str
        name to give to the line in the shape file
    dem: dict
        dem dictionary

    Returns
    -------
    Line : dict
        Line['Name'] : list of lines names
        Line['Coord'] : np array of the coords of points in lines
        Line['Start'] : list of starting index of each line in Coord
        Line['Length'] : list of length of each line in Coord
    """

    log.debug("Reading avalanche path : %s ", str(fname))
    header = dem["header"]
    rasterDEM = dem["rasterData"]
    Line = SHP2Array(fname, defname)
    coordx = Line["x"]
    coordy = Line["y"]
    for i in range(len(coordx)):
        Lx = (coordx[i] - header["xllcenter"]) / header["cellsize"]
        Ly = (coordy[i] - header["yllcenter"]) / header["cellsize"]
        if (Ly < 0) or (Ly > header["nrows"] - 1) or (Lx < 0) or (Lx > header["ncols"] - 1):
            message = "This shape file exceeds dem extent: %s " % fname
            log.error(message)
            raise ValueError(message)
        elif np.isnan(rasterDEM[int(np.floor(Ly)), int(np.floor(Lx))]):
            message = "This shape file is at least partially outside of dem extent: %s " % fname
            log.error(message)
            raise ValueError(message)
    return Line


def readPoints(fname, dem):
    """Read points from  .shp
    Use SHP2Array to read the shape file.
    Check if the points are laying inside the dem extend

    Parameters
    ----------
    fname: str
        path to shape file
    defname: str
        name to give to the points in the shape file
    dem: dict
        dem dictionary

    Returns
    -------
    Line : dict
        Line['Name'] : list of points names
        Line['Coord'] : np array of the coords of points in points
        Line['Start'] : list of starting index of each point in Coord
        Line['Length'] : list of length of each point in Coord
    """

    log.debug("Reading split point : %s ", str(fname))
    header = dem["header"]
    rasterDEM = dem["rasterData"]
    defname = "SHP"
    Points = SHP2Array(fname, defname)
    Pointx = Points["x"]
    Pointy = Points["y"]
    for i in range(len(Pointx)):
        Lx = (Pointx[i] - header["xllcenter"]) / header["cellsize"]
        Ly = (Pointy[i] - header["yllcenter"]) / header["cellsize"]
        if Ly < 0 or Ly > header["nrows"] - 1 or Lx < 0 or Lx > header["ncols"] - 1:
            raise ValueError("The split point is not on the dem. Try with another split point")
        elif np.isnan(rasterDEM[int(np.floor(Ly)), int(np.floor(Lx))]):
            raise ValueError("Nan Value encountered. Try with another split point")
    return Points


def removeFeature(featureIn, nFeature2Remove):
    """Remove feature number nFeature2Remove from featureIn

    Parameters
    ----------
    featureIn: dict
        shape file dicionary (structure produced by SHP2Array, readLine or readPoint)
    nFeature2Remove: int
        index of feature to remove from featureIn

    Returns
    -------
    featureOut : dict
        shape file dicionary without feature nFeature2Remove
    """
    StartRel = featureIn["Start"]
    LengthRel = featureIn["Length"]
    thickness = featureIn["thickness"]
    featureOut = copy.deepcopy(featureIn)
    start = StartRel[nFeature2Remove]
    end = start + LengthRel[nFeature2Remove]
    # remove feature
    featureOut["x"] = np.delete(featureIn["x"], np.arange(int(start), int(end)))
    featureOut["y"] = np.delete(featureIn["y"], np.arange(int(start), int(end)))
    if "z" in featureIn.keys():
        featureOut["z"] = np.delete(featureIn["z"], np.arange(int(start), int(end)))

    del featureOut["Name"][nFeature2Remove]
    StartRel = featureOut["Start"]
    StartRel[nFeature2Remove:] = StartRel[nFeature2Remove:] - LengthRel[nFeature2Remove]
    featureOut["Start"] = np.delete(StartRel, nFeature2Remove)
    featureOut["Length"] = np.delete(LengthRel, nFeature2Remove)
    featureOut["thickness"] = np.delete(thickness, nFeature2Remove)

    return featureOut


def extractFeature(featureIn, nFeature2Extract):
    """Extract feature nFeature2Extract from featureIn

    Parameters
    ----------
    featureIn: dict
        shape file dicionary (structure produced by SHP2Array, readLine or readPoint)
    nFeature2Extract: int
        index of feature to extract from featureIn

    Returns
    -------
    featureOut : dict
        shape file dicionary with feature nFeature2Extract
    """
    NameRel = featureIn["Name"]
    StartRel = featureIn["Start"]
    LengthRel = featureIn["Length"]
    thickness = featureIn["thickness"]
    featureOut = copy.deepcopy(featureIn)
    # extract feature
    featureOut["Name"] = [NameRel[nFeature2Extract]]
    featureOut["Start"] = np.array([0])
    featureOut["Length"] = np.array([LengthRel[nFeature2Extract]])
    featureOut["thickness"] = np.array([thickness[nFeature2Extract]])
    start = StartRel[nFeature2Extract]
    end = start + LengthRel[nFeature2Extract]
    featureOut["x"] = featureIn["x"][int(start) : int(end)]
    featureOut["y"] = featureIn["y"][int(start) : int(end)]
    if "z" in featureIn.keys():
        featureOut["z"] = featureIn["z"][int(start) : int(end)]

    return featureOut


def writeLine2SHPfile(lineDict, lineName, fileName, header=""):
    """write a line to shapefile

    Parameters
    ----------
    lineDict: dict
        line dict
    lineName: str
        line name
    fileName: str or pathlib path
        path where the line will be saved
        line name
    header: dict
        optional argument ('' by default). If provided, header dictionary with 'xllcenter' and 'yllcenter' to add to the
        line
    Returns
    -------
    fileName : str
        path where the line has been saved
    """
    fileName = str(fileName)
    line = np.zeros((np.size(lineDict["x"]), 2))
    line[:, 0] = lineDict["x"]
    line[:, 1] = lineDict["y"]
    if header:
        line[:, 0] = line[:, 0] + header["xllcenter"]
        line[:, 1] = line[:, 1] + header["yllcenter"]
    w = shapefile.Writer(fileName)
    w.field("name", "C")
    w.line([line])
    w.record(lineName)
    w.close()
    return fileName


def writePoint2SHPfile(pointDict, pointName, fileName):
    """write a point to shapefile

    Parameters
    ----------
    pointDict: dict
        point dict
    pointName: str
        point name
    fileName: str or pathlib path
        path where the point will be saved
    Returns
    -------
    fileName : str
        path where the point has been saved
    """
    fileName = str(fileName)
    w = shapefile.Writer(fileName)
    w.field("name", "C")
    if isinstance(pointDict["x"], (float, np.float64)) and isinstance(pointDict["y"], (float, np.float64)):
        w.point(pointDict["x"], pointDict["y"])
    elif isinstance(pointDict["x"], (np.ndarray)) and isinstance(pointDict["y"], (np.ndarray)):
        if len(pointDict["x"]) > 1 or len(pointDict["y"]) > 1:
            message = "Length of pointDict is not allowed to exceed one"
            log.error(message)
            raise ValueError(message)
        else:
            w.point(pointDict["x"][0], pointDict["y"][0])
    else:
        message = "Format of point is neither float nor array of length 1"
        log.error(message)
        raise TypeError(message)
    w.record(pointName)
    w.close()
    return fileName


def writeShapefile(outputPath, fields, fieldNames, features, srs=None):
    """Write features to a shapefile with given fields and properties.

    Parameters
    ----------
    outputPath: pathlib.Path
        path where the shapefile will be written
    fields: list
        the fields of the shapefile
    fieldNames: list
        the names of the fields
    features: list
        list of tuples containing (properties, geometry) for each feature
    srs: str, optional
        the spatial reference system for the .prj file

    Returns
    -------
    none
    """
    with shapefile.Writer(str(outputPath)) as dst:
        for field in fields:
            dst.field(*field)
        for properties, geometry in features:
            dst.shape(geometry)
            record = [properties.get(fieldName, "") for fieldName in fieldNames]
            dst.record(*record)

    if srs is not None:
        prjOutPath = outputPath.with_suffix(".prj")
        with open(prjOutPath, "w") as prjFile:
            prjFile.write(srs)


def readShapefile(inputShp):
    """Read the fields, properties, geometries, and spatial reference of an input shapefile.
    To be used in combination with shapefile.Reader. Could be expanded upon to get e.g.
    shapeTypes, bounds, numFeatures and metadata if needed. This does pure reading, no special
    checks for valid attributes are performed.

    Parameters
    ----------
    inputShp : pathlib.Path
        Path to input shapefile

    Returns
    -------
    fields: list
        the fields of the input shapefile
    fieldNames: list
        the names of the fields
    properties: list
        a list of dictionaries containing the properties of each feature
    geometries: list
        a list of geometry objects
    srs: str
        the spatial reference system fetched from eventual .prj file
    """
    with shapefile.Reader(str(inputShp)) as src:
        fields = src.fields[1:]  # Skip deletion flag
        fieldNames = [field[0] for field in fields]
        properties = []
        geometries = []
        for shapeRecord in src.iterShapeRecords():
            if shapeRecord.shape.shapeType == shapefile.NULL:
                log.warning(f"Skipping NULL shape in {inputShp}")
                continue
            properties.append(dict(zip(fieldNames, shapeRecord.record)))
            geometries.append(shape(shapeRecord.shape.__geo_interface__))

        srs = None
        # Check if .prj file exists and read it
        srsfile = inputShp.with_suffix(".prj")
        if srsfile.is_file():
            with open(srsfile, "r") as f:
                srs = f.read().strip()

    return fields, fieldNames, properties, geometries, srs
