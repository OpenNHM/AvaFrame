"""
  Create generic/idealised topographies
"""

# load modules
import logging
import numpy as np
import math
from scipy.stats import norm
from scipy.interpolate import griddata
import pathlib
from rasterio.crs import CRS

from avaframe.in3Utils import geoTrans
import avaframe.in2Trans.rasterUtils as IOf

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def getParabolaParams(cfg):
    """Compute parameters for parabola"""
    # input parameters
    C = float(cfg["TOPO"]["C"])
    fLens = float(cfg["TOPO"]["fLens"])
    meanAlpha = float(cfg["TOPO"]["meanAlpha"])
    
    # New parameter for tilted plane option
    tiltRunout = float(cfg["TOPO"].get("tiltRunout", 0))  # inclination of straight section
    
    # If tiltRunout is specified, compute junction point where parabola slope = tiltRunout
    if tiltRunout != 0:
        # For a descending parabola z = C + Bx + Ax² where:
        # - Starts at (0, C) with initial slope B (negative for descending)
        # - Derivative: dz/dx = B + 2Ax
        # - At junction x_j: B + 2Ax_j = -tan(tiltRunout) (negative for descending)
        
        # Given meanAlpha, we want the average slope from (0,C) to (x_j, z_j) to be -tan(meanAlpha)
        # Average slope = (z_j - C) / x_j = -tan(meanAlpha)
        # So: z_j = C - x_j * tan(meanAlpha)
        
        # Also, z_j = C + B*x_j + A*x_j²
        # Therefore: C - x_j*tan(meanAlpha) = C + B*x_j + A*x_j²
        # Simplifying: -x_j*tan(meanAlpha) = B*x_j + A*x_j²
        # Dividing by x_j: -tan(meanAlpha) = B + A*x_j  ... (equation 1)
        
        # At junction, slope must equal tiltRunout:
        # B + 2A*x_j = -tan(tiltRunout)  ... (equation 2)
        
        # From (1): B = -tan(meanAlpha) - A*x_j
        # Substitute into (2): -tan(meanAlpha) - A*x_j + 2A*x_j = -tan(tiltRunout)
        # Simplify: -tan(meanAlpha) + A*x_j = -tan(tiltRunout)
        # So: A*x_j = -tan(tiltRunout) + tan(meanAlpha)
        # Therefore: A = (tan(meanAlpha) - tan(tiltRunout)) / x_j
        
        # We still need to determine x_j. Use the total fall height C:
        # From z_j = C - x_j*tan(meanAlpha) and the parabola must drop by C total
        # Let's say the parabola section drops by C, so z_j = 0
        # Then: 0 = C - x_j*tan(meanAlpha)
        # So: x_j = C / tan(meanAlpha)
        
        tanMean = np.tan(np.radians(meanAlpha))
        tanTilt = np.tan(np.radians(tiltRunout))
        
        fLen = C / tanMean
        A = (tanMean - tanTilt) / fLen
        B = -tanMean - A * fLen
        
        log.info("Tilted plane mode: junction at x = %.2f meters" % fLen)
        log.info("Parabola slope at junction: %.2f degrees" % tiltRunout)
        log.info("A = %.6f, B = %.6f" % (A, B))
        
    # Original flat plane logic
    elif meanAlpha != 0:
        fLen = C / np.tan(np.radians(meanAlpha))
        log.info("fLen computed from mean alpha: %.2f meters" % fLen)
        A = C / (fLen**2)
        B = (-C * 2.0) / fLen
    else:
        fLen = fLens
        log.info("flen directly set to: %.2f meters" % fLen)
        A = C / (fLen**2)
        B = (-C * 2.0) / fLen
    
    return A, B, fLen


def getGridDefs(cfg):
    # determine number of rows and columns to define domain
    dx = float(cfg["TOPO"]["dx"])
    xEnd = float(cfg["TOPO"]["xEnd"])
    yEnd = float(cfg["TOPO"]["yEnd"])

    return dx, xEnd, yEnd


def computeCoordGrid(dx, xEnd, yEnd):

    # Compute coordinate grid
    xv = np.arange(0, xEnd + dx, dx)
    yv = np.arange(-0.5 * yEnd, 0.5 * (yEnd + dx), dx)
    nRows = len(yv)
    nCols = len(xv)
    x, y = np.meshgrid(xv, yv)
    zv = np.zeros((nRows, nCols))

    return xv, yv, zv, x, y, nRows, nCols


def flatplane(cfg):
    """Compute coordinates of flat plane topography"""

    dx, xEnd, yEnd = getGridDefs(cfg)

    zElev = float(cfg["TOPO"]["zElev"])

    xv, yv, zv, x, y, nRows, nCols = computeCoordGrid(dx, xEnd, yEnd)

    # Set elevation of surface
    zv = zv + zElev

    # Log info here
    log.info("Flatplane coordinates computed")

    return x, y, zv


def inclinedplane(cfg):
    """Compute coordinates of inclined plane with given slope (meanAlpha)"""

    # input parameters
    dx, xEnd, yEnd = getGridDefs(cfg)

    z0 = float(cfg["TOPO"]["z0"])
    meanAlpha = float(cfg["TOPO"]["meanAlpha"])

    cFf = float(cfg["CHANNELS"]["cff"])
    cRadius = float(cfg["CHANNELS"]["cRadius"])

    xv, yv, zv, x, y, nRows, nCols = computeCoordGrid(dx, xEnd, yEnd)

    # Set surface elevation from slope and max. elevation
    zv = z0 - np.tan(np.radians(meanAlpha)) * x

    # If a channel shall be introduced
    if cfg["TOPO"].getboolean("channel"):
        # Compute cumulative distribution function and set horizontal extent of channel
        c0 = norm.cdf(xv, 0, cFf)
        cExtent = cRadius
        yv = np.reshape(yv, (nRows, 1))

        # if location within horizontal extent of channel,
        # make half sphere shaped channel with radius given by channel horizontal extent
        mask = np.zeros(np.shape(yv))
        mask[np.where(abs(yv) < cExtent)] = 1
        if cfg["TOPO"].getboolean("topoAdd"):
            zv = zv + cExtent * c0 * (1.0 - np.sqrt(np.abs(1.0 - (np.square(yv) / (cExtent ** 2))))) * mask
            mask = np.zeros(np.shape(yv))
            mask[np.where(abs(yv) >= cExtent)] = 1
            zv = zv + cExtent * c0 * mask
        else:
            zv = zv - cExtent * c0 * np.sqrt(np.abs(1.0 - (np.square(yv) / (cExtent ** 2)))) * mask

    # Log info here
    log.info("Inclined plane coordinates computed")

    return x, y, zv


def addDrop(cfg, x, y, zv):
    """Add a drop to a given topography

    The drop is added in the x drection

    Parameters
    ----------
    cfg: configparser
        configuration for generateTopo
    x: 2D numpy array
        x coordinate of the raster
    y: 2D numpy array
        y coordinate of the raster
    zv: 2D numpy array
        z coordinate of the raster

    Returns
    -------
    zv: 2D numpy array
        z coordinate of the raster taking the drop into account
    """

    # input parameters
    dx, xEnd, yEnd = getGridDefs(cfg)

    xStartDrop = float(cfg["DROP"]["xStartDrop"])
    dxDrop = float(cfg["DROP"]["dxDrop"])
    alphaDrop = float(cfg["DROP"]["alphaDrop"])

    # get zcoord
    # deduce drop height from the drop steepness and length in x direction
    dzDrop = dxDrop * np.tan(np.radians(alphaDrop))
    xEndDrop = xStartDrop + dxDrop

    nrows, ncols = np.shape(x)
    # get the z coordinate of the point at the beginning of the drop
    zIniDrop, _ = geoTrans.projectOnGrid(
        xStartDrop * np.ones((nrows)),
        y[:, 0],
        np.vstack((zv[0, :], zv)),
        csz=dx,
        xllc=x[0, 0],
        yllc=y[0, 0],
        interp="bilinear",
    )
    zIniDrop = np.tile(zIniDrop, (ncols, 1)).transpose()
    # get the z coordinate of the point at the end of the drop
    zEndDrop, _ = geoTrans.projectOnGrid(
        xEndDrop * np.ones((nrows)),
        y[:, 0],
        np.vstack((zv[0, :], zv)),
        csz=dx,
        xllc=x[0, 0],
        yllc=y[0, 0],
        interp="bilinear",
    )
    zEndDrop = np.tile(zEndDrop, (ncols, 1)).transpose()
    # Set surface elevation from slope and max. elevation
    zv = np.where(
        ((x >= xStartDrop) & (x <= xEndDrop)),
        zIniDrop - (x - xStartDrop) * np.tan(np.radians(alphaDrop)),
        zv,
    )
    zv = np.where(x > xEndDrop, zv - (dzDrop + zEndDrop - zIniDrop), zv)

    # Log info here
    log.info("Added drop to the topography")

    return zv


def hockey(cfg):
    """
    Compute coordinates of an inclined plane with a flat or inclined foreland defined by
    total fall height z0, angle to foreland (meanAlpha), foreland angle (tiltRunout),
    and a radius (rCirc) to smooth the transition between planes
    
    Note: meanAlpha and tiltRunout are provided as POSITIVE values but represent
    DESCENDING slopes ! 
    Example: meanAlpha=35 creates a -35° slope, tiltRunout=5 creates a -5° slope
    meanAlpha should be larger than tiltRunout for typical runout geometry.

    Parameters
    ----------
    cfg dictionary of dictionaries
        parameters from configuration file

    Returns
    ----------

    x, y, zv 2D numpy arrays
        x, y and z coordinates of the hockey-stick topography in 2D arrays
    """
# input parameters
    rCirc = float(cfg["TOPO"]["rCirc"])
    meanAlpha = float(cfg["TOPO"]["meanAlpha"])
    z0 = float(cfg["TOPO"]["z0"])
    
    # New parameter for foreland inclination (default 0 for flat)
    tiltRunout = float(cfg["TOPO"].get("tiltRunout", 0))
    
    cff = float(cfg["CHANNELS"]["cff"])
    cRadius = float(cfg["CHANNELS"]["cRadius"])
    cInit = float(cfg["CHANNELS"]["cInit"])
    cMustart = float(cfg["CHANNELS"]["cMustart"])
    cMuendFP = float(cfg["CHANNELS"]["cMuendFP"])
    dx, xEnd, yEnd = getGridDefs(cfg)
    
    # Compute coordinate grid
    xv, yv, zv, x, y, nRows, nCols = computeCoordGrid(dx, xEnd, yEnd)
    
    # Compute distance to foreland transition for given meanAlpha
    # Note: meanAlpha is provided as positive but represents a descending slope
    x1 = z0 / np.tan(np.radians(meanAlpha))
    
    if x1 >= xEnd * 0.9:
        log.warning(
            "Your domain (xEnd) is too small or the slope angle (meanAlpha) too "
            "shallow to produce a significant (>10 percent of domain, in your case:"
            " %.2f m) foreland!" % (0.1 * (xEnd - dx))
        )
   
    # Compute circle parameters for smoothing the transition
    # Both angles are provided as positive values representing descending slopes
    # The actual angle between the two planes needs to account for their orientation
    # from horizontal: upper slope goes down at -meanAlpha, lower slope at -tiltRunout
    deltaAlpha = (180.0 - meanAlpha) + tiltRunout  # Interior angle between the two slopes
    
    if deltaAlpha <= 0 or deltaAlpha >= 180:
        log.warning(
            f"Invalid geometry: meanAlpha={meanAlpha}°, tiltRunout={tiltRunout}° "
            f"produces deltaAlpha={deltaAlpha}°. Check your angle values."
        )
    
    if meanAlpha <= tiltRunout:
        log.warning(
            f"meanAlpha ({meanAlpha}°) should be larger than tiltRunout ({tiltRunout}°) "
            "for typical runout geometry. The transition may not work as expected."
        )
    
    # Height at the kink point (where slopes would meet without smoothing)
    z1 = z0 - np.tan(np.radians(meanAlpha)) * x1
    log.info(f"found kink point at {x1, z1} - note z1 should be 0")
    
    # Calculate geometry for circular transition between two slopes
    # Both slopes descend (angles provided as positive values)
    
    # Distance from perpendicular to circle center
    # This is the key: perpendicular distance from each slope to the circle center
    perpDist = rCirc
    
    # Distance along each slope from kink point to tangent point
    # Using geometry: the tangent points are equidistant from the kink along their respective slopes
    tangentDist = rCirc / np.tan(np.radians(deltaAlpha / 2.0))
    
    # Tangent point on upper slope (going backward in x from x1)
    # Moving up along the slope means: -dx in x-direction, +dz in z-direction
    x_tangent_upper = x1 - tangentDist * np.cos(np.radians(meanAlpha))
    z_tangent_upper = z1 + tangentDist * np.sin(np.radians(meanAlpha))

    log.info(f"upper tangent point at {x_tangent_upper, z_tangent_upper}")
    
    # Tangent point on lower slope (going forward in x from x1)
    # Moving down along the runout means: +dx in x-direction, -dz in z-direction
    x_tangent_lower = x1 + tangentDist * np.cos(np.radians(tiltRunout))
    z_tangent_lower = z1 - tangentDist * np.sin(np.radians(tiltRunout))

    log.info(f"lower tangent point at {x_tangent_lower, z_tangent_lower}")
   
    # Circle center is perpendicular distance from each tangent point
    # toward the "inside" of the angle (above both slopes)
    xCirc = x_tangent_upper + perpDist * np.cos(np.radians(90 - meanAlpha))
    zCirc = z_tangent_upper + perpDist * np.sin(np.radians(90 - meanAlpha))

    # Verify with lower tangent point (should give same result)
    xCirc_check = x_tangent_lower + perpDist * np.cos(np.radians(90 - tiltRunout))
    zCirc_check = z_tangent_lower + perpDist * np.sin(np.radians(90 - tiltRunout))

    log.info(f"circle center at {xCirc, zCirc}")
    log.info(f"circle center at {xCirc_check, zCirc_check}")
    
    # Extents of the circular transition in x-direction
    x_start = x_tangent_upper
    x_end = x_tangent_lower
    
    # For plotting
    # d1 = np.tan(np.radians(beta)) * x1
    
    # Set surface elevation
    zv = np.zeros((nRows, nCols))
    
    # Upper inclined plane (before transition)
    mask = np.zeros(np.shape(x))
    mask[np.where(x < x_start)] = 1
    zv = zv + (z0 - np.tan(np.radians(meanAlpha)) * x) * mask
    
    # Circular transition zone
    mask = np.zeros(np.shape(x))
    mask[np.where((x_start <= x) & (x <= x_end))] = 1
    
    # Calculate height on the circle
    # Circle equation: (x - xCirc)^2 + (z - zCirc)^2 = rCirc^2
    # Solve for z (taking the lower part of the circle)
    zv = zv + (zCirc - np.sqrt(np.abs(rCirc ** 2 - (x - xCirc) ** 2))) * mask
    
    # Foreland (can now be inclined)
    mask = np.zeros(np.shape(x))
    mask[np.where(x > x_end)] = 1
    
    # Height at end of circular transition (use the calculated tangent point)
    z_transition_end = z_tangent_lower
    x_transition_end = x_tangent_lower
    
    # Inclined or flat foreland
    zv = zv + (z_transition_end - np.tan(np.radians(tiltRunout)) * (x - x_transition_end)) * mask
    
    # If a channel shall be introduced
    if cfg["TOPO"].getboolean("channel"):
        # Compute cumulative distribution function - c1 for upper part (start)
        # of channel and c2 for lower part (end) of channel
        c1 = norm.cdf(xv, cMustart * (x1), cff)
        c2 = 1.0 - norm.cdf(xv, cMuendFP * (x1), cff)
        
        # combine both into one function separated at the middle of
        # the channel longprofile location
        mask = np.zeros(np.shape(xv))
        mask[np.where(xv < (x1 * (0.5 * (cMustart + cMuendFP))))] = 1
        c0 = c1 * mask
        mask = np.zeros(np.shape(xv))
        mask[np.where(xv >= (x1 * (0.5 * (cMustart + cMuendFP))))] = 1
        c0 = c0 + c2 * mask
        
        # Is the channel of constant width or narrowing
        if cfg["TOPO"].getboolean("narrowing"):
            cExtent = cInit * (1 - c0[:]) + (c0[:] * cRadius)
        else:
            cExtent = np.zeros(np.shape(xv)) + cRadius
        
        # Set surface elevation
        mask = np.zeros(np.shape(y))
        mask[np.where(abs(y) < cExtent)] = 1
        
        # Add surface elevation modification introduced by channel
        if cfg["TOPO"].getboolean("topoAdd"):
            zv = zv + cExtent * c0 * (1.0 - np.sqrt(np.abs(1.0 - (np.square(y) / (cExtent ** 2))))) * mask
            # outside of the channel, add layer of channel thickness
            mask = np.zeros(np.shape(y))
            mask[np.where(abs(y) >= cExtent)] = 1
            zv = zv + cExtent * c0 * mask
        else:
            zv = zv - cExtent * c0 * np.sqrt(np.abs(1.0 - (np.square(y) / (cExtent ** 2)))) * mask
    
    # Log info here
    if tiltRunout == 0:
        log.info("Hockeystick coordinates computed (flat foreland)")
    else:
        log.info(f"Hockeystick coordinates computed (inclined foreland at {tiltRunout}°)")
    
    return x, y, zv


def parabola(cfg):
    """
    Compute coordinates of a parabolically-shaped slope with flat or tilted foreland
    """
    C = float(cfg["TOPO"]["C"])
    cff = float(cfg["CHANNELS"]["cff"])
    cRadius = float(cfg["CHANNELS"]["cRadius"])
    cInit = float(cfg["CHANNELS"]["cInit"])
    cMustart = float(cfg["CHANNELS"]["cMustart"])
    cMuend = float(cfg["CHANNELS"]["cMuend"])
    
    # New parameter for tilted plane
    tiltRunout = float(cfg["TOPO"].get("tiltRunout", 0))
    
    # Get grid definitions
    dx, xEnd, yEnd = getGridDefs(cfg)
    
    # Compute coordinate grid
    xv, yv, zv, x, y, nRows, nCols = computeCoordGrid(dx, xEnd, yEnd)
    
    # Get parabola Parameters
    [A, B, fLen] = getParabolaParams(cfg)
    
    # Compute elevation at junction point
    z_junction = A * fLen**2 + B * fLen + C
    
    # Set surface elevation
    zv = np.ones((nRows, nCols))
    
    if tiltRunout != 0:
        # Tilted plane mode: flat section is replaced by inclined plane
        tanTilt = np.tan(np.radians(tiltRunout))
        
        # Use x (2D meshgrid) instead of xv (1D array)
        # For x >= fLen: use straight line with slope = -tanTilt
        # z = z_junction - tanTilt * (x - fLen)
        zv = z_junction - tanTilt * (x - fLen)
        
        # For x < fLen: use parabola
        mask = np.zeros(np.shape(x))
        mask[np.where(x < fLen)] = 1
        zv = zv * (1 - mask) + (A * x**2 + B * x + C) * mask
        
        log.info("Parabola with tilted plane (%.2f deg) computed" % tiltRunout)
        log.info("A: %.5f, B: %.5f, C: %.5f"  % (A, B, C))
        
    else:
        # Original flat plane mode
        zv = zv * ((-(B**2)) / (4.0 * A) + C)
        mask = np.zeros(np.shape(x))
        mask[np.where(x < fLen)] = 1
        zv = zv + (A * x**2 + B * x + C) * mask
        
        log.info("Parabola with flat outrun computed")
    
    # Initialize superimposed channel and dam
    superChannel = np.zeros(np.shape(x))
    superDam = np.zeros(np.shape(x))
    
    # Channel logic (unchanged)
    if cfg["TOPO"].getboolean("channel"):
        c1 = norm.cdf(x, cMustart * fLen, cff)
        c2 = 1.0 - norm.cdf(x, cMuend * fLen, cff)
        
        mask = np.zeros(np.shape(x))
        mask[np.where(x < (fLen * (0.5 * (cMustart + cMuend))))] = 1
        c0 = c1 * mask
        mask = np.zeros(np.shape(x))
        mask[np.where(x >= (fLen * (0.5 * (cMustart + cMuend))))] = 1
        c0 = c0 + c2 * mask
        
        if cfg["TOPO"].getboolean("narrowing"):
            cExtent = cInit * (1 - c0[:]) + (c0[:] * cRadius)
        else:
            cExtent = np.zeros(nCols) + cRadius
        
        mask = np.zeros(np.shape(y))
        mask[np.where(abs(y) < cExtent)] = 1
        
        if cfg["TOPO"].getboolean("topoAdd"):
            superChannel = (
                superChannel
                + cExtent * c0 * (1.0 - np.sqrt(np.abs(1.0 - (np.square(y) / (cExtent**2))))) * mask
            )
            mask = np.zeros(np.shape(y))
            mask[np.where(abs(y) >= cExtent)] = 1
            superChannel = superChannel + cExtent * c0 * mask
        else:
            superChannel = (
                superChannel - cExtent * c0 * np.sqrt(np.abs(1.0 - (np.square(y) / (cExtent**2)))) * mask
            )
    
    # Dam logic (unchanged)
    if cfg["TOPO"].getboolean("dam"):
        damPos = cfg["DAMS"].getfloat("damPos")
        damHeight = cfg["DAMS"].getfloat("damHeight")
        damWidth = cfg["DAMS"].getfloat("damWidth")
        superDam = norm.pdf(x, damPos * (-B / 2 / A), damWidth)
        superDam = superDam / np.max(superDam) * damHeight
        if not cfg["TOPO"].getboolean("topoAdd"):
            superDam = superDam - cExtent * c0
    
    # Add channel and dam
    zv = zv + np.maximum(superDam, superChannel)
    
    return x, y, zv


def parabolaRotation(cfg):
    """
    Compute coordinates of a parabolically-shaped slope with a flat foreland
    defined by total fall height C, angle (meanAlpha) or distance (fLen) to flat foreland
    One parabolic slope in x direction, one sloped with 45° and one with 60°
    """

    C = float(cfg["TOPO"]["C"])
    fFlat = float(cfg["TOPO"]["fFlat"])

    # Get grid definitons
    dx, xEnd, yEnd = getGridDefs(cfg)

    # Compute coordinate grid, with center in (0, 0)
    xv = np.arange(-0.5 * xEnd, 0.5 * (xEnd + dx), dx)
    yv = np.arange(-0.5 * yEnd, 0.5 * (yEnd + dx), dx)
    nRows = len(yv)
    nCols = len(xv)
    xv, yv = np.meshgrid(xv, yv)
    zv = np.zeros((nRows, nCols))

    # Get parabola Parameters
    [A, B, fLen] = getParabolaParams(cfg)

    # Set surface elevation
    zv = np.ones((nRows, nCols))
    zv = zv * ((-(B ** 2)) / (4.0 * A) + C)
    # compute polar coordinates
    r = np.sqrt(xv**2 + yv**2)
    theta = np.arctan2(-yv, xv)

    # add parabola aligned with x (going from left to right)
    phi = math.pi
    # rotation of the polar coord system to be aligned with the parabola direction
    s = createParabolaAxis(phi, theta, r, zv, fLen, fFlat)

    mask = np.ones(np.shape(s))
    mask[np.where(theta < 2 * math.pi / 3)] = 0
    mask[np.where(theta <= -5 * math.pi / 8)] = 1
    mask[np.where(s > fLen)] = 0
    zv = zv + (A * s ** 2 + B * s + C) * mask

    # add parabola sloped 60° with x
    phi = math.pi / 3
    # rotation of the polar coord system to be aligned with the parabola direction
    s = createParabolaAxis(phi, theta, r, zv, fLen, fFlat)

    mask = np.ones(np.shape(s))
    mask[np.where(theta > 2 * math.pi / 3)] = 0
    mask[np.where(theta < math.pi / 24)] = 0
    mask[np.where(s > fLen)] = 0
    zv = zv + (A * s ** 2 + B * s + C) * mask

    # add parabola aligned with x (going from left to right)
    phi = -math.pi / 4
    # rotation of the polar coord system to be aligned with the parabola direction
    s = createParabolaAxis(phi, theta, r, zv, fLen, fFlat)

    # apply the parabola to the corresponding part of the dem
    mask = np.ones(np.shape(s))
    mask[np.where(theta > math.pi / 24)] = 0
    mask[np.where(theta < -5 * math.pi / 8)] = 0
    mask[np.where(s > fLen)] = 0
    zv = zv + (A * s ** 2 + B * s + C) * mask

    # Log info here
    log.info("Triple parabola with flat foreland coordinates computed")

    return xv, yv, zv


def createParabolaAxis(phi, theta, r, zv, fLen, fFlat):
    """create the s coordinate for a lined sloped from theta - phi from x axis"""
    # rotation of the polar coord system to be aligned with the parabola direction
    gamma = theta - phi
    gamma = np.where(gamma < 0, gamma + 2 * math.pi, gamma)
    gamma = np.where(gamma >= 2 * math.pi, gamma - 2 * math.pi, gamma)
    # compute the s in the cartesian coord system aligned with the parabola
    s = r * np.cos(gamma)
    # shift this so that origin is at the top of the parabola
    s = -s + fLen + fFlat
    # apply the parabola to the corresponding part of the dem
    return s


def bowl(cfg):
    """Compute coordinates of sphere with given radius (rBwol)"""

    # input parameters
    rBwol = float(cfg["TOPO"]["rBowl"])

    # Get grid definitions
    dx, xEnd, yEnd = getGridDefs(cfg)

    # Compute coordinate grid
    xv, yv, zv, x, y, nRows, nCols = computeCoordGrid(dx, xEnd, yEnd)

    # recompute xv yv and x, y as they are shifted
    xv = np.arange(-0.5 * xEnd, 0.5 * (xEnd + dx), dx)
    yv = np.arange(-0.5 * yEnd, 0.5 * (yEnd + dx), dx)
    x, y = np.meshgrid(xv, yv)

    # Set surface elevation
    zv = rBwol * np.ones((nRows, nCols))
    if cfg["TOPO"].getboolean("curvedSlope"):
        radius = np.sqrt(x**2)
    else:
        radius = np.sqrt(x**2 + y**2)
    mask = np.zeros(np.shape(x))
    mask[np.where(radius <= rBwol)] = 1
    zv = zv - (rBwol * np.sqrt(np.abs(1 - (radius / rBwol) ** 2))) * mask
    if cfg["TOPO"].getboolean("curvedSlope"):
        zv[x >= 0] = 0.0

    # Log info here
    log.info("Bowl coordinates computed")

    return x, y, zv


def helix(cfg):
    """Compute coordinates of helix-shaped topography with given radius (rHelix)"""

    # input parameters
    rHelix = float(cfg["TOPO"]["rHelix"])
    C = float(cfg["TOPO"]["C"])
    cff = float(cfg["CHANNELS"]["cff"])
    cRadius = float(cfg["CHANNELS"]["cRadius"])
    cInit = float(cfg["CHANNELS"]["cInit"])
    cMustart = float(cfg["CHANNELS"]["cMustart"])
    cMuend = float(cfg["CHANNELS"]["cMuend"])

    # Get grid definitions
    dx, xEnd, yEnd = getGridDefs(cfg)

    # Compute coordinate grid
    xv, yv, zv, x, y, nRows, nCols = computeCoordGrid(dx, xEnd, yEnd)

    # recompute xv yv and x, y as they are shifted
    xv = np.arange(-0.5 * xEnd, 0.5 * (xEnd + dx), dx)
    yv = np.arange(-yEnd, 0 + dx, dx)
    x, y = np.meshgrid(xv, yv)

    # Get parabola Parameters
    [A, B, fLen] = getParabolaParams(cfg)

    # Set surface elevation
    zv = np.ones((nRows, nCols))
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) + np.pi
    zv = zv * ((-(B ** 2)) / (4.0 * A) + C)
    mask = np.zeros(np.shape(x))
    mask[np.where((theta * rHelix) < fLen)] = 1

    zv = zv + (A * (theta * rHelix) ** 2 + B * (theta * rHelix) + C) * mask

    # If channel is introduced to topography
    if cfg["TOPO"].getboolean("channel"):
        c0 = np.zeros(np.shape(x))
        mask = np.zeros(np.shape(x))
        mask[np.where((theta * rHelix) < (0.5 * (cMustart + cMuend) * fLen))] = 1
        c0 = c0 + norm.cdf(theta * rHelix, cMustart * fLen, cff) * mask
        mask = np.zeros(np.shape(x))
        mask[np.where((theta * rHelix) >= (0.5 * (cMustart + cMuend) * fLen))] = 1
        c0 = c0 + (1.0 - norm.cdf(theta * rHelix, cMuend * fLen, cff)) * mask
        # c0 = np.ones(np.shape(zv))
        # If channel of constant width or becoming narrower in the middle
        if cfg["TOPO"].getboolean("narrowing"):
            cExtent = cInit * (1.0 - c0) + c0 * cRadius
        else:
            cExtent = cRadius

        if cfg["TOPO"].getboolean("topoAdd"):
            zv = zv + c0 * cExtent

        # Inner and outer boundaries of the channel
        boundIn = rHelix - cExtent
        boundExt = rHelix + cExtent

        # Set channel
        mask = np.zeros(np.shape(x))
        mask[np.where((radius >= rHelix) & (radius < boundExt))] = 1
        radius1 = radius - rHelix
        zv = zv - cExtent * c0 * np.sqrt(np.abs(1.0 - (np.square(radius1) / np.square(cExtent)))) * mask

        mask = np.zeros(np.shape(x))
        mask[np.where((radius < rHelix) & (radius > boundIn))] = 1
        radius2 = rHelix - radius
        zv = zv - cExtent * c0 * np.sqrt(np.abs(1.0 - (np.square(radius1) / np.square(cExtent)))) * mask

    # set last row at Center to fall height
    indCols = int(0.5 * nCols)
    zv[-1, 0:indCols] = C

    # Log info here
    log.info("Helix coordinates computed")

    return x, y, zv


def pyramid(cfg):
    """Generate a pyramid topography - in this case rectangular domain"""

    # get parameters from ini
    meanAlpha = float(cfg["TOPO"]["meanAlpha"])
    z0 = float(cfg["TOPO"]["z0"])
    flatx = float(cfg["TOPO"]["flatx"])
    flaty = float(cfg["TOPO"]["flaty"])
    phi = float(cfg["TOPO"]["phi"])
    dx = float(cfg["TOPO"]["dx"])

    # initialise pyramid corners and center point
    points = np.asarray([[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1, 0.0], [0.0, 0.0, 1.0]])
    dxPoints = abs(points[4, 0] - points[1, 0])

    # compute elevation of the apex point for given angle of pyramid facets
    zAlpha = dxPoints * np.tan(np.deg2rad(meanAlpha))
    points[4, 2] = zAlpha
    dcoors = points * z0 / zAlpha

    # if desired rotate pyramid
    if cfg["TOPO"].getboolean("flagRot"):
        dcoorsRot = np.zeros((len(dcoors), 3))
        for m in range(len(dcoorsRot)):
            dcoorsRot[m, 0] = np.cos(np.deg2rad(phi)) * dcoors[m, 0] - np.sin(np.deg2rad(phi)) * dcoors[m, 1]
            dcoorsRot[m, 1] = np.sin(np.deg2rad(phi)) * dcoors[m, 0] + np.cos(np.deg2rad(phi)) * dcoors[m, 1]
            dcoorsRot[m, 2] = dcoors[m, 2]
        dcoorsFin = dcoorsRot
    else:
        dcoorsFin = dcoors

    # split into horizontal and vertical coordinate points
    xyPoints = np.zeros((len(points), 2))
    xyPoints[:, 0] = dcoorsFin[:, 0]
    xyPoints[:, 1] = dcoorsFin[:, 1]
    zPoints = dcoorsFin[:, 2]

    # make meshgrid for final DEM
    xv = np.arange(-flatx + np.amin(dcoorsFin[:, 0]), np.amax(dcoorsFin[:, 0]) + flatx, dx)
    yv = np.arange(-flaty + np.amin(dcoorsFin[:, 1]), np.amax(dcoorsFin[:, 1]) + flaty, dx)
    x, y = np.meshgrid(xv, yv)

    # interpolate appex point information to meshgrid
    z = griddata(xyPoints, zPoints, (x, y), method="linear")
    zNan = np.isnan(z)
    z[zNan] = 0.0

    dX = np.amax(dcoorsFin[:, 0]) + flatx - (-flatx + np.amin(dcoorsFin[:, 0]))
    dY = np.amax(dcoorsFin[:, 1]) + flaty - (-flaty + np.amin(dcoorsFin[:, 1]))
    log.info("domain extent pyramid- inx: %f, in y: %f" % (dX, dY))

    return x, y, z


def writeDEM(cfg, z, outDir):
    """Write topography information to file"""
    nameExt = cfg["TOPO"]["demType"]
    nRows = z.shape[0]
    nCols = z.shape[1]

    # Read lower left center coordinates, cellsize and noDATA value
    xllcenter = float(cfg["DEMDATA"]["xl"])
    yllcenter = float(cfg["DEMDATA"]["yl"])
    cellsize = float(cfg["TOPO"]["dx"])
    noDATA = float(cfg["DEMDATA"]["nodata_value"])
    demName = cfg["DEMDATA"]["demName"]

    # Save elevation data to file and add header lines
    demFile = outDir / ("%s_%s_Topo" % (demName, nameExt))
    demHeader = {
        "ncols": nCols,
        "nrows": nRows,
        "xllcenter": xllcenter,
        "yllcenter": yllcenter,
        "cellsize": cellsize,
        "nodata_value": noDATA,
    }

    transform = IOf.transformFromASCHeader(demHeader)
    demHeader["transform"] = transform
    demHeader["driver"] = "AAIGrid"
    demHeader["crs"] = CRS()

    IOf.writeResultToRaster(demHeader, z, demFile, flip=False)

    # Log info here
    log.info("DEM written to: %s/%s_%s_Topo" % (outDir, demName, nameExt))


def generateTopo(cfg, avalancheDir):
    """Compute coordinates of desired topography with given inputs"""

    # Which DEM type
    demType = cfg["TOPO"]["demType"]

    log.info("DEM type is set to: %s" % demType)

    # Set Output directory
    outDir = pathlib.Path(avalancheDir, "Inputs")
    if outDir.is_dir():
        log.info("The new DEM is saved to %s" % (outDir))
    else:
        log.error(
            "Required folder structure: NameOfAvalanche/Inputs missing! \
                    Run runInitializeProject first!"
        )

    # Call topography type
    if demType == "FP":
        [x, y, z] = flatplane(cfg)

    elif demType == "IP":
        [x, y, z] = inclinedplane(cfg)

    elif demType == "PF":
        [x, y, z] = parabola(cfg)

    elif demType == "TPF":
        [x, y, z] = parabolaRotation(cfg)

    elif demType == "HS":
        [x, y, z] = hockey(cfg)

    elif demType == "BL":
        [x, y, z] = bowl(cfg)

    elif demType == "HX":
        [x, y, z] = helix(cfg)

    elif demType == "PY":
        [x, y, z] = pyramid(cfg)

    # If a drop shall be introduced
    if cfg["TOPO"].getboolean("drop"):
        z = addDrop(cfg, x, y, z)

    # moves topo in z direction
    if cfg["DEMDATA"]["zEdit"] != "":
        z = z + cfg["DEMDATA"].getfloat("zEdit")
        log.info("Changed topo elevation by %.2f" % cfg["DEMDATA"].getfloat("zEdit"))

    # Write DEM to file
    writeDEM(cfg, z, outDir)

    return (z, demType, outDir)
