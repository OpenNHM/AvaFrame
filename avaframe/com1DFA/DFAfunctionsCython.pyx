#!python
# cython: boundscheck=False, wraparound=False, cdivision=True
"""
    function related to SPH calculations in com1DFA
    to build: go to repository containing this file and run:
    python setup.py build_ext --inplace
"""

# Load modules
import copy
import logging
import numpy as np
import cython
cimport numpy as np
from libc cimport math as math

# Local imports
import avaframe.com1DFA.DFAtools as DFAtls
cimport avaframe.com1DFA.DFAToolsCython as DFAtlsC
cimport avaframe.com1DFA.damCom1DFA as damCom1DFA
import avaframe.com1DFA.particleTools as particleTools
import avaframe.in3Utils.geoTrans as geoTrans


# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def computeForceC(cfg, particles, fields, dem, int frictType, int resistanceType):
  """ compute forces acting on the particles (without the SPH component)

  Cython implementation implementation

  Parameters
  ----------
  cfg: configparser
      configuration for DFA simulation
  particles : dict
      particles dictionary at t
  fields: dict
    fields dictionary
  dem : dict
      dictionary with dem information
  frictType: int
    identifier for friction law to be used
  resistanceType: int
    identifier for resistance model to be used

  Returns
  -------
  force : dict
      force dictionary
  particles : dict
  """
  cdef double enthRef = cfg.getfloat('enthRef')
  cdef double muSamosAt = cfg.getfloat('musamosat')
  cdef double tau0SamosAt = cfg.getfloat('tau0samosat')
  cdef double Rs0SamosAt = cfg.getfloat('Rs0samosat')
  cdef double kappaSamosAt = cfg.getfloat('kappasamosat')
  cdef double BSamosAt = cfg.getfloat('Bsamosat')
  cdef double RSamosAt = cfg.getfloat('Rsamosat')
  cdef double muSamosAtSmall = cfg.getfloat('musamosatsmall')
  cdef double tau0SamosAtSmall = cfg.getfloat('tau0samosatsmall')
  cdef double Rs0SamosAtSmall = cfg.getfloat('Rs0samosatsmall')
  cdef double kappaSamosAtSmall = cfg.getfloat('kappasamosatsmall')
  cdef double BSamosAtSmall = cfg.getfloat('Bsamosatsmall')
  cdef double RSamosAtSmall = cfg.getfloat('Rsamosatsmall')
  cdef double muSamosAtMedium = cfg.getfloat('musamosatmedium')
  cdef double tau0SamosAtMedium = cfg.getfloat('tau0samosatmedium')
  cdef double Rs0SamosAtMedium = cfg.getfloat('Rs0samosatmedium')
  cdef double kappaSamosAtMedium = cfg.getfloat('kappasamosatmedium')
  cdef double BSamosAtMedium = cfg.getfloat('Bsamosatmedium')
  cdef double RSamosAtMedium = cfg.getfloat('Rsamosatmedium')
  cdef double entEroEnergy = cfg.getfloat('entEroEnergy')
  cdef double entShearResistance = cfg.getfloat('entShearResistance')
  cdef double entDefResistance = cfg.getfloat('entDefResistance')
  cdef double rho = cfg.getfloat('rho')
  cdef double rhoEnt = cfg.getfloat('rhoEnt')
  cdef double gravAcc = cfg.getfloat('gravAcc')
  cdef double xsiVoellmy = cfg.getfloat('xsivoellmy')
  cdef double muVoellmy = cfg.getfloat('muvoellmy')
  cdef double xsiVoellmyMinShear = cfg.getfloat('xsivoellmyminshear')
  cdef double muVoellmyMinShear = cfg.getfloat('muvoellmyminshear')
  cdef double tau0VoellmyMinShear = cfg.getfloat('tau0voellmyminshear')
  cdef double muCoulomb = cfg.getfloat('mucoulomb')
  cdef double muCoulombMinShear = cfg.getfloat('mucoulombminshear')
  cdef double tau0CoulombMinShear = cfg.getfloat('tau0coulombminshear')
  cdef double rhoSediment = cfg.getfloat('rhoSediment')
  cdef double sizeSediment = cfg.getfloat('sizeSediment')
  cdef double cvMaxSediment = cfg.getfloat('cvMaxSediment')
  cdef double cvSediment = cfg.getfloat('cvSediment')
  cdef double alpha1EtaObrienAndJulien = cfg.getfloat('alpha1EtaObrienAndJulien')
  cdef double beta1EtaObrienAndJulien = cfg.getfloat('beta1EtaObrienAndJulien')
  cdef double alpha2TauyObrienAndJulien = cfg.getfloat('alpha2TauyObrienAndJulien')
  cdef double beta2TauyObrienAndJulien = cfg.getfloat('beta2TauyObrienAndJulien')
  cdef double alphaObrienAndJulien = cfg.getfloat('alphaObrienAndJulien')
  cdef double alpha1EtaHerschelAndBulkley = cfg.getfloat('alpha1EtaHerschelAndBulkley')
  cdef double beta1EtaHerschelAndBulkley = cfg.getfloat('beta1EtaHerschelAndBulkley')
  cdef double alpha2TauyHerschelAndBulkley = cfg.getfloat('alpha2TauyHerschelAndBulkley')
  cdef double beta2TauyHerschelAndBulkley = cfg.getfloat('beta2TauyHerschelAndBulkley')
  cdef double nHerschelAndBulkley = cfg.getfloat('nHerschelAndBulkley')
  cdef double alpha1EtaBingham = cfg.getfloat('alpha1EtaBingham')
  cdef double beta1EtaBingham = cfg.getfloat('beta1EtaBingham')
  cdef double alpha2TauyBingham = cfg.getfloat('alpha2TauyBingham')
  cdef double beta2TauyBingham = cfg.getfloat('beta2TauyBingham')
  cdef double curvAccInFriction = cfg.getfloat('curvAccInFriction')
  cdef double curvAccInTangent = cfg.getfloat('curvAccInTangent')
  cdef int curvAccInGradient = cfg.getint('curvAccInGradient')
  cdef double velMagMin = cfg.getfloat('velMagMin')
  cdef double depMin = cfg.getfloat('depMin')
  cdef int interpOption = cfg.getint('interpOption')
  cdef int explicitFriction = cfg.getint('explicitFriction')
  cdef int reprojMethod = cfg.getint('reprojMethodForce')
  cdef int reprojectionIterations = cfg.getint('reprojectionIterations')
  cdef double thresholdProjection = cfg.getfloat('thresholdProjection')
  cdef double subgridMixingFactor = cfg.getfloat('subgridMixingFactor')
  cdef int viscOption = cfg.getint('viscOption')
  cdef double dt = particles['dt']
  cdef double mu0 = cfg.getfloat('mu0wetsnow')
  cdef double xsiWetSnow = cfg.getfloat('xsiwetsnow')
  cdef int nPart = particles['nPart']
  cdef double csz = dem['header']['cellsize']
  cdef int nrows = dem['header']['nrows']
  cdef int ncols = dem['header']['ncols']
  cdef double[:, :] ZDEM = dem['rasterData']
  cdef double[:, :] nxArray = dem['Nx']
  cdef double[:, :] nyArray = dem['Ny']
  cdef double[:, :] nzArray = dem['Nz']
  cdef double[:, :] areaRatser = dem['areaRaster']
  cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] outOfDEM
  outOfDEM = np.array(dem['outOfDEM'], dtype=bool)
  # read particles and fields
  cdef double[:] mass = particles['m']
  cdef double[:] hArray = particles['h']
  cdef double[:] xArray = particles['x']
  cdef double[:] yArray = particles['y']
  cdef double[:] zArray = particles['z']
  cdef double[:] uxArray = particles['ux']
  cdef double[:] uyArray = particles['uy']
  cdef double[:] uzArray = particles['uz']
  cdef long long[:] ID = particles['ID']
  cdef double[:] totalEnthalpyArray = particles['totalEnthalpy']
  cdef double[:, :] VX = fields['Vx']
  cdef double[:, :] VY = fields['Vy']
  cdef double[:, :] VZ = fields['Vz']
  cdef double[:, :] entrMassRaster = fields['entrMassRaster']
  cdef double[:, :] entrEnthRaster = fields['entrEnthRaster']
  cdef double[:, :] muRaster = fields['muField']
  cdef double[:, :] xsiRaster = fields['xiField']
  cdef double[:, :] detRaster = fields['detRaster']
  cdef double[:, :] cResRaster = fields['cResRaster']
  cdef int[:] indXDEM = particles['indXDEM']
  cdef int[:] indYDEM = particles['indYDEM']
  # initialize outputs
  cdef double[:] forceX = np.zeros(nPart, dtype=np.float64)
  cdef double[:] forceY = np.zeros(nPart, dtype=np.float64)
  cdef double[:] forceZ = np.zeros(nPart, dtype=np.float64)
  cdef double[:] forceFrict = np.zeros(nPart, dtype=np.float64)
  cdef double[:] dM = np.zeros(nPart, dtype=np.float64)
  cdef double[:] dMDet = np.zeros(nPart, dtype=np.float64)
  cdef double[:] gEff = np.zeros(nPart, dtype=np.float64)
  cdef double[:] curvAcc = np.zeros(nPart, dtype=np.float64)
  # declare intermediate step variables
  cdef int indCellX, indCellY
  cdef double areaPart, areaCell, areaEntrPart, cResCell, cResPart, uMag, uMagRes, m, dm, h, entrMassCell, entrEnthCell, dEnergyEntr, dis
  cdef double dmDet, detCell, areaDetPart
  cdef double x, y, z, xEnd, yEnd, zEnd, ux, uy, uz, uxDir, uyDir, uzDir, totalEnthalpy, enthalpy, dTotalEnthalpy
  cdef double nx, ny, nz, nxEnd, nyEnd, nzEnd, nxAvg, nyAvg, nzAvg
  cdef double gravAccNorm, accNormCurv, effAccNorm, gravAccTangX, gravAccTangY, gravAccTangZ, forceBotTang, sigmaB, tau
  cdef double muVoellmyRaster, xsiVoellmyRaster
  cdef double shearRate, etaObrienAndJulien, tauyObrienAndJulien, lmObrienAndJulien
  cdef double lambdaBagnold, cObrienAndJulien, etaHerschelAndBulkley, tauyHerschelAndBulkley, etaBingham, tauyBingham                                                                          
  # variables for interpolation
  cdef int Lx0, Ly0, LxEnd0, LyEnd0, iCell, iCellEnd
  cdef double w[4]
  cdef double wEnd[4]
  cdef int k

  force = {}
  # loop on particles
  for k in range(nPart):
      m = mass[k]
      x = xArray[k]
      y = yArray[k]
      z = zArray[k]
      h = hArray[k]
      if h < depMin:
        h = depMin
      ux = uxArray[k]
      uy = uyArray[k]
      uz = uzArray[k]
      indCellX = indXDEM[k]
      indCellY = indYDEM[k]
      # deduce area
      areaPart = m / (h * rho)
      # get cell and weights
      Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = DFAtlsC.getCellAndWeights(x, y, ncols, nrows, csz, interpOption)

      # get normal at the particle location
      nx, ny, nz = DFAtlsC.getVector(Lx0, Ly0, w[0], w[1], w[2], w[3], nxArray, nyArray, nzArray)
      nx, ny, nz = DFAtlsC.normalize(nx, ny, nz)

      if viscOption == 1:
        # add artificial viscosity
        ux, uy, uz = addArtificialViscosity(m, h, dt, rho, ux, uy, uz, subgridMixingFactor, Lx0, Ly0,
                                            w[0], w[1], w[2], w[3], VX, VY, VZ, nx, ny, nz)

      # get normal at the particle estimated end location
      xEnd = x + dt * ux
      yEnd = y + dt * uy
      zEnd = z + dt * uz
      # this point is not necessarily on the surface, project it on the surface
      if reprojMethod == 0:
        # Project vertically on the dem
        iCellEnd = DFAtlsC.getCells(xEnd, yEnd, ncols, nrows, csz)
        if iCellEnd >= 0 and outOfDEM[iCellEnd] == 0:
          LxEnd0, LyEnd0, iCellEnd, wEnd[0], wEnd[1], wEnd[2], wEnd[3] = DFAtlsC.getCellAndWeights(xEnd, yEnd, ncols, nrows, csz, interpOption)
      elif reprojMethod == 1:
        # project trying to keep the travelled distance constant
        xEnd, yEnd, zEnd, iCellEnd, LxEnd0, LyEnd0, wEnd[0], wEnd[1], wEnd[2], wEnd[3] = DFAtlsC.distConservProjectionIteratrive(
          x, y, z, ZDEM, nxArray, nyArray, nzArray, xEnd, yEnd, zEnd, csz, ncols, nrows, interpOption,
          reprojectionIterations, thresholdProjection)
      elif reprojMethod == 2:
        # project using samos method
        xEnd, yEnd, iCellEnd, LxEnd0, LyEnd0, wEnd[0], wEnd[1], wEnd[2], wEnd[3] = DFAtlsC.samosProjectionIteratrive(
          xEnd, yEnd, zEnd, ZDEM, nxArray, nyArray, nzArray, csz, ncols, nrows, interpOption, reprojectionIterations)

      if iCellEnd < 0:
        # if not on the DEM take x, y as end point
        LxEnd0 = Lx0
        LyEnd0 = Ly0
        wEnd = w
      elif outOfDEM[iCellEnd]:
        # if in a noData area from the DEM take x, y as end point
        LxEnd0 = Lx0
        LyEnd0 = Ly0
        wEnd = w

      # get the normal at this location
      nxEnd, nyEnd, nzEnd = DFAtlsC.getVector(LxEnd0, LyEnd0, wEnd[0], wEnd[1], wEnd[2], wEnd[3], nxArray, nyArray, nzArray)
      nxEnd, nyEnd, nzEnd = DFAtlsC.normalize(nxEnd, nyEnd, nzEnd)
      # get average of those normals
      nxAvg = nx + nxEnd
      nyAvg = ny + nyEnd
      nzAvg = nz + nzEnd
      nxAvg, nyAvg, nzAvg = DFAtlsC.normalize(nxAvg, nyAvg, nzAvg)

      # acceleration due to curvature
      accNormCurv = (ux*(nxEnd-nx) + uy*(nyEnd-ny) + uz*(nzEnd-nz)) / dt
      # normal component of the acceleration of gravity
      gravAccNorm = - gravAcc * nzAvg  # use nzAvg because we want the average gNorm on the time step
      # turn off curvature with the  curvAccInFriction coefficient
      effAccNorm = gravAccNorm + curvAccInFriction * accNormCurv
      # save the normal component of the gravity (plus the curvature term if desiered)
      # this is needed to compute the pressure gradient
      # save the norm of the gEff, because we know in which direction it goes (minus the normal vector)
      if curvAccInGradient == 1:
        # use gravity plus curvature (if the effAccNorm > 0, material is detatched, then gEff = 0)
        if(effAccNorm <= 0.0):
          gEff[k] = -effAccNorm
        else:
          gEff[k] = 0
      else:
        # only use gravity
        gEff[k] = -gravAccNorm

      # body forces (tangential component of acceleration of gravity with the term due to curvature)
      gravAccTangX = - (gravAccNorm + curvAccInTangent * accNormCurv) * nxAvg
      gravAccTangY = - (gravAccNorm + curvAccInTangent * accNormCurv) * nyAvg
      gravAccTangZ = - gravAcc - (gravAccNorm + curvAccInTangent * accNormCurv) * nzAvg
      # adding gravity force contribution
      forceX[k] = forceX[k] + gravAccTangX * m
      forceY[k] = forceY[k] + gravAccTangY * m
      forceZ[k] = forceZ[k] + gravAccTangZ * m

      # Calculating bottom shear and normal stress
      # get new velocity magnitude (leave 0 if uMag is 0)
      # this is important because uMag is first used to compute tau
      uMag = DFAtlsC.norm(ux, uy, uz)
      if(-effAccNorm < 0.0):
          # if fluid detatched
          # log.info('fluid detatched for particle %s, effAccNorm %.16g' % (k, effAccNorm))
          tau = 0.0
      else:
          # bottom normal stress sigmaB
          sigmaB = - effAccNorm * rho * h
          if frictType >= 10:
            # substitution of shear rate gamma
            if math.fabs(h) <= 1e-9:
               message = "h is 0 - division by 0"
               log.error(message)
               raise ZeroDivisionError(message)
            shearRate = 3 * uMag / h
          if frictType == 1:
            # SamosAT friction type (bottom shear stress)
            tau = DFAtlsC.SamosATfric(rho, tau0SamosAt, Rs0SamosAt, muSamosAt, kappaSamosAt, BSamosAt, RSamosAt, uMag, sigmaB, h)
          elif frictType == 2:
            # coulomb friction type (bottom shear stress)
            tau = muCoulomb * sigmaB
          elif frictType == 3:
            # voellmy friction type
            tau = muVoellmy * sigmaB + rho * uMag * uMag * gravAcc / xsiVoellmy
          elif frictType == 4:
            # add enthalpy dependent mu if wetSnow is activated
            totalEnthalpy = totalEnthalpyArray[k]
            enthalpy = totalEnthalpy - gravAcc * z - 0.5 * uMag * uMag
            mu = mu0 * math.exp(-enthalpy / enthRef)
            tau = mu * sigmaB + rho * uMag * uMag * gravAcc / xsiWetSnow
          elif frictType == 5:
            # SamosAT friction type (bottom shear stress) - for small ava calibration parameters
            tau = DFAtlsC.SamosATfric(rho, tau0SamosAtSmall, Rs0SamosAtSmall, muSamosAtSmall, kappaSamosAtSmall, BSamosAtSmall, RSamosAtSmall, uMag, sigmaB, h)
          elif frictType == 6:
            # SamosAT friction type (bottom shear stress) - for medium ava calibration parameters
            tau = DFAtlsC.SamosATfric(rho, tau0SamosAtMedium, Rs0SamosAtMedium, muSamosAtMedium, kappaSamosAtMedium, BSamosAtMedium, RSamosAtMedium, uMag, sigmaB, h)
          elif frictType == 7:
            # voellmy MinShear friction type
            tau = muVoellmyMinShear * sigmaB + rho * uMag * uMag * gravAcc / xsiVoellmyMinShear + tau0VoellmyMinShear
          elif frictType == 8:
            # coulomb MinShear friction type
            tau = muCoulombMinShear * sigmaB + tau0CoulombMinShear
          elif frictType == 9:
            muVoellmyRaster = muRaster[indCellY, indCellX]
            xsiVoellmyRaster = xsiRaster[indCellY, indCellX]
            # Voellmy with optional spatially variable mu and xi values provided as rasters
            tau = muVoellmyRaster * sigmaB + rho * uMag * uMag * gravAcc / xsiVoellmyRaster
          elif frictType == 10:
            ## O`Brien and Julien
            # viscosity
            etaObrienAndJulien = alpha1EtaObrienAndJulien * math.exp(beta1EtaObrienAndJulien * cvSediment)
            # yield shear stress
            tauyObrienAndJulien = alpha2TauyObrienAndJulien * math.exp(beta2TauyObrienAndJulien * cvSediment)
            # Prandtl mixing length
            lmObrienAndJulien = 0.4 * h
            # grain concentration
            lambdaBagnold = 1.0 / (math.pow(cvMaxSediment / cvSediment, 1.0 / 3.0) - 1.0)
            # dispersive shear stress
            cObrienAndJulien = rho * lmObrienAndJulien * lmObrienAndJulien + alphaObrienAndJulien * rhoSediment * lambdaBagnold * lambdaBagnold * sizeSediment * sizeSediment
            # shear stress
            tau = tauyObrienAndJulien + etaObrienAndJulien * shearRate + cObrienAndJulien * (shearRate * shearRate)
          elif frictType == 11:
            ## Herschel and Bulkley
            # viscosity
            etaHerschelAndBulkley = alpha1EtaHerschelAndBulkley * math.exp(beta1EtaHerschelAndBulkley * cvSediment)
            # yield shear stress
            tauyHerschelAndBulkley = alpha2TauyHerschelAndBulkley * math.exp(beta2TauyHerschelAndBulkley * cvSediment)
            # shear stress
            tau = tauyHerschelAndBulkley + etaHerschelAndBulkley * math.pow(shearRate, nHerschelAndBulkley)
          elif frictType == 12:
            ## Bingham
            # viscosity
            etaBingham = alpha1EtaBingham * math.exp(beta1EtaBingham * cvSediment)
            # yield shear stress
            tauyBingham = alpha2TauyBingham * math.exp(beta2TauyBingham * cvSediment)
            # shear stress
            tau = tauyBingham + etaBingham * shearRate
          else:
            tau = 0.0

      # adding bottom shear resistance contribution
      # norm of the friction force >=0 (if 0 -> detatchment)
      forceBotTang = - areaPart * tau
      if explicitFriction == 1:
        # explicit formulation
        forceFrict[k] = forceFrict[k] - forceBotTang
      elif explicitFriction == 0:
        # make sure uMag is not 0
        if uMag<velMagMin:
          uMagRes = velMagMin
        else:
          uMagRes = uMag
        forceFrict[k] = forceFrict[k] - forceBotTang/uMagRes

      # compute entrained mass and if wetSnow case compute total enthalpy of particles with entrainment
      entrMassCell = entrMassRaster[indCellY, indCellX]
      if entrMassCell > 0.0:
        # compute entrained mass
        dm, areaEntrPart = computeEntMassAndForce(dt, entrMassCell, areaPart, uMag, tau, entEroEnergy, rhoEnt)
        # speed loss due to energy loss due to entrained mass
        dEnergyEntr = areaEntrPart * entShearResistance + dm * entDefResistance

        if frictType == 4 and dm > 0.0:
          # enthalpy change due to entrained mass
          entrEnthCell = entrEnthRaster[indCellY, indCellX]
          dTotalEnthalpy = (entrEnthCell + gravAcc * z) * dm
          totalEnthalpy = (totalEnthalpy * m + dTotalEnthalpy)

        # update velocity
        ux = ux * m / (m + dm)
        uy = uy * m / (m + dm)
        uz = uz * m / (m + dm)
        # update mass
        m = m + dm
        mass[k] = m
        dM[k] = dm

        # speed loss due to energy loss due to entrained mass
        if dEnergyEntr > 0.0:
          dis = 1.0 - dEnergyEntr / (0.5 * m * (uMag*uMag + velMagMin))
          if dis < 0.0:
            dis = 0.0
          # update velocity
          ux = ux * dis
          uy = uy * dis
          uz = uz * dis

        # if wetSnow update total enthalpy according to new particle mass
        if frictType == 4 and dm > 0.0:
          # update specific enthalpy of particle
          totalEnthalpyArray[k] = totalEnthalpy / m

      # adding detrainment analogous to entrainment
      detCell = detRaster[indCellY, indCellX]
      
      if detCell > 0:
        # compute detrained mass
        dmDet = computeDetMass(dt, detCell, areaPart, uMag)
        if abs(dmDet) > m:
          # mass of a particle can't be negative
          dmDet = - 1 * m
      
        # update mass
        m = m + dmDet
        mass[k] = m
        dMDet[k] = dmDet

      # adding resistance force due to obstacles
      cResCell = cResRaster[indCellY][indCellX]
      cResPart = computeResForce(areaPart, rho, cResCell, uMag, explicitFriction, resistanceType)
      forceFrict[k] = forceFrict[k] - cResPart

      uxArray[k] = ux
      uyArray[k] = uy
      uzArray[k] = uz

  # save results
  force['dM'] = np.asarray(dM)
  force['dMDet'] = np.asarray(dMDet)
  force['forceX'] = np.asarray(forceX)
  force['forceY'] = np.asarray(forceY)
  force['forceZ'] = np.asarray(forceZ)
  force['forceFrict'] = np.asarray(forceFrict)
  particles['gEff'] = np.asarray(gEff)
  particles['curvAcc'] = np.asarray(curvAcc)
  particles['ux'] = np.asarray(uxArray)
  particles['uy'] = np.asarray(uyArray)
  particles['uz'] = np.asarray(uzArray)
  particles['m'] = np.asarray(mass)
  particles['dmDet'] = np.asarray(dMDet)
  particles['dmEnt'] = np.asarray(dM)
  particles['totalEnthalpy'] = np.asarray(totalEnthalpyArray)

  # update mass available for entrainement
  # TODO: this allows to entrain more mass then available...
  for k in range(nPart):
    indCellX = indXDEM[k]
    indCellY = indYDEM[k]
    entrMassCell = entrMassRaster[indCellY, indCellX]
    areaCell = areaRatser[indCellY, indCellX]
    dm = dM[k]
    # update surface entrainment mass available
    entrMassCell = entrMassCell - dm/areaCell
    if entrMassCell < 0:
      entrMassCell = 0
    entrMassRaster[indCellY, indCellX] = entrMassCell
  fields['entrMassRaster'] = np.asarray(entrMassRaster)

  return particles, force, fields


cpdef (double, double) computeEntMassAndForce(double dt, double entrMassCell,
                                              double areaPart, double uMag,
                                              double tau, double entEroEnergy,
                                              double rhoEnt):
  """ compute force component due to entrained mass

  Parameters
  ----------
  dt: float
    time step
  entrMassCell : float
      available mass for entrainement
  areaPart : float
      particle area
  uMag : float
      particle speed (velocity magnitude)
  tau : float
      bottom shear stress

  Returns
  -------
  dm : float
      entrained mass
  areaEntrPart : float
      Area for entrainement energy loss computation
  entEroEnergy: float
    erosion entrainment energy constant
  rhoEnt: float
    entrainement density
  """
  cdef double width, ABotSwiped
  # compute entrained mass
  cdef double dm = 0.0
  cdef double areaEntrPart = 0.0

  if entrMassCell > 0:
      # either erosion or ploughing but not both

      if(entEroEnergy > 0):
          # erosion: erode according to shear and erosion energy
          dm = areaPart * tau * uMag * dt / entEroEnergy
          areaEntrPart = areaPart
          # TODO why not this?
          #areaEntrPart = math.sqrt(areaPart) * uMag * dt
      else:
          # ploughing in at avalanche front: erode full area weight
          # mass available in the cell [kg/m²]
          # width of the particle
          width = math.sqrt(areaPart)
          # bottom area covered by the particle during dt
          ABotSwiped = width * uMag * dt
          dm = entrMassCell * ABotSwiped
          areaEntrPart = entrMassCell / rhoEnt

  return dm, areaEntrPart


cpdef double computeDetMass(double dt, double detCell,
                                              double areaPart, double uMag):
  """ compute detrained mass

  Parameters
  ----------
  dt: float
    time step
  detCell : float
      coefficient for detrainment
  areaPart : float
      particle area
  uMag : float
      particle speed (velocity magnitude)

  Returns
  -------
  dm : float
      detrained mass
  """

  cdef double dmDet = 0
  
  # compute detrained mass
  if detCell > 0 and uMag > 0:
      dmDet = - detCell / uMag * dt * areaPart
  else:
      dmDet = 0

  if dmDet > 0 or np.isnan(dmDet):
      log.error('uMag, dt or areaPart is 0')

  return dmDet


cpdef double computeResForce(double areaPart, double rho, double cResCell,
                             double uMag, int explicitFriction, int resistanceType):
  """ compute force component due to resistance

  Parameters
  ----------
  areaPart : float
      particle area
  rho : float
      snow density
  cResCell : float
      resisance coefficient of cell
  uMag : float
      particle speed (velocity magnitude)
  explicitFriction: int
    if 1 add resistance with an explicit method. Implicit otherwise
  resistanceType: int
    identifier for resistance model to be used

  Returns
  -------
  cResPart : float
      resistance component for particle
  """

  cdef double cRecResPart = 0.0
  # explicit formulation (explicitFriction == 1)
  if explicitFriction == 1:
    if resistanceType == 1:
      # cResH
      cRecResPart = - rho * areaPart * cResCell * uMag * uMag
  elif explicitFriction == 0:
    if resistanceType == 1:
      # cResH
      cRecResPart = - rho * areaPart * cResCell * uMag
  return cRecResPart


cdef (double, double, double) addArtificialViscosity(double m, double h, double dt, double rho,
                                                     double ux, double uy, double uz, double subgridMixingFactor,
                                                     int Lx0, int Ly0, double w0, double w1, double w2, double w3,
                                                     double[:, :] VX, double[:, :] VY, double[:, :] VZ,
                                                     double nx, double ny, double nz):
  """ add artificial viscosity

  Add the artificial viscosity in an implicit way and this before adding the other forces.

  Parameters
  ----------
  m : float
      mass of the particle
  h : float
      flow thickness of the particle
  dt : float
      time step
  rho : float
      density
  subgridMixingFactor : float
      viscosity coefficient
  Lx0: int
      colomn of the nearest lower left cell
  Ly0: int
      row of the nearest lower left cell
  w: float[4]
      corresponding weights
      location in the y location of desiered interpolation
  VX: 2D numpy array
      x component of the velocity vector field at the grid nodes
  VY: 2D numpy array
      y component of the velocity vector field at the grid nodes
  VZ: 2D numpy array
      z component of the velocity vector field at the grid nodes
  nx: float
      x component of the interpolated vector field at particle position
  ny: float
      y component of the interpolated vector field at particle position
  nz: float
      z component of the interpolated vector field at particle position

  Returns
  -------
  ux: float
    x component of the velocity uptated with the viscous force
  uy: float
    y component of the uptated with the viscous force
  uz: float
    z component of the uptated with the viscous force
  """
  cdef double vMeanx, vMeany, vMeanz, vMeanNorm, dvX, dvY, dvZ
  vMeanx, vMeany, vMeanz = DFAtlsC.getVector(Lx0, Ly0, w0, w1, w2, w3, VX, VY, VZ)
  # compute normal component of the velocity
  vMeanNorm = DFAtlsC.scalProd(vMeanx, vMeany, vMeanz, nx, ny, nz)
  # remove normal component (make sure vMean is in the tangent plane)
  vMeanx = vMeanx - vMeanNorm * nx
  vMeany = vMeany - vMeanNorm * ny
  vMeanz = vMeanz - vMeanNorm * nz
  # compute particle to field velocity difference
  dvX = vMeanx - ux
  dvY = vMeany - uy
  dvZ = vMeanz - uz
  dvMag = DFAtlsC.norm(dvX, dvY, dvZ)
  Alat = 2.0 * math.sqrt((m * h) / rho)
  fDrag = (subgridMixingFactor * 0.5 * rho * dvMag * Alat * dt) / m

  # update velocity with artificial viscosity - implicit method
  ux = ux + fDrag * vMeanx
  uy = uy + fDrag * vMeany
  uz = uz + fDrag * vMeanz
  ux = ux / (1.0 + fDrag)
  uy = uy / (1.0 + fDrag)
  uz = uz / (1.0 + fDrag)
  return ux, uy, uz


def updatePositionC(cfg, particles, dem, force, fields, int typeStop=0):
  """ update particle position using euler forward scheme

  Cython implementation

  Parameters
  ----------
  cfg: configparser
      configuration for DFA simulation
  particles : dict
      particles dictionary at t
  dem : dict
      dictionary with dem information
  force : dict
      force dictionary
  fields : dict
      fields dictionary with flow thickness (needed if there is a dam)
  typeStop: int
    0 if standard stopping criterion, if 1 stopping criterion based on SPHforce - used for iniStep
  Returns
  -------
  particles : dict
      particles dictionary at t + dt
  """

  # read input parameters
  cdef double stopCrit = cfg.getfloat('stopCrit')
  cdef double stopCritIni = cfg.getfloat('stopCritIni')
  cdef double stopCritIniSmall = cfg.getfloat('stopCritIniSmall')
  cdef double uFlowingThreshold = cfg.getfloat('uFlowingThreshold')
  cdef double dt = particles['dt']
  log.debug('dt used now is %f' % dt)
  cdef double gravAcc = cfg.getfloat('gravAcc')
  cdef double velMagMin = cfg.getfloat('velMagMin')
  cdef double rho = cfg.getfloat('rho')
  cdef int interpOption = cfg.getint('interpOption')
  cdef int explicitFriction = cfg.getint('explicitFriction')
  cdef int reprojMethod = cfg.getint('reprojMethodPosition')
  cdef int reprojectionIterations = cfg.getint('reprojectionIterations')
  cdef double thresholdProjection = cfg.getfloat('thresholdProjection')
  cdef double centeredPosition = cfg.getfloat('centeredPosition')
  cdef int snowSlide = cfg.getint('snowSlide')
  cdef int adaptSfcStopped = cfg.getint('adaptSfcStopped')
  cdef int adaptSfcDetrainment = cfg.getint('adaptSfcDetrainment')
  cdef int adaptSfcEntrainment = cfg.getint('adaptSfcEntrainment')
  cdef int dissDam = cfg.getint('dissDam')
  cdef double csz = dem['header']['cellsize']
  cdef int nrows = dem['header']['nrows']
  cdef int ncols = dem['header']['ncols']
  cdef int nPart = particles['nPart']
  cdef double[:, :] nxArray = dem['Nx']
  cdef double[:, :] nyArray = dem['Ny']
  cdef double[:, :] nzArray = dem['Nz']
  cdef double[:, :] ZDEM = dem['rasterData']
  cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] outOfDEM
  outOfDEM = np.array(dem['outOfDEM'], dtype=bool)
  # read particles and fields
  cdef double[:] mass = particles['m']
  cdef double[:] idFixed = particles['idFixed']
  cdef double[:] sArray = particles['trajectoryLengthXY']
  cdef double[:] sCorArray = particles['trajectoryLengthXYCor']
  cdef double[:] lArray = particles['trajectoryLengthXYZ']
  cdef double[:] xArray = particles['x']
  cdef double[:] yArray = particles['y']
  cdef double[:] zArray = particles['z']
  cdef double[:] uxArray = particles['ux']
  cdef double[:] uyArray = particles['uy']
  cdef double[:] uzArray = particles['uz']
  cdef double[:] velocityMagArray = particles['velocityMag']
  cdef double[:] uAccArray = particles['uAcc']
  cdef double[:] totalEnthalpyArray = particles['totalEnthalpy']
  cdef double TotkinEne = particles['kineticEne']
  cdef double TotpotEne = particles['potentialEne']
  cdef double peakKinEne = particles['peakKinEne']
  cdef double peakForceSPH = particles['peakForceSPH']
  cdef double totKForceSPH = particles['forceSPHIni']
  cdef long long[:] ID = particles['ID']
  # Stopped particles
  cdef double[:] xStoppedArray = np.empty(0, dtype=np.float64)
  cdef double[:] yStoppedArray = np.empty(0, dtype=np.float64)
  cdef double[:] mStoppedArray = np.empty(0, dtype=np.float64)
  cdef double[:] idStoppedArray = np.empty(0, dtype=np.float64)
  cdef double[:] uMagStoppedArray = np.empty(0, dtype=np.float64)
  cdef long[:] indRemoveParticle
  # read fields
  cdef double[:] forceX = force['forceX']
  cdef double[:] forceY = force['forceY']
  cdef double[:] forceZ = force['forceZ']
  cdef double[:] forceFrict = force['forceFrict']
  cdef double[:] forceSPHX = force['forceSPHX']
  cdef double[:] forceSPHY = force['forceSPHY']
  cdef double[:] forceSPHZ = force['forceSPHZ']
  cdef double[:] dM = force['dM']
  cdef double[:] dMDet = force['dMDet']
  # read dam
  dam = dem['damLine']
  cdef int flagDam = dam['dam']
  cdef int restitutionCoefficient = dam['restitutionCoefficient']
  cdef int nIterDam = dam['nIterDam']
  cdef int nDamPoints = dam['nPoints']
  cdef long long[:] cellsCrossed = dam['cellsCrossed']
  cdef double[:] xFootArray = dam['x']
  cdef double[:] yFootArray = dam['y']
  cdef double[:] zFootArray = dam['z']
  cdef double[:] xCrownArray = dam['xCrown']
  cdef double[:] yCrownArray = dam['yCrown']
  cdef double[:] zCrownArray = dam['zCrown']
  cdef double[:] xTangentArray = dam['xTangent']
  cdef double[:] yTangentArray = dam['yTangent']
  cdef double[:] zTangentArray = dam['zTangent']
  # read fields
  cdef double[:, :] FD = fields['FT']
  # initialize outputs
  cdef double TotkinEneNew = 0
  cdef double TotpotEneNew = 0
  cdef double totForceSPHNew = 0
  cdef double[:] mNewArray = np.zeros(nPart, dtype=np.float64)
  cdef double[:] xNewArray = np.zeros(nPart, dtype=np.float64)
  cdef double[:] yNewArray = np.zeros(nPart, dtype=np.float64)
  cdef double[:] zNewArray = np.zeros(nPart, dtype=np.float64)
  cdef double[:] sNewArray = np.zeros(nPart, dtype=np.float64)
  cdef double[:] sCorNewArray = np.zeros(nPart, dtype=np.float64)
  cdef double[:] lNewArray = np.zeros(nPart, dtype=np.float64)
  cdef double[:] uxArrayNew = np.zeros(nPart, dtype=np.float64)
  cdef double[:] uyArrayNew = np.zeros(nPart, dtype=np.float64)
  cdef double[:] uzArrayNew = np.zeros(nPart, dtype=np.float64)
  cdef double[:] velocityMagArrayNew = np.zeros(nPart, dtype=np.float64)
  cdef int[:] keepParticle = np.ones(nPart, dtype=np.int32)
  cdef int[:] notStopParticle = np.ones(nPart, dtype=np.int32)
  # declare intermediate step variables
  cdef double m, h, x, y, z, sCor, s, l, ux, uy, uz, nx, ny, nz, dtStop, idfixed
  cdef double mNew, xNew, yNew, zNew, uxNew, uyNew, uzNew, txWall, tyWall, tzWall, totalEnthalpy, totalEnthalpyNew
  cdef double sCorNew, sNew, lNew, ds, dl, uN, uMag, uMagNew, fNx, fNy, fNz, dv, uMagt0, uMagt1
  cdef double ForceDriveX, ForceDriveY, ForceDriveZ
  cdef double massEntrained = 0, massDetrained = 0, massStopped = 0, massFlowing = 0, dissEm = 0
  cdef int k, inter
  cdef int nRemove = 0
  cdef int nStop = 0
  # variables for interpolation
  cdef int Lx0, Ly0, LxNew0, LyNew0, iCell, iCellNew
  cdef double w[4]
  cdef double wNew[4]

  # loop on particles
  for k in range(nPart):
    m = mass[k]
    x = xArray[k]
    y = yArray[k]
    z = zArray[k]
    ux = uxArray[k]
    uy = uyArray[k]
    uz = uzArray[k]
    s = sArray[k]
    sCor = sCorArray[k]
    l = lArray[k]
    idfixed = idFixed[k]

    # Force magnitude (without friction)
    ForceDriveX = forceX[k] + forceSPHX[k]
    ForceDriveY = forceY[k] + forceSPHY[k]
    ForceDriveZ = forceZ[k] + forceSPHZ[k]

    # velocity magnitude
    uMag = DFAtlsC.norm(ux, uy, uz)
    uMagt0 = DFAtlsC.norm(ux, uy, uz)

    # check if particle's mass is zero then remove particle
    if m == 0:
      keepParticle[k] = 0
      nRemove = nRemove + 1
      continue  # this particle will be removed, skip what is below and directly go to the next particle

    # procede to time integration
    # operator splitting
    # estimate new velocity due to driving force
    uxNew = ux + ForceDriveX * dt / m
    uyNew = uy + ForceDriveY * dt / m
    uzNew = uz + ForceDriveZ * dt / m

    # take friction force into account
    if typeStop != 1:
      uxNew, uyNew, uzNew, dtStop = account4FrictionForce(uxNew, uyNew, uzNew, m, dt, forceFrict[k], uMag, explicitFriction)
    else:
      dtStop = dt

    uMagNew = DFAtlsC.norm(uxNew, uyNew, uzNew)

    # update mass (already done un computeForceC)
    mNew = m
    massEntrained = massEntrained + dM[k]
    massDetrained = massDetrained + dMDet[k]

    # Stop particles with velocity = 0
    if adaptSfcStopped == 1 and uMagNew == 0:
        xStoppedArray = np.append(xStoppedArray, xArray[k])
        yStoppedArray = np.append(yStoppedArray, yArray[k])
        mStoppedArray = np.append(mStoppedArray, mass[k])
        idStoppedArray = np.append(idStoppedArray, ID[k])
        uMagStoppedArray = np.append(uMagStoppedArray, uMagNew)
        massStopped = massStopped + mass[k]
        notStopParticle[k] = 0  # particle is deleted
        nStop = nStop + 1
        continue

    # update position
    if centeredPosition:
      xNew = x + dtStop * 0.5 * (ux + uxNew)
      yNew = y + dtStop * 0.5 * (uy + uyNew)
      zNew = z + dtStop * 0.5 * (uz + uzNew)
    else:
      xNew = x + dtStop * uxNew
      yNew = y + dtStop * uyNew
      zNew = z + dtStop * uzNew
    # make sure particle is on the mesh (normal reprojection!!)

    if reprojMethod == 0:
      # Project vertically on the dem
      iCellNew = DFAtlsC.getCells(xNew, yNew, ncols, nrows, csz)
      if iCellNew >= 0 and outOfDEM[iCellNew] == 0:
        Lx0, Ly0, iCellNew, wNew[0], wNew[1], wNew[2], wNew[3] = DFAtlsC.getCellAndWeights(xNew, yNew, ncols, nrows, csz, interpOption)
        zNew = DFAtlsC.getScalar(Lx0, Ly0, wNew[0], wNew[1], wNew[2], wNew[3], ZDEM)

    elif reprojMethod == 1:
      # project trying to keep the travelled distance constant
      xNew, yNew, zNew, iCellNew, LxNew0, LyNew0, wNew[0], wNew[1], wNew[2], wNew[3] = DFAtlsC.distConservProjectionIteratrive(
        x, y, z, ZDEM, nxArray, nyArray, nzArray, xNew, yNew, zNew, csz, ncols, nrows, interpOption,
        reprojectionIterations, thresholdProjection)
    elif reprojMethod == 2:
      # project using samos method
      xNew, yNew, iCellNew, LxNew0, LyNew0, wNew[0], wNew[1], wNew[2], wNew[3] = DFAtlsC.samosProjectionIteratrive(
        xNew, yNew, zNew, ZDEM, nxArray, nyArray, nzArray, csz, ncols, nrows, interpOption, reprojectionIterations)
      if iCellNew >= 0:
        zNew = DFAtlsC.getScalar(LxNew0, LyNew0, wNew[0], wNew[1], wNew[2], wNew[3], ZDEM)

    if iCellNew < 0:
      # if the particle is not on the DEM, memorize it and remove it at the next update
      keepParticle[k] = 0
      LxNew0 = 0
      LyNew0 = 0
      wNew = [0, 0, 0, 0]
      nRemove = nRemove + 1
      continue  # this particle will be removed, skip the what is bellow and directly go to the next particle
    elif outOfDEM[iCellNew]:
      # if the particle is on the DEM but in a noData area,
      # memorize it and remove it at the next update
      keepParticle[k] = 0
      nRemove = nRemove + 1
      continue  # this particle will be removed, skipp to the next particle

    # get cell and weights at old position
    Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = DFAtlsC.getCellAndWeights(x, y, ncols, nrows, csz, interpOption)
    # get normal at the old particle location
    nx, ny, nz = DFAtlsC.getVector(Lx0, Ly0, w[0], w[1], w[2], w[3], nxArray, nyArray, nzArray)
    # get normal at the new particle location
    nxNew, nyNew, nzNew = DFAtlsC.getVector(LxNew0, LyNew0, wNew[0], wNew[1], wNew[2], wNew[3], nxArray, nyArray, nzArray)
    # get average normal between old and new position
    nx, ny, nz = DFAtlsC.normalize(nx+nxNew, ny+nyNew, nz+nzNew)
    # normalize new normal
    nxNew, nyNew, nzNew = DFAtlsC.normalize(nxNew, nyNew, nzNew)
    # compute the distance traveled by the particle
    dl = DFAtlsC.norm((xNew-x), (yNew-y), (zNew-z))
    lNew = l + dl
    # compute the horizontal distance traveled by the particle
    ds = DFAtlsC.norm((xNew-x), (yNew-y), 0)
    sNew = s + ds
    # compute the horizontal distance traveled by the particle (corrected with
    # the angle difference between the slope and the normal)
    sCorNew = sCor + nz*dl

    # reproject velocity
    uxNew, uyNew, uzNew, uMag = DFAtlsC.reprojectVelocity(uxNew, uyNew, uzNew, nxNew, nyNew, nzNew, velMagMin, 1)

    #  ############## Start add Dam interaction ##############################################
    if flagDam and cellsCrossed[iCell] == 1:
      # if there is an interaction with the dam, update position and velocity
      inter, xNew, yNew, zNew, uxNew, uyNew, uzNew, txWall, tyWall, tzWall, dissEm = damCom1DFA.getWallInteraction(x, y, z,
        xNew, yNew, zNew, uxNew, uyNew, uzNew, nDamPoints, xFootArray, yFootArray, zFootArray,
        xCrownArray, yCrownArray, zCrownArray, xTangentArray, yTangentArray, zTangentArray,
        ncols, nrows, csz, interpOption, restitutionCoefficient, nIterDam, nxArray, nyArray, nzArray, ZDEM, FD)
      # if there was an interaction with the dam, reproject and take dEM into account
      if inter == 1:
        LxNew0, LyNew0, iCellNew, wNew[0], wNew[1], wNew[2], wNew[3] = DFAtlsC.getCellAndWeights(xNew, yNew, ncols, nrows, csz, interpOption)
        if iCellNew >= 0:
          zNew = DFAtlsC.getScalar(LxNew0, LyNew0, wNew[0], wNew[1], wNew[2], wNew[3], ZDEM)
        # do we need to remove the particle?
        if iCellNew < 0:
          # if the particle is not on the DEM, memorize it and remove it at the next update
          keepParticle[k] = 0
          LxNew0 = 0
          LyNew0 = 0
          wNew = [0, 0, 0, 0]
          nRemove = nRemove + 1
          continue  # this particle will be removed, skipp to the next particle
        elif outOfDEM[iCellNew]:
          # if the particle is on the DEM but in a noData area,
          # memorize it and remove it at the next update
          keepParticle[k] = 0
          nRemove = nRemove + 1
          continue  # this particle will be removed, skipp to the next particle
        # reduce velocity normal to footline for energy loss due to flowing over dam
        # dEm>0 means the snow thickness exceeds the dam height???
        # reduce velocity normal to footline for energy loss due to flowing over dam
        # (but there is maybe nothing flowing over...)
        if dissDam == 1 and dissEm < 0.0:
          # First multiply dissEm by the gravAcc magnitude (because this was not done in the dam function)
          dissEm = gravAcc*dissEm
          fNx, fNy, fNz = DFAtlsC.crossProd(nxNew, nyNew, nzNew, txWall, tyWall, tzWall)
          fNx, fNy, fNz = DFAtlsC.normalize(fNx, fNy, fNz)
          dv = math.sqrt(-2.0 * dissEm)
          uN = -DFAtlsC.scalProd(uxNew, uyNew, uzNew, fNx, fNy, fNz)
          if uN < dv:
            dv = uN
          if dv > 0.0:
            uxNew = uxNew + dv * fNx
            uyNew = uyNew + dv * fNy
            uzNew = uzNew + dv * fNz
            # reproject velocity
            uxNew, uyNew, uzNew, uMag = DFAtlsC.reprojectVelocity(uxNew, uyNew, uzNew, nxNew, nyNew, nzNew, velMagMin, 1)
    #  ############## End add Dam interaction ##############################################

    # prepare for stopping criterion
    if uMag > uFlowingThreshold:
      # if velocity is bigger then threshold add to flowing mass
      massFlowing = massFlowing + mNew

    TotkinEneNew = TotkinEneNew + 0.5 * m * uMag * uMag
    TotpotEneNew = TotpotEneNew + mNew * gravAcc * zNew
    totForceSPHNew = totForceSPHNew + mNew * DFAtlsC.norm(forceSPHX[k], forceSPHY[k], forceSPHZ[k])

    if idfixed == 1:
      # idfixed = 1 if particles belong to 'fixed' boundary particles - so zero velocity and fixed position
      xNewArray[k] = x
      yNewArray[k] = y
      zNewArray[k] = z
      uxArrayNew[k] = ux
      uyArrayNew[k] = uy
      uzArrayNew[k] = uz
      lNewArray[k] = l
      sNewArray[k] = s
      sCorNewArray[k] = sCor
      mNewArray[k] = m
    else:
      # idfixed = 0 particles belong to the actual releae area
      xNewArray[k] = xNew
      yNewArray[k] = yNew
      zNewArray[k] = zNew
      uxArrayNew[k] = uxNew
      uyArrayNew[k] = uyNew
      uzArrayNew[k] = uzNew
      lNewArray[k] = lNew
      sNewArray[k] = sNew
      sCorNewArray[k] = sCorNew
      mNewArray[k] = mNew

      # compute acceleration
      uMagt1 = DFAtlsC.norm(uxNew, uyNew, uzNew)
      uAcc = (uMagt1 - uMagt0) / dtStop
      uAccArray[k] = uAcc
      velocityMagArrayNew[k] = uMagt1

  particles['ux'] = np.asarray(uxArrayNew)
  particles['uy'] = np.asarray(uyArrayNew)
  particles['uz'] = np.asarray(uzArrayNew)
  particles['velocityMag'] = np.asarray(velocityMagArrayNew)
  particles['uAcc'] = np.asarray(uAccArray)
  particles['trajectoryLengthXYZ'] = np.asarray(lNewArray)
  particles['trajectoryLengthXY'] = np.asarray(sNewArray)
  particles['trajectoryLengthXYCor'] = np.asarray(sCorNewArray)
  particles['m'] = np.asarray(mNewArray)
  particles['mTot'] = np.sum(particles['m'])
  particles['x'] = np.asarray(xNewArray)
  particles['y'] = np.asarray(yNewArray)
  particles['z'] = np.asarray(zNewArray)
  particles['massEntrained'] = massEntrained
  particles['massDetrained'] = massDetrained
  log.debug('Entrained DFA mass: %s kg', np.asarray(massEntrained))
  log.debug('Detrained DFA mass: %s kg', np.asarray(massDetrained))
  particles['kineticEne'] = TotkinEneNew
  particles['potentialEne'] = TotpotEneNew

  particles['stoppedParticles']['x'] = np.asarray(xStoppedArray)
  particles['stoppedParticles']['y'] = np.asarray(yStoppedArray)
  particles['stoppedParticles']['m'] = np.asarray(mStoppedArray)
  particles['stoppedParticles']['ID'] = np.asarray(idStoppedArray)

  if typeStop == 1:
    # typeStop = 1 for initialisation step where particles are redistributed to reduce SPH force
    # here stop criterion based on SPHForce
    value = totForceSPHNew
    peakValue = particles['peakForceSPH']
    oldValue = particles['forceSPHIni']
    particles['forceSPHIni'] = totForceSPHNew
    if oldValue == 0.0:
      stop = False
    elif value < 1.:
      stop = oldValue/value < stopCritIniSmall
    else:
      stop = value <= stopCritIni*peakValue
      log.debug('SPHFORCE value %f and stop value %f' % (totForceSPHNew, stopCritIni*peakValue))
    if peakForceSPH < totForceSPHNew:
      particles['peakForceSPH'] = totForceSPHNew
  else:
    # avalanche computation stop criterion based on kinetic energy or massFlowing
    if peakKinEne < TotkinEneNew:
      particles['peakKinEne'] = TotkinEneNew
    if particles['peakMassFlowing'] < massFlowing:
      particles['peakMassFlowing'] = massFlowing

    if cfg['stopCritType'] == 'massFlowing':
      value = massFlowing
      peakValue = particles['peakMassFlowing']
      stop = (value <= stopCrit*peakValue) and peakValue > 0
    elif cfg['stopCritType'] == 'kinEnergy':
      value = TotkinEneNew
      peakValue = particles['peakKinEne']
      stop = value <= stopCrit*peakValue
    else:
      message = 'stopCritType %s is not defined' % cfg['stopCritType']
      log.error(message)
      raise AssertionError(message)

  if stop:
    particles['iterate'] = False
    particles['stoppedParticles'] = particles
    massStopped = massStopped + massFlowing

    if typeStop == 1:
      log.debug('stopping initial particle distribution')
    else:
      log.debug('stopping because of %s stopCriterion.' % (cfg['stopCritType']))

  particles['massStopped'] = - massStopped

  # remove particles that are not located on the mesh any more OR have zero mass
  if nRemove > 0:
    mask = np.array(np.asarray(keepParticle), dtype=bool)
    particles = particleTools.removePart(particles, mask, nRemove, 'because they exited the domain', snowSlide=snowSlide)

  # remove particles that have velocity = 0 (only when surface is adapted due to stopping)
  if nStop > 0 and adaptSfcStopped == 1:
    indRemoveParticle = np.array([], dtype=np.int64)
    for k in range(len(keepParticle)):
      # consider particles that are removed because they exit the domain
      if keepParticle[k] == 0:
          indRemoveParticle = np.append(indRemoveParticle, int(k))
          if notStopParticle[k] == 0:
            nStop = nStop - 1
    notStopParticle = np.delete(notStopParticle, indRemoveParticle)
    mask = np.array(np.asarray(notStopParticle), dtype=bool)
    particles = particleTools.removePart(particles, mask, nStop, 'because their mass or velocity is zero', snowSlide=snowSlide)

  return particles


cpdef (double, double, double, double) account4FrictionForce(double ux, double uy,
                                                            double uz, double m,
                                                            double dt, double forceFrict,
                                                            double uMag, int explicitFriction):
  """ update velocity with friction force

  Parameters
  ----------
  ux: float
      x velocity
  uy: float
      y velocity
  uz: float
      z velocity
  m : float
      particle area
  dt : float
      time step
  forceFrict : float
      friction force (actually a force for the explicit method, force/vel for implicit)
  uMag : float
      particle speed (velocity magnitude)
  explicitFriction: int
    if 1 add resistance with an explicit method. Implicit otherwise

  Returns
  -------
  uxNew: float
      x velocity
  uyNew: float
      y velocity
  uzNew: float
      z velocity
  dtStop : float
      time step (smaller then dt if the particle stops during this time step)
  """
  cdef double xDir, yDir, zDir
  cdef double uxNew = 0.0
  cdef double uyNew = 0.0
  cdef double uzNew = 0.0
  cdef double uMagNew = 0.0
  cdef double dtStop = 0.0

  if explicitFriction == 1:
    # explicite method
    uMagNew = DFAtlsC.norm(ux, uy, uz)
    # will friction force stop the particle
    if uMagNew<dt*forceFrict/m:
      # stop the particle
      uxNew = 0
      uyNew = 0
      uzNew = 0
      # particle stops after
      if uMag<=0:
        dtStop = 0
      else:
        dtStop = m * uMagNew / (forceFrict)
    else:
      # add friction force in the opposite direction of the motion
      xDir, yDir, zDir = DFAtlsC.normalize(ux, uy, uz)
      uxNew = ux - xDir * forceFrict * dt / m
      uyNew = uy - yDir * forceFrict * dt / m
      uzNew = uz - zDir * forceFrict * dt / m
      dtStop = dt
  elif explicitFriction == 0:
    # implicite method
    uxNew = ux / (1.0 + dt * forceFrict / m)
    uyNew = uy / (1.0 + dt * forceFrict / m)
    uzNew = uz / (1.0 + dt * forceFrict / m)
    dtStop = dt

  return uxNew, uyNew, uzNew, dtStop


def updateFieldsC(cfg, particles, dem, fields):
  """ update fields and particles flow thickness

  Cython implementation

 Parameters
 ----------
 cfg: configparser
     configuration for DFA simulation
 particles : dict
     particles dictionary
 dem : dict
     dictionary with dem information
 fields : dict
     fields dictionary

 Returns
 -------
 particles : dict
     particles dictionary
 fields : dict
     fields dictionary
 """
  # read input parameters
  cdef double rho = cfg.getfloat('rho')
  cdef double rhoEnt = cfg.getfloat('rhoEnt')
  cdef int interpOption = cfg.getint('interpOption')
  header = dem['header']
  cdef int nrows = header['nrows']
  cdef int ncols = header['ncols']
  cdef double xllc = 0
  cdef double yllc = 0
  cdef double csz = header['cellsize']
  cdef int nPart = np.size(particles['x'])
  cdef double[:, :] areaRaster = dem['areaRaster']
  # read particles and fields
  cdef double[:] mass = particles['m']
  cdef double[:] massDet = particles['dmDet']
  cdef double[:] massEnt = particles['dmEnt']
  cdef double[:] xArray = particles['x']
  cdef double[:] yArray = particles['y']
  cdef double[:] uxArray = particles['ux']
  cdef double[:] uyArray = particles['uy']
  cdef double[:] uzArray = particles['uz']
  cdef double[:] trajectoryAngleArray = particles['trajectoryAngle']
  cdef double[:] mStoppedArray = particles['stoppedParticles']['m']
  cdef double[:] xStoppedArray = particles['stoppedParticles']['x']
  cdef double[:] yStoppedArray = particles['stoppedParticles']['y']

  cdef bint computeTA = fields['computeTA']
  cdef bint computeKE = fields['computeKE']
  cdef bint computeP = fields['computeP']
  cdef double[:, :] PFV = fields['pfv']
  cdef double[:, :] PP = fields['ppr']
  cdef double[:, :] PFT = fields['pft']
  cdef double[:, :] PTA = fields['pta']
  cdef double[:, :] PKE = fields['pke']
  cdef double[:, :] DMDet = fields['dmDet']
  # initialize outputs
  cdef double[:, :] MassBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] MassDetBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] MassStopBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] MassEntBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] VBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] PBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] FTBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] MomBilinearX = np.zeros((nrows, ncols))
  cdef double[:, :] MomBilinearY = np.zeros((nrows, ncols))
  cdef double[:, :] MomBilinearZ = np.zeros((nrows, ncols))
  cdef double[:, :] VXBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] VYBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] VZBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] kineticEnergy = np.zeros((nrows, ncols))
  cdef double[:, :] travelAngleField = np.zeros((nrows, ncols))
  cdef double[:, :] FTDetBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] FTStopBilinear = np.zeros((nrows, ncols))
  cdef double[:, :] FTEntBilinear = np.zeros((nrows, ncols))
  # declare intermediate step variables
  cdef double[:] hBB = np.zeros((nPart))
  cdef double m, dmDet, demEnt, h, x, y, z, s, ux, uy, uz, nx, ny, nz, hbb, hLim, areaPart, trajectoryAngle
  cdef double mStop, xStop, yStop
  cdef int k, i, l
  cdef int indx, indy
  cdef int ind1[4]
  cdef int ind2[4]
  ind1[:] = [0, 1, 0, 1]
  ind2[:] = [0, 0, 1, 1]
  # variables for interpolation
  cdef int Lx0, Ly0, iCell
  cdef double w[4]
  cdef double mwi, dmDetWi, dmEntWi

  for k in range(nPart):
    x = xArray[k]
    y = yArray[k]
    ux = uxArray[k]
    uy = uyArray[k]
    uz = uzArray[k]
    m = mass[k]
    dmDet = massDet[k]
    dmEnt = massEnt[k]
    # find coordinates in normalized ref (origin (0,0) and cellsize 1)
    # find coordinates of the 4 nearest cornes on the raster
    # prepare for bilinear interpolation
    Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = DFAtlsC.getCellAndWeights(x, y, ncols, nrows, csz, interpOption)
    # for the travel angle we simply do a nearest interpolation
    indx = <int>math.round(x / csz)
    indy = <int>math.round(y / csz)
    if computeTA:
      trajectoryAngle = trajectoryAngleArray[k]
      travelAngleField[indy, indx] = max(travelAngleField[indy, indx], trajectoryAngle)
    # add the component of the points value to the 4 neighbour grid points
    # TODO : check if giving the arrays [0 1 0 1].. is faster
    for i in range(4):
      indx = Lx0 + ind1[i]
      indy = Ly0 + ind2[i]
      mwi = m * w[i]
      dmDetWi = dmDet * w[i]
      dmEntWi = dmEnt * w[i]
      MassBilinear[indy, indx] = MassBilinear[indy, indx] + mwi
      MassDetBilinear[indy, indx] = MassDetBilinear[indy, indx] + dmDetWi
      MassEntBilinear[indy, indx] = MassEntBilinear[indy, indx] + dmEntWi 
      MomBilinearX[indy, indx] = MomBilinearX[indy, indx] + mwi * ux
      MomBilinearY[indy, indx] = MomBilinearY[indy, indx] + mwi * uy
      MomBilinearZ[indy, indx] = MomBilinearZ[indy, indx] + mwi * uz

  for l in range(len(mStoppedArray)):
    xStop = xStoppedArray[l]
    yStop = yStoppedArray[l]
    mStop = - mStoppedArray[l]  # Stopped mass is negativ for the flow

    Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = DFAtlsC.getCellAndWeights(xStop, yStop, ncols, nrows, csz, interpOption)
    # for the travel angle we simply do a nearest interpolation
    indx = <int>math.round(xStop / csz)
    indy = <int>math.round(yStop / csz)
  
    for i in range(4):
      indx = Lx0 + ind1[i]
      indy = Ly0 + ind2[i]
      mwi = mStop * w[i]
      MassStopBilinear[indy, indx] = MassStopBilinear[indy, indx] + mwi

  for i in range(ncols):
    for j in range(nrows):
      m = MassBilinear[j, i]

      DMDet[j, i] = DMDet[j, i] + MassDetBilinear[j, i]

      if m > 0:
        # TODO: here we devide by the area of the vertex, would it not make
        # more sense to devide by the area of the cell in the previous loop?
        FTBilinear[j, i] = m / (areaRaster[j, i] * rho)
        VXBilinear[j, i] = MomBilinearX[j, i]/m
        VYBilinear[j, i] = MomBilinearY[j, i]/m
        VZBilinear[j, i] = MomBilinearZ[j, i]/m
        VBilinear[j, i] = DFAtlsC.norm(VXBilinear[j, i], VYBilinear[j, i], VZBilinear[j, i])
        if VBilinear[j, i] > PFV[j, i]:
          PFV[j, i] = VBilinear[j, i]
        if FTBilinear[j, i] > PFT[j, i]:
          PFT[j, i] = FTBilinear[j, i]
        if computeP:
          PBilinear[j, i] = computePressure(VBilinear[j, i], rho)
          if PBilinear[j, i] > PP[j, i]:
            PP[j, i] = PBilinear[j, i]
        if computeTA:
          if travelAngleField[j, i] > PTA[j, i]:
            PTA[j, i] = travelAngleField[j, i]
        if computeKE:
          # in J/cell (this is not normalized yet and depends on the cell size used for the computation)
          kineticEnergy[j, i] = 0.5*m*VBilinear[j, i]*VBilinear[j, i]
          if kineticEnergy[j, i] > PKE[j, i]:
            PKE[j, i] = kineticEnergy[j, i]

      # thickness change due to detrainment, stopping and entrainment
      FTDetBilinear[j, i] = - MassDetBilinear[j, i] / (areaRaster[j, i] * rho)  # / m * FTBilinear[j, i]
      FTStopBilinear[j, i] = - MassStopBilinear[j, i] / (areaRaster[j, i] * rho)  # / m * FTBilinear[j, i] 
      FTEntBilinear[j, i] = - MassEntBilinear[j, i] / (areaRaster[j, i] * rhoEnt)

  fields['FM'] = np.asarray(MassBilinear)
  fields['FV'] = np.asarray(VBilinear)
  fields['Vx'] = np.asarray(VXBilinear)
  fields['Vy'] = np.asarray(VYBilinear)
  fields['Vz'] = np.asarray(VZBilinear)
  fields['FT'] = np.asarray(FTBilinear)
  fields['pfv'] = np.asarray(PFV)
  fields['pft'] = np.asarray(PFT)
  fields['dmDet'] = np.asarray(DMDet)
  fields['FTStop'] = np.asarray(FTStopBilinear)
  fields['FTDet'] = np.asarray(FTDetBilinear)
  fields['FTEnt'] = np.asarray(FTEntBilinear)

  if computeP:
    fields['ppr'] = np.asarray(PP)
    fields['P'] = np.asarray(PBilinear)
  if computeTA:
    fields['TA'] = np.asarray(travelAngleField)
    fields['pta'] = np.asarray(PTA)
  if computeKE:
    fields['pke'] = np.asarray(PKE)


  for k in range(nPart):
    x = xArray[k]
    y = yArray[k]
    Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = DFAtlsC.getCellAndWeights(x, y, ncols, nrows, csz, interpOption)
    hbb = DFAtlsC.getScalar(Lx0, Ly0, w[0], w[1], w[2], w[3], FTBilinear)
    hBB[k] = hbb

  particles['h'] = np.asarray(hBB)

  return particles, fields


cpdef double computePressure(double v, double rho):
  """Compute pressure using the p = rho*v² equation

  Parameters
  ----------
  v : float
      velocity magnitude
  rho : float
      snow density

  Returns
  -------
  p : float
      pressure computed using p = rho*v²
  """
  cdef double p
  p = rho * v * v
  return p


def computeTrajectoryAngleC(particles, zPartArray0):
  """Compute the travel angle associated to the particles

  Parameters
  ----------
  particles : dict
      particles dictionary at t
  zPartArray0 : dict
      z coordinate of particles at t=0s

  Returns
  -------
  particles : dict
      particles dictionary updated with the travel angle
  """
  cdef int[:] parentIDArray = particles['parentID'].astype('intc')
  cdef int nPart = particles['nPart']
  cdef double[:] zArray = particles['z']
  cdef double[:] sArray = particles['trajectoryLengthXY']
  cdef double[:] z0Array = zPartArray0
  cdef double[:] gammaArray = np.zeros(nPart)
  cdef int parentID, j
  cdef double tanGamma, gamma, s, z, z0
  # get particle location
  # first compute travel angle for each particle
  for k in range(nPart):
    # get parent Id in order to  get the first z position
    parentID = parentIDArray[k]
    # get z0
    z0 = z0Array[parentID]
    # compute tan of the travel angle
    z = zArray[k]
    s = sArray[k]
    if s > 0:
      tanGamma = (z0 - z) / s
    else:
      tanGamma = 0.0
    # get the travel angle
    gamma = math.atan(tanGamma) * 180.0 / math.pi
    gammaArray[k] = gamma
  particles['trajectoryAngle'] = np.asarray(gammaArray)
  return particles


def getNeighborsC(particles, dem):
    """ Locate particles on DEM and neighbour search grid

    Ĺocate each particle in a grid cell (both DEM and neighbour search grid) and build the
    indPartInCell and partInCell arrays for SPH neighbourSearch and
    InCell, IndX and IndY arrays location on the DEM grid.
    See issue #200 and documentation for details

    Parameters
    ----------
    particles : dict
    dem : dict
      dem dict with neighbour search grid header (information about neighbour search grid)

    Returns
    -------
    particles : dict
      updated particles dictionary with
      neighbours info indPartInCell and partInCell arrays for neighbour search and
      with InCell, IndX and IndY arrays location on the DEM grid
    """
    # get DEM grid information
    header = dem['header']
    cdef int nColsDEM = header['ncols']
    cdef int nRowsDEM = header['nrows']
    cdef float cszDEM = header['cellsize']
    # get neighbour search grid information
    headerNeighbourGrid = dem['headerNeighbourGrid']
    cdef int nColsNeighbourGrid = headerNeighbourGrid['ncols']
    cdef int nRowsNeighbourGrid = headerNeighbourGrid['nrows']
    cdef float cszNeighbourGrid = headerNeighbourGrid['cellsize']
    # get particle location
    cdef int nPart = particles['nPart']
    cdef int k
    cdef double[:] xArray = particles['x']
    cdef double[:] yArray = particles['y']

    # initialize outputs
    cdef int nCellsNeighbourGrid = nColsNeighbourGrid*nRowsNeighbourGrid
    cdef int[:] indPartInCell = np.zeros(nCellsNeighbourGrid + 1).astype('intc')
    cdef int[:] indPartInCell2 = np.zeros(nCellsNeighbourGrid + 1).astype('intc')
    cdef int[:] partInCell = np.zeros(nPart).astype('intc')
    cdef int[:] indXDEM = np.zeros(nPart).astype('intc')
    cdef int[:] indYDEM = np.zeros(nPart).astype('intc')
    cdef int[:] inCellDEM = np.zeros(nPart).astype('intc')
    # Count number of particles in each SPH grid cell
    cdef int indx, indy, ic
    for k in range(nPart):
      indx = <int>math.round(xArray[k] / cszNeighbourGrid)
      indy = <int>math.round(yArray[k] / cszNeighbourGrid)
      # get index of cell containing the particle
      ic = indx + nColsNeighbourGrid * indy
      indPartInCell[ic+1] = indPartInCell[ic+1] + 1
    for ic in range(nCellsNeighbourGrid):
      indPartInCell[ic+1] = indPartInCell[ic] + indPartInCell[ic+1]
      indPartInCell2[ic+1] = indPartInCell[ic+1]

    # make the list of which particles are in which cell
    for k in range(nPart):
        indx = <int>math.round(xArray[k] / cszNeighbourGrid)
        indy = <int>math.round(yArray[k] / cszNeighbourGrid)
        ic = indx + nColsNeighbourGrid * indy
        partInCell[indPartInCell2[ic+1]-1] = k
        indPartInCell2[ic+1] = indPartInCell2[ic+1] - 1
        indXDEM[k] = <int>math.round(xArray[k] / cszDEM)
        indYDEM[k] = <int>math.round(yArray[k] / cszDEM)
        # get index of cell containing the particle
        inCellDEM[k] = indXDEM[k] + nColsDEM * indYDEM[k]

    particles['inCellDEM'] = np.asarray(inCellDEM)
    particles['indXDEM'] = np.asarray(indXDEM)
    particles['indYDEM'] = np.asarray(indYDEM)
    particles['indPartInCell'] = np.asarray(indPartInCell)
    particles['partInCell'] = np.asarray(partInCell)

    return particles


def computeCohesionForceC(cfg, particles, force):
  """ compute elastic cohesion forces acting on the particles
  this is computed when the snow slide option is activated (snowSlide = 1
  )
  Parameters
  ----------
  cfg: configparser
      configuration for DFA simulation
  particles : dict
      particles dictionary at time t
  force : dict
      force dictionary
  Returns
  -------
  particles : dict
    particles dictionary at t (updated bondDist due to breaking)
  force : dict
    force dictionary with bonding elastic force
  """
  # read input parameters
  cdef double dt = particles['dt']
  cdef double cohesiveSurfaceTension = cfg.getfloat('cohesiveSurfaceTension')
  cdef double cohesionMaxStrain = cfg.getfloat('cohesionMaxStrain')
  cdef double minDistCohesion = cfg.getfloat('minDistCohesion')
  cdef int nPart = particles['nPart']
  # read particles and fields
  cdef double[:] mass = particles['m']
  cdef double[:] hArray = particles['h']
  cdef double[:] xArray = particles['x']
  cdef double[:] yArray = particles['y']
  cdef double[:] zArray = particles['z']
  cdef double[:] uxArray = particles['ux']
  cdef double[:] uyArray = particles['uy']
  cdef double[:] uzArray = particles['uz']
  cdef int[:] bondStart = particles['bondStart'].astype('intc')
  cdef double[:] bondDist = particles['bondDist']
  cdef int[:] bondPart = particles['bondPart'].astype('intc')
  cdef double[:] forceSPHX = force['forceSPHX']
  cdef double[:] forceSPHY = force['forceSPHY']
  cdef double[:] forceSPHZ = force['forceSPHZ']
  # initialize outputs
  cdef int k, l, ib
  cdef double dist0, dist, vx, vy, vz, xk, yk, zk, xl, yl, zl
  cdef double hk, hl, ux, uy, uz, dx, dy, dz
  cdef double epsilon, mkl, Akl, contact, forceCohesion
  # loop on particles
  for k in range(nPart):
    hk = hArray[k]
    xk = xArray[k]
    yk = yArray[k]
    zk = zArray[k]
    ux = uxArray[k]
    uy = uyArray[k]
    uz = uzArray[k]
    # loop on all bonded particles
    for ib in range(bondStart[k], bondStart[k + 1]):
      # get initial distance between particles
       dist0 = bondDist[ib]
       # if bond already broken, just skip it
       if (dist0 < 0.0):
         continue
       # get the bonded particle
       l = bondPart[ib]
       xl = xArray[l]
       yl = yArray[l]
       zl = zArray[l]
       vx = uxArray[l]
       vy = uyArray[l]
       vz = uzArray[l]
       dx = xl - xk + dt * (vx - ux)
       dy = yl - yk + dt * (vy - uy)
       dz = zl - zk + dt * (vz - uz)
       dist = DFAtlsC.norm(dx, dy, dz)
       # only if the distance is bigger than a threshold (avoid dividing by 0)
       if (dist > minDistCohesion):
         # strain
         epsilon = dist / dist0 - 1.0
         # allow breaking in both compression and extension
         if (abs(epsilon) > cohesionMaxStrain):
           # break bond
           bondDist[ib] = -1
         # if not broken add elastic force
         else:
           hl = hArray[l]
           # compute average depth and mass
           hkl = 0.5 * (hk + hl)
           # sqrt(3) from 6-neighbors-assumption (Iwould add a 2 here... 2/sqrt(3))
           Akl = hkl * dist / math.sqrt(3)
           # cohesiveSurfaceTension used as elasticity modulus (Pa)
           contact = Akl * cohesiveSurfaceTension
           # dx / dist is  the unit direction vector towards bonded particle l
           # cohesion force (including the normalizing factor 1/dist)
           forcecohesion = (1 / dist) * epsilon * contact
           forceSPHX[k] = forceSPHX[k] + forcecohesion * dx
           forceSPHY[k] = forceSPHY[k] + forcecohesion * dy
           forceSPHZ[k] = forceSPHZ[k] + forcecohesion * dz
  force['forceSPHX'] = np.asarray(forceSPHX)
  force['forceSPHY'] = np.asarray(forceSPHY)
  force['forceSPHZ'] = np.asarray(forceSPHZ)
  particles['bondDist'] = np.asarray(bondDist)
  return force, particles


def plotBondC(particles):
  """ update edges for plot (bonds that still exist)
  Cython implementation
  Parameters
  ----------
  particles : dict
      particles dictionary
  Returns
  -------
  edges : 2D numpy array
      bonds that still exist
  """
  # read input parameters
  cdef int nPart = particles['nPart']
  # read particles and fields
  cdef int[:] bondStart = particles['bondStart'].astype('intc')
  cdef double[:] bondDist = particles['bondDist']
  cdef int[:] bondPart = particles['bondPart'].astype('intc')
  cdef int nEdges = bondPart.shape[0]
  cdef int[:, :] edges = np.zeros((nEdges, 2), dtype=np.int32)
  # initialize outputs
  cdef int k, l, ib, count = 0
  cdef double dist0
  # loop on particles
  for k in range(nPart):
    # loop on all bonded particles
    for ib in range(bondStart[k], bondStart[k + 1]):
      # get initial distance between particles
       dist0 = bondDist[ib]
       # if bond already broken, just skip it
       if (dist0 > 0.0):
         l = bondPart[ib]
         edges[count, 0] = k
         edges[count, 1] = l
         count = count + 1
  return edges


def initializeBondsC(particles, triangles):
  """ Initialize bonds if snowSlide is activated
  Parameters
  ----------
  particles : dict
      particles dictionary at t start
  triangles : matplotlib object
      result from tri.Triangulation(x, y)
  Returns
  -------
  particles : dict
    particles dictionary updated with the bonds
  """
  # get all inputs
  # particle information
  cdef double[:] xArray = particles['x']
  cdef double[:] yArray = particles['y']
  cdef double[:] zArray = particles['z']
  cdef int[:, :] tri = triangles.triangles
  cdef int[:, :] edges = triangles.edges
  cdef int nPart = xArray.shape[0]
  cdef int nTri = tri.shape[0]
  cdef int nEdges = edges.shape[0]

  # initialize variables and outputs
  cdef int[:] bondStart = np.zeros(nPart + 1, dtype=np.int32)
  cdef int[:] bondStart2 = np.zeros(nPart + 1, dtype=np.int32)
  cdef int[:] bondPart = np.zeros(2 * nEdges, dtype=np.int32)
  cdef double[:] bondDist = np.zeros(2 * nEdges, dtype=np.float64)
  cdef double dx, dy, dz
  cdef int k, e, p1, p2
  cdef double distanceIni

  # count bonds for each particle
  for e in range(nEdges):
    p1 = edges[e, 0]
    p2 = edges[e, 1]
    bondStart[p1+1] = bondStart[p1+1] + 1
    bondStart[p2+1] = bondStart[p2+1] + 1
    # create cumulative sum
  for k in range(nPart):
    bondStart[k+1] = bondStart[k] + bondStart[k+1]
    bondStart2[k+1] = bondStart[k+1]

  # make the list of which particles are bonded with which particle
  for e in range(nEdges):
    p1 = edges[e, 0]
    p2 = edges[e, 1]
    dx = xArray[p1] - xArray[p2]
    dy = yArray[p1] - yArray[p2]
    dz = zArray[p1] - zArray[p2]
    distanceIni = DFAtlsC.norm(dx, dy, dz)
    bondPart[bondStart2[p1+1]-1] = p2
    bondDist[bondStart2[p1+1]-1] = distanceIni
    bondStart2[p1+1] = bondStart2[p1+1] - 1
    bondPart[bondStart2[p2+1]-1] = p1
    bondDist[bondStart2[p2+1]-1] = distanceIni
    bondStart2[p2+1] = bondStart2[p2+1] - 1

  particles['bondStart'] = np.asarray(bondStart)
  particles['bondPart'] = np.asarray(bondPart)
  particles['bondDist'] = np.asarray(bondDist)
  return particles

def countRemovedBonds(particles, mask, nRemove):
  """ count number of bonds to remove
  Parameters
  ----------
  particles : dict
    particles dictionary at t start
  mask : 1D numpy array
    particles to keep
  nRemove : int
    number of particles removed
  Returns
  -------
  nBondRemove : int
    number of bonds removed
  """
  # read input parameters
  cdef int nPart = particles['nPart']
  cdef int[:] bondStart = particles['bondStart'].astype('intc')
  # read particles and fields
  cdef int[:] keepParticle = mask.astype('intc')
  cdef int nBondRemove = 0
  cdef int k, ib

  # count bonds for each particle
  for k in range(nPart):
    if keepParticle[k] == 0:
      # loop on all bonded particles
      for ib in range(bondStart[k], bondStart[k + 1]):
        nBondRemove = nBondRemove + 2
  return nBondRemove


def removedBonds(particles, mask, nRemove, nBondRemove):
  """ Remove bonds of removed particles
  Parameters
  ----------
  particles : dict
      particles dictionary at t start
      mask : 1D numpy array
          particles to keep
      nRemove : int
          number of particles removed
  Returns
  -------
  particles : dict
    particles dictionary according to removed particles
  """
  # read input parameters
  cdef int nPart = particles['nPart']
  # read particles and fields
  cdef int[:] bondStart = particles['bondStart'].astype('intc')
  cdef double[:] bondDist = particles['bondDist']
  cdef int[:] bondPart = particles['bondPart'].astype('intc')
  cdef int[:] keepParticle = mask.astype('intc')
  cdef int nEdges = bondPart.shape[0]
  cdef int nPartRemoved = nRemove
  cdef int[:] bondStartNew = np.zeros(nPart + 1, dtype=np.int32)
  cdef int[:] bondPartNew = np.zeros(nEdges - nBondRemove, dtype=np.int32)
  cdef double[:] bondDistNew = np.zeros(nEdges - nBondRemove, dtype=np.float64)
  cdef int countBond = 0
  cdef int countBondNew = 0
  cdef int k, ib, l, nBond

  # countBond bonds for each particle
  for k in range(nPart):
    # count the bonds of the particle
    nBond = 0
    if keepParticle[k] == 1:
      # loop on all bonded particles to k
      for ib in range(bondStart[k], bondStart[k + 1]):
         l = bondPart[ib]
         if keepParticle[l] == 1:
           # we keep the particle l so add it to the arrays
           bondPartNew[countBondNew] = l
           bondDistNew[countBondNew] = bondDist[ib]
           countBondNew = countBondNew + 1
           countBond = countBond + 1
           nBond = nBond + 1
         else:
           countBond = countBond + 1
      bondStartNew[k + 1] = bondStartNew[k] + nBond
    else:
      # we do not keep the particle
      bondStartNew[k + 1] = bondStartNew[k]
      # loop on all bonded particles
      for ib in range(bondStart[k], bondStart[k + 1]):
        # increment the counter without doing any thing else
         countBond = countBond + 1


  particles['bondStart'] = np.asarray(bondStartNew)
  particles['bondPart'] = np.asarray(bondPartNew)
  particles['bondDist'] = np.asarray(bondDistNew)
  return particles


def computeForceSPHC(cfg, particles, force, dem, int sphOption, gradient=0):
  """ Prepare data for C computation of lateral forces (SPH component)

  acting on the particles (SPH component)

  Parameters
  ----------
  cfg: configparser
      configuration for DFA simulation
  particles : dict
      particles dictionary at t
  force : dict
      force dictionary
  dem : dict
      dictionary with dem information
  sphOption: int
      which sphOption should be use
  Returns
  -------
  particles : dict
      particles dictionary at t
  force : dict
      force dictionary
  """
  # get SPH grid information
  headerNeighbourGrid = dem['headerNeighbourGrid']
  # get DEM header and normal field
  headerNormalGrid = dem['header']
  nxArray = dem['Nx']
  nyArray = dem['Ny']
  nzArray = dem['Nz']

  forceSPHX, forceSPHY, forceSPHZ = computeGradC(cfg, particles, headerNeighbourGrid, headerNormalGrid, nxArray, nyArray, nzArray, gradient, sphOption)
  forceSPHX = np.asarray(forceSPHX)
  forceSPHY = np.asarray(forceSPHY)
  forceSPHZ = np.asarray(forceSPHZ)

  force['forceSPHX'] = forceSPHX
  force['forceSPHY'] = forceSPHY
  force['forceSPHZ'] = forceSPHZ

  return particles, force


def computeGradC(cfg, particles, headerNeighbourGrid, headerNormalGrid, double[:, :] nxArray, double[:, :] nyArray,
                 double[:, :] nzArray, gradient, int SPHoption):
  """ compute lateral forces acting on the particles (SPH component)

  Cython implementation

  Parameters
  ----------
  cfg: configparser
      configuration for DFA simulation
  particles : dict
      particles dictionary at t
  headerNeighbourGrid : dict
      neighbour search header dictionary (information about SPH grid)
  headerNormalGrid : double
      normal grid header dictionary (information about the DEM grid)
  nxArray : 2D numpy array
      x component of the normal vector of the DEM
  nyArray : 2D numpy array
      y component of the normal vector of the DEM
  nzArray : 2D numpy array
      z component of the normal vector of the DEM
  gradient : int
    Return the gradient (if 1) or the force associated (if 0, default)

  Returns
  -------
  GHX : 1D numpy array
      x component of the lateral force
  GHY : 1D numpy array
      y component of the lateral force
  GHZ : 1D numpy array
      z component of the lateral force
  """
  # get all inputs
  # configuration parameters
  cdef double rho = cfg.getfloat('rho')
  cdef double minRKern = cfg.getfloat('minRKern')
  cdef double velMagMin = cfg.getfloat('velMagMin')
  cdef double gravAcc = cfg.getfloat('gravAcc')
  cdef int interpOption = cfg.getint('interpOption')
  cdef int viscOption = cfg.getint('viscOption')

  # grid normal raster information
  cdef double cszNormal = headerNormalGrid['cellsize']
  cdef int nRowsNormal = headerNormalGrid['nrows']
  cdef int nColsNormal = headerNormalGrid['ncols']
  # neighbour search grid information and neighbour information
  cdef double cszNeighbourGrid = headerNeighbourGrid['cellsize']
  cdef int nRowsNeighbourGrid = headerNeighbourGrid['nrows']
  cdef int nColsNeighbourGrid = headerNeighbourGrid['ncols']
  cdef int[:] indPartInCell = particles['indPartInCell']
  cdef int[:] partInCell = particles['partInCell']
  # SPH kernel
  # use "spiky" kernel: w = (rKernel - r)**3 * 10/(pi*rKernel**5)
  cdef double rKernel = cszNeighbourGrid
  cdef double facKernel = 10.0 / (math.pi * rKernel * rKernel * rKernel * rKernel * rKernel)
  cdef double dfacKernel = - 3.0 * facKernel
  # particle information
  cdef double[:] gEff = particles['gEff']
  cdef double[:] mass = particles['m']
  cdef double[:] hArray = particles['h']
  cdef double[:] xArray = particles['x']
  cdef double[:] yArray = particles['y']
  cdef double[:] zArray = particles['z']
  cdef double[:] uxArray = particles['ux']
  cdef double[:] uyArray = particles['uy']
  cdef double[:] uzArray = particles['uz']
  cdef int N = xArray.shape[0]

  # initialize variables and outputs
  cdef double[:] GHX = np.zeros(N, dtype=np.float64)
  cdef double[:] GHY = np.zeros(N, dtype=np.float64)
  cdef double[:] GHZ = np.zeros(N, dtype=np.float64)
  cdef double K1 = 1
  cdef double K2 = 1
  cdef double gravAcc3
  cdef double gradhX, gradhY, gradhZ, uMag, nx, ny, nz, G1, G2, mdwdrr
  cdef double g1, g2, g11, g12, g22, g33
  cdef double x, y, z, ux, uy, uz, vx, vy, wx, wy, uxOrtho, uyOrtho, uzOrtho
  cdef double dx, dy, dz, dux, duy, duz, dn, r, hr, dwdr, wKernel
  cdef int Lx0, Ly0, iCell
  cdef double w[4]
  cdef int lInd, rInd
  cdef int indx, indy
  cdef int k, ic, n, p, l, imax, imin, iPstart, iPend
  cdef int grad = gradient
  # artificial viscosity parameters
  cdef double dwdrr, area
  cdef double pikl, flux
  cdef double mk, ml, hk, hl, ck, cl, lambdakl

  # loop on particles
  for k in range(N):
    gradhX = 0
    gradhY = 0
    gradhZ = 0
    pikl = 0
    G1 = 0
    G2 = 0
    m11 = 0
    m12 = 0
    m22 = 0
    x = xArray[k]
    y = yArray[k]
    z = zArray[k]
    ux = uxArray[k]
    uy = uyArray[k]
    uz = uzArray[k]
    hk = hArray[k]
    mk = mass[k]

    # locate particle in SPH grid
    indx = <int>math.round(x / cszNeighbourGrid)
    indy = <int>math.round(y / cszNeighbourGrid)

    if SPHoption > 1:
        # get normal vector
        Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = DFAtlsC.getCellAndWeights(x, y, nColsNormal, nRowsNormal, cszNormal, interpOption)
        nx, ny, nz = DFAtlsC.getVector(Lx0, Ly0, w[0], w[1], w[2], w[3], nxArray, nyArray, nzArray)
        nx, ny, nz = DFAtlsC.normalize(nx, ny, nz)
        # projection of gravity on normal vector or use the effective gravity (gravity + curvature acceleration part)
        # This was done I computeForce and is passed over here
        gravAcc3 = gEff[k]
        if SPHoption > 2:
            uMag = DFAtlsC.norm(ux, uy, uz)
            if uMag < velMagMin:
                ux = 1
                uy = 0
                uz = -(1*nx + 0*ny) / nz
                ux, uy, uz = DFAtlsC.normalize(ux, uy, uz)
                K1 = 1
                K2 = 1
            else:
                ux, uy, uz = DFAtlsC.normalize(ux, uy, uz)

                uxOrtho, uyOrtho, uzOrtho = DFAtlsC.crossProd(nx, ny, nz, ux, uy, uz)
                uxOrtho, uyOrtho, uzOrtho = DFAtlsC.normalize(uxOrtho, uyOrtho, uzOrtho)

                g1 = nx/(nz)
                g2 = ny/(nz)

    # check if we are on the bottom ot top row!!!
    lInd = -1
    rInd = 2
    if indy == 0:
        lInd = 0
    if indy == nRowsNeighbourGrid - 1:
        rInd = 1
    for n in range(lInd, rInd):
        ic = (indx - 1) + nColsNeighbourGrid * (indy + n)
        # make sure not to take particles from the other edge
        imax = max(ic, nColsNeighbourGrid * (indy + n))
        imin = min(ic+3, nColsNeighbourGrid * (indy + n + 1))
        iPstart = indPartInCell[imax]
        iPend = indPartInCell[imin]
        # loop on all particles in neighbour boxes
        for p in range(iPstart, iPend):
            # index of particle in neighbour box
            l = partInCell[p]
            if k != l:
                dx = xArray[l] - x
                dy = yArray[l] - y
                dz = zArray[l] - z

#----------------------------SPHOPTION = 1--------------------------------------
                # SamosAT style (no reprojecion on the surface, dz = 0 and gz is used)
                if SPHoption == 1:
                    dz = 0
                    # get norm of r = xk - xl
                    r = DFAtlsC.norm(dx, dy, dz)
                    if r < minRKern * rKernel:
                        # impose a minimum distance between particles
                        r = minRKern * rKernel
                    if r < rKernel:
                        hr = rKernel - r
                        dwdr = dfacKernel * hr * hr
                        mdwdrr = mass[l] * dwdr / r
                        gradhX = gradhX + mdwdrr*dx
                        gradhY = gradhY + mdwdrr*dy
                        gradhZ = gradhZ + mdwdrr*dz
                        gravAcc3 = gravAcc

#----------------------------SPHOPTION = 2--------------------------------------
                # Compute the gradient in the cartesian coord system with reprojecion on the surface
                # and dz != 0 and g3 are used
                # It is here also possible to add the artificial viscosity comming from the Ata theory
                elif SPHoption == 2:
                    # like option 1 but with dz!=0
                    # get norm of r = xk - xl
                    r = DFAtlsC.norm(dx, dy, dz)
                    if r < minRKern * rKernel:
                        # impose a minimum distance between particles
                        dx = minRKern * rKernel * dx
                        dy = minRKern * rKernel * dy
                        dz = minRKern * rKernel * dz
                        r = minRKern * rKernel
                    if r < rKernel:
                        hl = hArray[l]
                        hr = rKernel - r
                        dwdrr = dfacKernel * hr * hr / r
                        ml = mass[l]
                        area = ml/(rho*hl)
                        flux = gravAcc3*hl
                        if viscOption == 2:
                            # ATA artificial viscosity
                            dux = uxArray[l] - ux
                            duy = uyArray[l] - uy
                            duz = uzArray[l] - uz
                            ck = math.sqrt(gravAcc3*hk)
                            cl = math.sqrt(gravAcc3*hl)
                            lamdbakl = (ck+cl)/2
                            pikl = - lamdbakl * DFAtlsC.scalProd(dux, duy, duz, dx, dy, dz) / r
                        # SPH gradient computation - standard SPH formulation
                        gradhX = gradhX + (flux + pikl)*dwdrr*dx*area
                        gradhY = gradhY + (flux + pikl)*dwdrr*dy*area
                        gradhZ = gradhZ + (flux + pikl)*dwdrr*dz*area


#----------------------------SPHOPTION = 3--------------------------------------
                # Compute the gradient in the local coord system (will allow us to add the earth pressure coef)
                # and this time reprojecion on the surface, dz != 0 and g3 are used
                if SPHoption == 3:
                  # get coordinates in local coord system
                  r1 = DFAtlsC.scalProd(dx, dy, dz, ux, uy, uz)
                  r2 = DFAtlsC.scalProd(dx, dy, dz, uxOrtho, uyOrtho, uzOrtho)
                  # impse r3=0 even if the particle is not exactly on the tengent plane
                  # get norm of r = xk - xl
                  r = DFAtlsC.norm(r1, r2, 0)
                  if r < minRKern * rKernel:
                      # impose a minimum distance between particles
                      r1 = minRKern * rKernel * r1
                      r2 = minRKern * rKernel * r2
                      r = minRKern * rKernel
                  if r < rKernel:
                      hr = rKernel - r
                      dwdr = dfacKernel * hr * hr
                      ml = mass[l]
                      mdwdrr = ml * dwdr / r
                      G1 = mdwdrr * K1*r1
                      G2 = mdwdrr * K2*r2

                      gradhX = gradhX + ux*G1 + uxOrtho*G2
                      gradhY = gradhY + uy*G1 + uyOrtho*G2
                      gradhZ = gradhZ + (- g1*(ux*G1 + uxOrtho*G2) - g2*(uy*G1 + uyOrtho*G2))


    if grad == 1:
      if SPHoption == 2:
        GHX[k] = GHX[k] + gradhX / gravAcc3
        GHY[k] = GHY[k] + gradhY / gravAcc3
        GHZ[k] = GHZ[k] + gradhZ / gravAcc3
      else:
        GHX[k] = GHX[k] - gradhX / rho
        GHY[k] = GHY[k] - gradhY / rho
        GHZ[k] = GHZ[k] - gradhZ / rho
    elif grad == 0:
      if SPHoption == 2:
        # grad here is g*grad(h)
        GHX[k] = GHX[k] + gradhX * mk
        GHY[k] = GHY[k] + gradhY * mk
        GHZ[k] = GHZ[k] + gradhZ * mk
      else :
        GHX[k] = GHX[k] + gradhX*gravAcc3 / rho* mk
        GHY[k] = GHY[k] + gradhY*gravAcc3 / rho* mk
        GHZ[k] = GHZ[k] + gradhZ*gravAcc3 / rho* mk


  return GHX, GHY, GHZ

def computeIniMovement(cfg, particles, dem, dT, fields):
  """ add artifical viscosity effect on velocity

      Parameters
      ------------
      cfg: configparser
          configuration for DFA simulation
      particles : dict
          particles dictionary at t
      dem : dict
          dictionary with dem information
      dT : float
          time step
      fields: dict
        fields dictionary

      Returns
      --------
      particles: dict
        updated particle dictionary
      force: dict
        force dictionary

  """

  cdef int interpOption = cfg.getint('interpOption')
  cdef double rho = cfg.getfloat('rho')
  cdef double subgridMixingFactor = cfg.getfloat('subgridMixingFactorIni')
  cdef double gravAcc = cfg.getfloat('gravAcc')
  cdef double dt = dT
  cdef int nPart = particles['nPart']
  cdef double csz = dem['header']['cellsize']
  cdef int nrows = dem['header']['nrows']
  cdef int ncols = dem['header']['ncols']
  cdef double[:, :] ZDEM = dem['rasterData']
  cdef double[:, :] nxArray = dem['Nx']
  cdef double[:, :] nyArray = dem['Ny']
  cdef double[:, :] nzArray = dem['Nz']
  cdef double[:, :] areaRatser = dem['areaRaster']
  cdef double[:] mass = particles['m']
  cdef double[:] hArray = particles['h']
  cdef double[:] xArray = particles['x']
  cdef double[:] yArray = particles['y']
  cdef double[:] zArray = particles['z']
  cdef double[:] uxArray = particles['ux']
  cdef double[:] uyArray = particles['uy']
  cdef double[:] uzArray = particles['uz']
  cdef int[:] indXDEM = particles['indXDEM']
  cdef int[:] indYDEM = particles['indYDEM']
  cdef double[:, :] VX = fields['Vx']
  cdef double[:, :] VY = fields['Vy']
  cdef double[:, :] VZ = fields['Vz']
  cdef double[:] forceX = np.zeros(nPart, dtype=np.float64)
  cdef double[:] forceY = np.zeros(nPart, dtype=np.float64)
  cdef double[:] forceZ = np.zeros(nPart, dtype=np.float64)
  cdef double[:] forceFrict = np.zeros(nPart, dtype=np.float64)
  cdef double[:] dM = np.zeros(nPart, dtype=np.float64)
  cdef double[:] dMDet = np.zeros(nPart, dtype=np.float64)
  cdef double[:] curvAcc = np.zeros(nPart, dtype=np.float64)
  cdef double[:] gEff = np.zeros(nPart, dtype=np.float64)

  cdef int indCellX, indCellY
  cdef double areaPart, uMag, m, h
  cdef double vMeanx, vMeany, vMeanz, vMeanNorm, dvX, dvY, dvZ
  cdef double x, y, z, xEnd, yEnd, zEnd, ux, uy, uz, uxDir, uyDir, uzDir
  cdef double nx, ny, nz, nxEnd, nyEnd, nzEnd, nxAvg, nyAvg, nzAvg
  # variables for interpolation
  cdef int Lx0, Ly0, LxEnd0, LyEnd0, iCell, iCellEnd
  cdef double w[4]
  cdef double wEnd[4]
  cdef int k
  force = {}

  for k in range(nPart):
      m = mass[k]
      x = xArray[k]
      y = yArray[k]
      z = zArray[k]
      h = hArray[k]
      ux = uxArray[k]
      uy = uyArray[k]
      uz = uzArray[k]
      indCellX = indXDEM[k]
      indCellY = indYDEM[k]

      # deduce area
      areaPart = m / (h * rho)
      # get cell and weights
      Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = DFAtlsC.getCellAndWeights(x, y, ncols, nrows, csz, interpOption)

      # get normal at the particle location
      nx, ny, nz = DFAtlsC.getVector(Lx0, Ly0, w[0], w[1], w[2], w[3], nxArray, nyArray, nzArray)
      nx, ny, nz = DFAtlsC.normalize(nx, ny, nz)

      # add artificial viscosity
      ux, uy, uz = addArtificialViscosity(m, h, dt, rho, ux, uy, uz, subgridMixingFactor, Lx0, Ly0,
                                          w[0], w[1], w[2], w[3], VX, VY, VZ, nx, ny, nz)


      # update velocity
      uxArray[k] = ux
      uyArray[k] = uy
      uzArray[k] = uz
      gEff[k] = gravAcc * nz

  # save results
  force['dM'] = np.asarray(dM)
  force['dMDet'] = np.asarray(dMDet)
  force['forceX'] = np.asarray(forceX)
  force['forceY'] = np.asarray(forceY)
  force['forceZ'] = np.asarray(forceZ)
  force['forceFrict'] = np.asarray(forceFrict)
  particles['gEff'] = np.asarray(gEff)
  particles['curvAcc'] = np.asarray(curvAcc)
  particles['ux'] = np.asarray(uxArray)
  particles['uy'] = np.asarray(uyArray)
  particles['uz'] = np.asarray(uzArray)
  particles['m'] = np.asarray(mass)

  return particles, force


def updateInitialVelocity(cfg, particles, dem, float velocityMag):
  """
  update particles' velocity in direction of the steepest descent

  Parameters
  ------------
  cfg: configparser
    configuration for DFA simulation
  particles : dict
    particles dictionary at t
  dem : dict
    dictionary with dem information
  velocityMag: float
    velocity of the particles

  Returns
  ------------
  particles : dict
    particles dictionary at t with updated velocity
  """

  cdef double[:] xArray = particles['x']
  cdef double[:] yArray = particles['y']
  cdef double[:] uxArray = particles['ux']
  cdef double[:] uyArray = particles['uy']
  cdef double[:] uzArray = particles['uz']
  cdef double[:] velocityMagArray = particles['velocityMag']
  cdef int nPart = particles['nPart']
  cdef int nrows = dem['header']['nrows']
  cdef int ncols = dem['header']['ncols']
  cdef double[:, :] nxArray = dem['Nx']
  cdef double[:, :] nyArray = dem['Ny']
  cdef double[:, :] nzArray = dem['Nz']
  cdef int interpOption = cfg.getint('interpOption')
  cdef double csz = dem['header']['cellsize']

  cdef double x, y, z
  cdef double nx, ny, nz
  cdef double tangentialTopoX, tangentialTopoY, tangentialTopoZ, tangentialTopoXNorm, tangentialTopoYNorm, tangentialTopoZNorm
  cdef int Lx0, Ly0, iCell
  cdef double w[4]
  cdef int k

  for k in range(nPart):
    x = xArray[k]
    y = yArray[k]

    # get cell and weights
    Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = DFAtlsC.getCellAndWeights(x, y, ncols, nrows, csz, interpOption)

    # get normal at the particle location
    nx, ny, nz = DFAtlsC.getVector(Lx0, Ly0, w[0], w[1], w[2], w[3], nxArray, nyArray, nzArray)
    nx, ny, nz = DFAtlsC.normalize(nx, ny, nz)
    # get vector in the tangent plane that points in the direction gravity would pull a particle along the surface
    tangentialTopoX = nx * nz
    tangentialTopoY = ny * nz
    tangentialTopoZ = nz * nz -1
    tangentialTopoXNorm, tangentialTopoYNorm, tangentialTopoZNorm = DFAtlsC.normalize(tangentialTopoX, tangentialTopoY, tangentialTopoZ)

    uxArray[k] = tangentialTopoXNorm * velocityMag
    uyArray[k] = tangentialTopoYNorm * velocityMag
    uzArray[k] = tangentialTopoZNorm * velocityMag

    velocityMagArray[k] = DFAtlsC.norm(uxArray[k], uyArray[k], uzArray[k])
  particles['ux'] = np.asarray(uxArray)
  particles['uy'] = np.asarray(uyArray)
  particles['uz'] = np.asarray(uzArray)
  particles['velocityMag'] = np.asarray(velocityMagArray)

  return particles