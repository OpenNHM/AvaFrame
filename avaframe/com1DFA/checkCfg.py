"""
Check if the .ini file is consistent for com1DFA computations
"""

import logging
import numpy as np

# create local logger
log = logging.getLogger(__name__)


def checkCfgConsistency(cfg):
    """Check the provided configuration for necessary consistency/relation between
        parameters for com1DFA and for plausibility.

    Parameters
    -----------
    cfg : configuration object
        com1DFA configuration
    Returns
    --------
    True if checks are passed, otherwise error is thrown
    """

    # Check if Ata Parameters are consistent
    sphOption = float(cfg["GENERAL"]["sphOption"])
    viscOption = float(cfg["GENERAL"]["viscOption"])

    if viscOption == 2:
        if sphOption != 2:
            # raise an error
            message = (
                "If viscOption is set to 2 (Ata viscosity), sphOption = 2 is needed "
                "(or implement the Ata viscosity for other sphOption values)"
            )
            log.error(message)
            raise AssertionError(message)
        else:
            log.debug("Ata parameters are consistent")

    num = float(cfg["GENERAL"]["methodMeshNormal"])
    if num not in [1, 4, 6, 8]:
        # raise an error
        message = (
            "The methodMeshNormal set in the ini file does NOT have a plausible value. "
            "Please set the parameter either to 1, 4, 6 or 8."
        )
        log.error(message)
        raise AssertionError(message)
    else:
        log.debug("MethodMeshNormal value is plausible")

    return True


def checkCellSizeKernelRadius(cfg):
    """Check if sphKernelRadius is set to match the meshCellSize if so adapt sphKernelRadius value

    Parameters
    -----------
    cfg : dict
        configuration settings here used: sphKernelRadius, meshCellSize

    Returns
    --------
    cfg : dict
        updated configuration settings here used: sphKernelRadius

    """

    # Check if Ata Parameters are consistent
    sphKernelRadius = cfg["GENERAL"]["sphKernelRadius"]
    meshCellSize = cfg["GENERAL"]["meshCellSize"]

    if sphKernelRadius == "meshCellSize":
        cfg["GENERAL"]["sphKernelRadius"] = cfg["GENERAL"]["meshCellSize"]
        log.info("sphKernelRadius is set to match meshCellSize of %s meters" % meshCellSize)

    return cfg


def checkCfgFrictionModel(cfg, inputSimFiles, relVolume=""):
    """check which friction model is chosen and if friction model parameters are of valid type
    if samosATAuto - check if

    relVolume < volClassSmall - set frictModel=samosATSmall
    volClassSmall <= relVolume < volClassMedium - set frictModel= samosAtMedium
    relVolume >= volClassMedium - set frictModel=samosAT

    Parameters
    -------------
    cfg: dict
        configuration settings
    inputSimFiles: dict
        info about available input files for simulation
    relVolume: float
        The release volume; optional - only needed if samosATAuto is set

    Returns
    --------
    cfg: dict
        updated configuration settings
    """

    frictParameters = [
        "musamosat",
        "tau0samosat",
        "Rs0samosat",
        "kappasamosat",
        "Rsamosat",
        "Bsamosat",
        "muvoellmy",
        "xsivoellmy",
        "mucoulomb",
        "mu0wetsnow",
        "xsiwetsnow",
        "musamosatsmall",
        "tau0samosatsmall",
        "Rs0samosatsmall",
        "kappasamosatsmall",
        "Rsamosatsmall",
        "Bsamosatsmall",
        "musamosatmedium",
        "tau0samosatmedium",
        "Rs0samosatmedium",
        "kappasamosatmedium",
        "Rsamosatmedium",
        "Bsamosatmedium",
        "mucoulombminshear",
        "tau0coulombminshear",
        "muvoellmyminshear",
        "xsivoellmyminshear",
        "etaObrienAndJulien",
        "alpha1EtaObrienAndJulien",
        "beta1EtaObrienAndJulien",
        "tauyObrienAndJulien",
        "alpha2TauyObrienAndJulien",
        "beta2TauyObrienAndJulien",
        "alphaObrienAndJulien",
        "tauyHerschelAndBulkley",
        "alpha2TauyHerschelAndBulkley",
        "beta2TauyHerschelAndBulkley",
        "kHerschelAndBulkley",
        "nHerschelAndBulkley",
        "etaBingham",
        "alpha1EtaBingham",
        "beta1EtaBingham",
        "tauyBingham",
        "alpha2TauyBingham",
        "beta2TauyBingham",
    ]

    # if samosATAuto check for release volume and volume class to determine which parameter setup
    if cfg["GENERAL"]["frictModel"].lower() == "samosatauto":
        if relVolume < float(cfg["GENERAL"]["volClassSmall"]):
            cfg["GENERAL"]["frictModel"] = "samosATSmall"
        elif float(cfg["GENERAL"]["volClassSmall"]) <= relVolume < float(cfg["GENERAL"]["volClassMedium"]):
            cfg["GENERAL"]["frictModel"] = "samosATMedium"
        elif relVolume > float(cfg["GENERAL"]["volClassMedium"]):
            cfg["GENERAL"]["frictModel"] = "samosAT"
        log.info(
            "samosATAuto - %.2f meter grid based release volume is %.2f and hence friction model: %s is chosen"
            % (
                float(cfg["GENERAL"]["meshCellSize"]),
                relVolume,
                cfg["GENERAL"]["frictModel"],
            )
        )

    # fetch friction model
    frictModel = cfg["GENERAL"]["frictModel"]
    # remove friction parameter values that are not used
    if frictModel.lower() == "samosat":
        frictParameterIn = [
            frictModel.lower() in frictP and "small" not in frictP and "medium" not in frictP
            for frictP in frictParameters
        ]
    else:
        if frictModel.lower() == "coulomb":
            frictParameterIn = [
                frictModel.lower() in frictP and "coulombminshear" not in frictP
                for frictP in frictParameters
            ]
        elif frictModel.lower() == "voellmy":
            frictParameterIn = [
                frictModel.lower() in frictP and "voellmyminshear" not in frictP
                for frictP in frictParameters
            ]
        elif frictModel.lower() == "obrienandjulien":
            if cfg["GENERAL"]["etaObrienAndJulien"] == "0":
                frictParameters.remove("etaObrienAndJulien")
            else:
                frictParameters.remove("alpha1EtaObrienAndJulien")
                frictParameters.remove("beta1EtaObrienAndJulien")
                cfg["GENERAL"]["alpha1EtaObrienAndJulien"] = "0"
                cfg["GENERAL"]["beta1EtaObrienAndJulien"] = "0"

            if cfg["GENERAL"]["tauyObrienAndJulien"] == "0":
                frictParameters.remove("tauyObrienAndJulien")
            else:
                frictParameters.remove("alpha2TauyObrienAndJulien")
                frictParameters.remove("beta2TauyObrienAndJulien")
                cfg["GENERAL"]["alpha2TauyObrienAndJulien"] = "0"
                cfg["GENERAL"]["beta2TauyObrienAndJulien"] = "0"

            frictParameterIn = [frictModel.lower() in frictP.lower() for frictP in frictParameters]
        elif frictModel.lower() == "herschelandbulkley":
            if cfg["GENERAL"]["tauyHerschelAndBulkley"] == "0":
                frictParameters.remove("tauyHerschelAndBulkley")
            else:
                frictParameters.remove("alpha2TauyHerschelAndBulkley")
                frictParameters.remove("beta2TauyHerschelAndBulkley")
                cfg["GENERAL"]["alpha2TauyHerschelAndBulkley"] = "0"
                cfg["GENERAL"]["beta2TauyHerschelAndBulkley"] = "0"

            frictParameterIn = [frictModel.lower() in frictP.lower() for frictP in frictParameters]
        elif frictModel.lower() == "bingham":
            if cfg["GENERAL"]["etaBingham"] == "0":
                frictParameters.remove("etaBingham")
            else:
                frictParameters.remove("alpha1EtaBingham")
                frictParameters.remove("beta1EtaBingham")
                cfg["GENERAL"]["alpha1EtaBingham"] = "0"
                cfg["GENERAL"]["beta1EtaBingham"] = "0"

            if cfg["GENERAL"]["tauyBingham"] == "0":
                frictParameters.remove("tauyBingham")
            else:
                frictParameters.remove("alpha2TauyBingham")
                frictParameters.remove("beta2TauyBingham")
                cfg["GENERAL"]["alpha2TauyBingham"] = "0"
                cfg["GENERAL"]["beta2TauyBingham"] = "0"

            frictParameterIn = [frictModel.lower() in frictP.lower() for frictP in frictParameters]
        else:
            frictParameterIn = [frictModel.lower() in frictP for frictP in frictParameters]

    for indexP, frictP in enumerate(frictParameters):
        if frictParameterIn[indexP]:
            noF = False
            try:
                float(cfg["GENERAL"][frictP])
            except ValueError:
                noF = True
            if noF:
                message = "Friction model used %s, but %s is not of valid float type" % (
                    frictModel,
                    cfg["GENERAL"][frictP],
                )
                log.error(message)
                raise ValueError(message)
            elif np.isnan(float(cfg["GENERAL"][frictP])):
                message = "Friction model used %s, but %s is nan - not valid" % (
                    frictModel,
                    frictP,
                )
                log.error(message)
                raise ValueError(message)
            else:
                log.info(
                    "Friction model parameter used: %s with value %s" % (frictP, cfg["GENERAL"][frictP])
                )

    # if spatialVoellmy check if mu and xi file are provided
    if frictModel.lower() == "spatialvoellmy":
        if inputSimFiles["muFile"] is None:
            message = "No file for mu found"
            log.error(message)
            raise FileNotFoundError(message)
        elif inputSimFiles["xiFile"] is None:
            message = "No file for xi found"
            log.error(message)
            raise FileNotFoundError(message)
        else:
            log.info(
                "Mu field initialized from: %s and xi field from: %s and read from: %s, %s"
                % (
                    inputSimFiles["muFile"].name,
                    inputSimFiles["xiFile"].name,
                    cfg["INPUT"]["muFile"],
                    cfg["INPUT"]["xiFile"],
                )
            )
    # O´Brien and Julien friction model
    # fetch parameters and check if cvSediment < cvMaxSediment
    if frictModel.lower() == "obrienandjulien":
        cvSediment = cfg["GENERAL"]["cvSediment"]
        cvMaxSediment = cfg["GENERAL"]["cvMaxSediment"]
        if cvSediment >= cvMaxSediment:
            message = "cvSediment must be smaller than cvMaxSediment"
            log.error(message)
            raise ValueError(message)

    # check if explicitFriction has valid values
    if cfg["GENERAL"]["explicitFriction"] not in ["0", "1"]:
        message = "explicitFriction should be either 0 (implicit method) or 1 (explicit method)."
        log.error(message)
        raise ValueError(message)

    return cfg
