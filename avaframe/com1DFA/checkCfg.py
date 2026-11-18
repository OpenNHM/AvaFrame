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
        "muIMuS",
        "muIMu2",
        "muII0",
        "muIGrainSize",
        "muIGrainDensity",
        "muIShearFactor",
        "hysteresisMu1",
        "hysteresisMu2",
        "hysteresisMu3",
        "hysteresisL",
        "hysteresisBeta",
        "hysteresisBetaStar",
        "hysteresisGamma",
        "hysteresisViscosity",
        "hysteresisNu",
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
        elif frictModel.lower() in ["mui", "hysteresis"]:
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

    # mu(I) friction model: check parameter ranges and plausibility
    if frictModel.lower() == "mui":
        checkMuIParameters(cfg)

    # hysteresis friction model: check parameter ranges, plausibility and required numerical settings
    if frictModel.lower() == "hysteresis":
        checkHysteresisParameters(cfg)

    # check if explicitFriction has valid values
    if cfg["GENERAL"]["explicitFriction"] not in ["0", "1"]:
        message = "explicitFriction should be either 0 (implicit method) or 1 (explicit method)."
        log.error(message)
        raise ValueError(message)

    return cfg


def checkMuIParameters(cfg):
    """check the mu(I) friction model parameters for validity (errors) and plausibility (warnings)

    Parameters
    -----------
    cfg: dict
        configuration settings, here used: rho and the muI parameters in GENERAL

    Raises
    -------
    ValueError
        if a parameter is not larger than 0 or if muIMu2 <= muIMuS
    """

    muIParameters = ["muIMuS", "muIMu2", "muII0", "muIGrainSize", "muIGrainDensity", "muIShearFactor"]
    for muIP in muIParameters:
        if float(cfg["GENERAL"][muIP]) <= 0:
            message = "Friction model used muI, but %s must be larger than 0" % muIP
            log.error(message)
            raise ValueError(message)

    if float(cfg["GENERAL"]["muIMu2"]) <= float(cfg["GENERAL"]["muIMuS"]):
        message = "Friction model used muI, but muIMu2 must be larger than muIMuS"
        log.error(message)
        raise ValueError(message)

    # a grain density below the flow density suggests the bulk density was entered
    if float(cfg["GENERAL"]["muIGrainDensity"]) < float(cfg["GENERAL"]["rho"]):
        log.warning(
            "muIGrainDensity (%s) is smaller than the flow density rho (%s): muIGrainDensity is the grain "
            "density, not the bulk density" % (cfg["GENERAL"]["muIGrainDensity"], cfg["GENERAL"]["rho"])
        )

    if float(cfg["GENERAL"]["muIShearFactor"]) != 2.5:
        log.warning(
            "muIShearFactor is %s, the default of 2.5 corresponds to a Bagnold velocity profile"
            % cfg["GENERAL"]["muIShearFactor"]
        )


def checkHysteresisParameters(cfg):
    """check the hysteresis friction model parameters for validity (errors) and plausibility (warnings)

    The yield condition of material at rest is only exact with the explicit friction method, so
    explicitFriction = 1 is required.

    Parameters
    -----------
    cfg: dict
        configuration settings

    Raises
    -------
    ValueError
        if hysteresisMu1, hysteresisL, hysteresisBeta or hysteresisBetaStar is not larger than 0,
        if hysteresisGamma is negative,
        if hysteresisMu2 <= hysteresisMu1 or hysteresisMu3 <= hysteresisMu1,
        if explicitFriction is not 1
    """

    cfgGen = cfg["GENERAL"]
    for hystP in ["hysteresisMu1", "hysteresisL", "hysteresisBeta", "hysteresisBetaStar"]:
        if float(cfgGen[hystP]) <= 0:
            message = "Friction model used hysteresis, but %s must be larger than 0" % hystP
            log.error(message)
            raise ValueError(message)

    if float(cfgGen["hysteresisGamma"]) < 0:
        message = "Friction model used hysteresis, but hysteresisGamma must not be negative"
        log.error(message)
        raise ValueError(message)

    mu1 = float(cfgGen["hysteresisMu1"])
    mu2 = float(cfgGen["hysteresisMu2"])
    mu3 = float(cfgGen["hysteresisMu3"])
    for muName, mu in [("hysteresisMu2", mu2), ("hysteresisMu3", mu3)]:
        if mu <= mu1:
            message = "Friction model used hysteresis, but %s must be larger than hysteresisMu1" % muName
            log.error(message)
            raise ValueError(message)

    # the friction should decrease with Fr in the intermediate regime: muStart(h) > muDyn(h, betaStar) for all h.
    # With x = h / L and r = beta / (betaStar + gamma): muStart - muDyn = (mu3 - mu1) + (mu2 - mu1) *
    # (1 / (1 + x) - 1 / (1 + r x)), which for r < 1 has its minimum at x = 1 / sqrt(r)
    ratio = float(cfgGen["hysteresisBeta"]) / (
        float(cfgGen["hysteresisBetaStar"]) + float(cfgGen["hysteresisGamma"])
    )
    if ratio < 1.0:
        minMargin = (mu3 - mu1) - (mu2 - mu1) * (1.0 - np.sqrt(ratio)) / (1.0 + np.sqrt(ratio))
        if minMargin <= 0:
            log.warning(
                "hysteresis friction: the static friction muStart(h) is smaller than the dynamic friction at "
                "hysteresisBetaStar for some flow thicknesses, so the friction increases with Fr in the intermediate "
                "regime (minimum of muStart - muDyn: %.4f)" % minMargin
            )

    if cfgGen["explicitFriction"] != "1":
        message = (
            "Friction model used hysteresis, but explicitFriction is %s: the hysteresis friction model requires "
            "explicitFriction = 1, material at rest is only held by the static friction with the explicit "
            "friction method" % cfgGen["explicitFriction"]
        )
        log.error(message)
        raise ValueError(message)

    if str(cfgGen.get("adaptSfcStopped", "0")).lower() in ["1", "true"]:
        log.warning(
            "hysteresis friction with adaptSfcStopped = 1: particles that come to rest are removed, so deposits "
            "cannot start moving again and do not contribute to the flow thickness"
        )

    # depth-averaged viscous term
    viscosityOption = cfgGen.get("hysteresisViscosity", "0")
    if viscosityOption not in ["0", "1", "2"]:
        message = (
            "Friction model used hysteresis, but hysteresisViscosity is %s, must be 0, 1 or 2"
            % viscosityOption
        )
        log.error(message)
        raise ValueError(message)
    if float(cfgGen.get("hysteresisNu", "0")) < 0:
        message = "Friction model used hysteresis, but hysteresisNu must not be negative"
        log.error(message)
        raise ValueError(message)
    if viscosityOption != "0" and float(cfgGen.get("hysteresisNu", "0")) == 0:
        log.warning(
            "hysteresisViscosity = %s but hysteresisNu = 0: the viscous term has no effect" % viscosityOption
        )


def checkCfgEntrainmentMethod(cfg):
    """check the entrainment method (entMethod) and the settings it requires

    entMethod = staticParticles initialises the entrainment layer as particles at rest. It requires the hysteresis
    friction model (static layers are held by the static friction), and it is not compatible with
    entrainableDeposition (deposits are particles at rest already), with snowSlide (no bonds for the layer
    particles) or with adaptSfcStopped (the layer particles at rest would be removed in the first time step).
    adaptSfcEntrainment has no effect (warning).

    Parameters
    -----------
    cfg: dict
        configuration settings, here used: entMethod, entPartDistType, entLayerInFields, frictModel,
        entrainableDeposition, snowSlide, adaptSfcStopped, adaptSfcEntrainment and viscOption in GENERAL

    Raises
    -------
    ValueError
        if entMethod, entPartDistType or entLayerInFields has an invalid value or
        if entMethod = staticParticles is combined with a friction model other than hysteresis,
        with entrainableDeposition, with snowSlide or with adaptSfcStopped
    """

    cfgGen = cfg["GENERAL"]
    entMethod = cfgGen.get("entMethod", "rate")
    if entMethod not in ["rate", "staticParticles"]:
        message = "entMethod is %s, but must be either rate or staticParticles" % entMethod
        log.error(message)
        raise ValueError(message)

    if entMethod == "staticParticles":
        if cfgGen["frictModel"].lower() != "hysteresis":
            message = (
                "entMethod = staticParticles requires frictModel = hysteresis, but frictModel is %s: with other "
                "friction models the layer is not held by a static friction" % cfgGen["frictModel"]
            )
            log.error(message)
            raise ValueError(message)
        if str(cfgGen.get("entrainableDeposition", "False")).lower() in ["1", "true"]:
            message = (
                "entMethod = staticParticles is not compatible with entrainableDeposition = True: deposits remain "
                "particles that can start to flow again"
            )
            log.error(message)
            raise ValueError(message)
        if str(cfgGen.get("snowSlide", "0")) == "1":
            message = "entMethod = staticParticles is not compatible with snowSlide = 1"
            log.error(message)
            raise ValueError(message)
        if str(cfgGen.get("adaptSfcStopped", "0")).lower() in ["1", "true"]:
            message = (
                "entMethod = staticParticles is not compatible with adaptSfcStopped = 1: the layer particles at rest "
                "would be removed in the first time step"
            )
            log.error(message)
            raise ValueError(message)
        if str(cfgGen.get("adaptSfcEntrainment", "0")).lower() in ["1", "true"]:
            log.warning(
                "adaptSfcEntrainment = 1 has no effect with entMethod = staticParticles (no entrainment rate)"
            )
        entLayerInFields = str(cfgGen.get("entLayerInFields", "False"))
        if entLayerInFields.lower() not in ["true", "false", "1", "0", "yes", "no", "on", "off"]:
            message = "entLayerInFields is %s, but must be True or False" % entLayerInFields
            log.error(message)
            raise ValueError(message)
        partDistType = cfgGen.get("entPartDistType", "uniform")
        if partDistType.lower() not in ["random", "semirandom", "uniform", "triangular"]:
            message = (
                "entPartDistType is %s, but must be one of random, semirandom, uniform or triangular"
                % partDistType
            )
            log.error(message)
            raise ValueError(message)
        if str(cfgGen.get("viscOption", "1")) != "1":
            log.warning(
                "entMethod = staticParticles with viscOption = %s: erosion of the layer relies on the momentum "
                "exchange of the subgrid mixing (viscOption = 1), without it the flow can pass over the layer"
                % cfgGen.get("viscOption")
            )

    return cfg
