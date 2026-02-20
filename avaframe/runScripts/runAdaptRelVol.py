"""
Run adaption of release volme by modifying the release thickness
"""
from pathlib import Path
import tempfile
import shutil
import logging

# Local imports
from avaframe.com1DFA import DFAtools as DFAtls
from avaframe.in1Data import getInput as gI
import avaframe.in2Trans.rasterUtils as IOf
from avaframe.in3Utils import cfgUtils
import avaframe.in3Utils.geoTrans as geoTrans
from avaframe.in3Utils import logUtils
import avaframe.in2Trans.shpConversion as shpConv
from avaframe.com1DFA import com1DFA
import avaframe.com1DFA.com1DFATools as com1DFATools

def runAdaptRelVol(avaDir, cfgDebris, relThVal):
    '''
    '''

    # initialize logging
    logUtils.initiateLogger(avaDir, "getReleaseVolume")
    # logging.getLogger().setLevel(logging.CRITICAL)
    logging.disable(logging.INFO)


    # ------------------------------------------------------------------ #
    # cfgMain aufbauen (wird von com1DFAPreprocess benötigt)
    # ------------------------------------------------------------------ #
    cfgMain = cfgUtils.getModuleConfig(com1DFA)
    cfgMain["MAIN"] = {"avalancheDir": str(avaDir)}

    # Work-Verzeichnis bereinigen falls vorhanden (com1DFAPreprocess benötigt leeres Work-Dir)
    workDir = Path(avaDir) / "Work" / "com1DFA"
    if workDir.exists():
        shutil.rmtree(workDir)
        print(f"Work-Verzeichnis bereinigt: {workDir}")
    
    # ------------------------------------------------------------------ #
    # Preprocessing: vollständig aufgelöste Konfiguration + Input-Dateien
    # ------------------------------------------------------------------ #
    simDict, outDir, inputSimFiles, _ = com1DFA.com1DFAPreprocess(cfgMain, cfgInfo="")

    # Erste Simulation als Vorlage nehmen
    cuSimName = list(simDict.keys())[0]
    cfgSim = simDict[cuSimName]["cfgSim"]
    releaseFile = simDict[cuSimName]["relFile"]

    print(f"Release-Datei:   {releaseFile}")
    print(f"Simulations-Typ: {simDict[cuSimName]['simType']}")
    print()

    # outDir für initializeSimulation (kein Schreiben gewünscht, temp-Verzeichnis)
    outDirTmp = Path(tempfile.mkdtemp())

    # ------------------------------------------------------------------ #
    # Volumen für verschiedene Release-Dicken berechnen
    # ------------------------------------------------------------------ #
    rho = cfgSim["GENERAL"].getfloat("rho")
    print(f"Schneedichte rho = {rho} kg/m³\n")
    print(f"{'relTh [m]':>12}  {'Volumen [m³]':>14}  {'Masse [kg]':>14}  {'Partikel':>10}")
    print("-" * 56)

    volumes = {}

    for relTh in relThVal:

        # relTh in der aufgelösten Konfiguration überschreiben
        cfgSim["GENERAL"]["relTh"] = str(relTh)
        # sicherstellen dass Dicke nicht aus Datei/Shapefile kommt
        cfgSim["GENERAL"]["relThFromFile"] = "False"
        cfgSim["GENERAL"]["timeDependentRelease"] = "False"

        # Input-Dateien für diese Simulation vorbereiten
        inputSimFilesSim = inputSimFiles.copy()
        inputSimFilesSim = gI.selectReleaseFile(inputSimFilesSim, simDict[cuSimName]["releaseScenario"])

        # DEM und inputSimLines aufbereiten
        demOri, inputSimLines = com1DFA.prepareInputData(inputSimFilesSim, cfgSim)

        # Dicken auf releaseLine setzen
        _, inputSimLines, _ = com1DFA.prepareReleaseEntrainment(
            cfgSim, inputSimFilesSim["releaseScenario"], inputSimLines
        )

        # Simulation initialisieren (t=0) → Partikel erzeugen
        particles, fields, dem, reportAreaInfo = com1DFA.initializeSimulation(
            cfgSim, outDirTmp, demOri, inputSimLines, "getReleaseVolume"
        )

        # tatsächliches Simulationsvolumen
        mTot = particles["mTot"]
        volume = mTot / rho
        nPart = particles["nPart"]

        volumes[relTh] = volume
        print(f"{relTh:>12.2f}  {volume:>14.2f}  {mTot:>14.2f}  {nPart:>10d}")

    # ------------------------------------------------------------------ #
    # Zusammenfassung
    # ------------------------------------------------------------------ #
    print("\n--- Zusammenfassung ---")
    ref_th = relThVal[0]
    ref_vol = volumes[ref_th]
    print(f"Referenz: relTh = {ref_th} m → V = {ref_vol:.2f} m³\n")
    for relTh, vol in volumes.items():
        factor = vol / ref_vol
        print(f"  relTh = {relTh:.2f} m → V = {vol:.2f} m³  (Faktor {factor:.2f}x)")

    # # initialize logging
    # logUtils.initiateLogger(avaDir, "getReleaseVolume")

    # cfgGen = cfgDebris['GENERAL']

    # # Sicherstellen, dass INPUT-Sektion existiert und relThFile leer ist
    # # (leer = Dicke kommt aus cfg["GENERAL"]["relTh"], nicht aus einer Datei)
    # if not cfgDebris.has_section("INPUT"):
    #     cfgDebris.add_section("INPUT")
    # cfgDebris["INPUT"]["relThFile"] = ""
    # cfgDebris["INPUT"]["secondaryRelThFile"] = ""
    # cfgGen["timeDependentRelease"] = "False"
    # cfgGen["relThFromFile"] = "False"
    # cfgGen["secRelArea"] = "False"

    # # debris-flow density [kg/m³]
    # rho = cfgGen.getfloat("rho")
    # print(f"debris-flow density rho = {rho} kg/m³\n")

    # # read input
    # inputSimFiles = gI.getInputDataCom1DFA(avaDir)
    # pathToDem = inputSimFiles["demFile"]

    # # Erstes Release-Szenario (Shapefile) verwenden
    # releaseFile = inputSimFiles["relFiles"][0]
    # secondaryReleaseFile = inputSimFiles["secondaryRelFile"]

    # print(f"DEM:             {pathToDem}")
    # print(f"Release-Datei:   {releaseFile}")
    # print()

    # # ------------------------------------------------------------------ #
    # # Volumen für verschiedene Release-Dicken berechnen
    # # ------------------------------------------------------------------ #
    # print(f"{'relTh [m]':>12}  {'Volumen [m³]':>14}")
    # print("-" * 30)

    # volumes = {}


    # # relTh in der Konfiguration überschreiben
    # cfgDebris["GENERAL"]["relTh"] = str(relThVal[1])

    # # fetchRelVolume erwartet cfg als einfaches dict (konvertiert intern zurück zu ConfigParser)
    # cfgDict = {section: dict(cfgDebris[section]) for section in cfgDebris.sections()}

    # volume = com1DFA.fetchRelVolume(
    #     releaseFile,
    #     cfgDict,
    #     pathToDem,
    #     secondaryReleaseFile,
    # )

  

if __name__ == "__main__":
#     # +++++++++REQUIRED+++++++++++++
#     # log file name; leave empty to use default runLog.log
#     # logName = "runDepthToThickness"
#     # comMod = "com1DFA"
#     # resType = "pft"
#     # profileAxis = "x"
#     # profileIndex = None
#     # ++++++++++++++++++++++++++++++
    # variation of release thickness
    relThVal = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # fetch input directory
    cfgMain = cfgUtils.getGeneralConfig()
    avaDir = cfgMain["MAIN"]["avalancheDir"]
    cfgDebris = cfgUtils.getModuleConfig(com1DFA)

    # call conversion
    runAdaptRelVol(avaDir, cfgDebris,relThVal)