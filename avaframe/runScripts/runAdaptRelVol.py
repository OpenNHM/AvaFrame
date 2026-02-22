"""
Run adaption of release volme by modifying the release thickness
"""
#TODO: Code bereinigen

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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

def getActRelVol(avaDir, cfgDebris, relThVal):
    '''
    '''

    # initialize logging
    # logUtils.initiateLogger(avaDir, "getReleaseVolume")
    # logging.getLogger().setLevel(logging.CRITICAL)
    logging.disable(logging.WARNING)


    # ------------------------------------------------------------------ #
    # cfgMain aufbauen (wird von com1DFAPreprocess benötigt)
    # ------------------------------------------------------------------ #
    cfgDebris["MAIN"] = {"avalancheDir": str(avaDir)}

    # Work-Verzeichnis bereinigen falls vorhanden (com1DFAPreprocess benötigt leeres Work-Dir)
    workDir = Path(avaDir) / "Work" / "com1DFA"
    if workDir.exists():
        shutil.rmtree(workDir)
        # print(f"Work-Verzeichnis bereinigt: {workDir}")
    
    # ------------------------------------------------------------------ #
    # Preprocessing: vollständig aufgelöste Konfiguration + Input-Dateien
    # ------------------------------------------------------------------ #
    simDict,_, inputSimFiles, _ = com1DFA.com1DFAPreprocess(cfgDebris, cfgInfo="")

    # Erste Simulation als Vorlage nehmen
    cuSimName = list(simDict.keys())[0]
    cfgSim = simDict[cuSimName]["cfgSim"]
    releaseFile = simDict[cuSimName]["relFile"]

    # print(f"Release-Datei:   {releaseFile}")
    # print(f"Simulations-Typ: {simDict[cuSimName]['simType']}")
    # print()

    # outDir für initializeSimulation (kein Schreiben gewünscht, temp-Verzeichnis)
    outDirTmp = Path(tempfile.mkdtemp())

    # ------------------------------------------------------------------ #
    # Volumen für verschiedene Release-Dicken berechnen
    # ------------------------------------------------------------------ #
    rho = cfgSim["GENERAL"].getfloat("rho")
    # print(f"Schneedichte rho = {rho} kg/m³\n")
    # print(f"{'relTh [m]':>12}  {'Volumen [m³]':>14}  {'Masse [kg]':>14}  {'Partikel':>10}")
    # print("-" * 56)

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

        # Report-Ausgabe in temp-Verzeichnis umleiten
        cfgSim["GENERAL"]["avalancheDir"] = str(outDirTmp)

        # Simulation initialisieren (t=0) → Partikel erzeugen
        particles, *_ = com1DFA.initializeSimulation(
            cfgSim, outDirTmp, demOri, inputSimLines, "getReleaseVolume"
        )

        # avalancheDir wieder zurücksetzen für nächste Iteration
        cfgSim["GENERAL"]["avalancheDir"] = str(avaDir)

        # tatsächliches Simulationsvolumen
        mTot = particles["mTot"]
        volume = mTot / rho
        nPart = particles["nPart"]

        volumes[relTh] = volume
        # print(f"{relTh:>12.2f}  {volume:>14.2f}  {mTot:>14.2f}  {nPart:>10d}")
    
    return volumes


def getGeomRelVol(avaDir,cfgDebris,relThval):

    # initialize logging
    # logUtils.initiateLogger(avaDir, "get geometric release volume")
    logging.disable(logging.WARNING)

    cfgGen = cfgDebris['GENERAL']

    # Sicherstellen, dass INPUT-Sektion existiert und relThFile leer ist
    # (leer = Dicke kommt aus cfg["GENERAL"]["relTh"], nicht aus einer Datei)
    if not cfgDebris.has_section("INPUT"):
        cfgDebris.add_section("INPUT")
    cfgDebris["INPUT"]["relThFile"] = ""
    cfgDebris["INPUT"]["secondaryRelThFile"] = ""
    cfgGen["timeDependentRelease"] = "False"
    cfgGen["relThFromFile"] = "False"
    cfgGen["secRelArea"] = "False"

    # debris-flow density [kg/m³]
    rho = cfgGen.getfloat("rho")
    # print(f"debris-flow density rho = {rho} kg/m³\n")

    # read input
    inputSimFiles = gI.getInputDataCom1DFA(avaDir)
    pathToDem = inputSimFiles["demFile"]

    # Erstes Release-Szenario (Shapefile) verwenden
    releaseFile = inputSimFiles["relFiles"][0]
    secondaryReleaseFile = inputSimFiles["secondaryRelFile"]

    # print(f"DEM:             {pathToDem}")
    # print(f"Release-Datei:   {releaseFile}")

    # ------------------------------------------------------------------ #
    # Volumen für verschiedene Release-Dicken berechnen
    # ------------------------------------------------------------------ #
    volumes = {}

    for relTh in relThVal:

        # relTh in der Konfiguration überschreiben
        cfgDebris["GENERAL"]["relTh"] = str(relTh)

        # fetchRelVolume erwartet cfg als einfaches dict (konvertiert intern zurück zu ConfigParser)
        cfgDict = {section: dict(cfgDebris[section]) for section in cfgDebris.sections()}

        volume = com1DFA.fetchRelVolume(
            releaseFile,
            cfgDict,
            pathToDem,
            secondaryReleaseFile,
        )

        volumes[relTh] = volume

    return volumes

def adaptRelVol(geomRelVol,actRelVol):
    #TODO: optimierte Iteration

    geomTh = list(geomRelVol.keys())
    geomVol = list(geomRelVol.values())

    
    for Th in geomTh:
        actTh = Th
        reldV = 1
        dV = 10
        dTh = 0.001
        diffTh = 1

        while diffTh > 1e-3:
        # while abs(dV) >= 5:# and dV > 0:

            actVol = getActRelVol(avaDir,cfgDebris,[actTh])
            actTh_0 = list(actVol.keys())[0]
            dV = geomRelVol[Th] - actVol[actTh_0]
            reldV = (abs(dV)) / geomRelVol[Th]
            print(f'abs. error: {dV:.2f} m³')
            print(f'rel. error: {reldV:.5f}')
            print(actVol)
            # Newton method
            actTh_i = actTh_0 + dTh
            print(f'Th + dTh: {actTh_i} m')
            dVdTh = getActRelVol(avaDir,cfgDebris,[actTh_i])
            dVdTh = geomRelVol[Th] - dVdTh[actTh_i]
            print(f'dV_new: {dVdTh}')
            dVdTh = (dVdTh - dV) / dTh
            print(f'dvdTh: {dVdTh}')
            actTh = actTh - dV / dVdTh
            # diff = geomRelVol[i] - actVol[j]
            # actTh = [j * (1 + relDiff/2)]
            diffTh = abs(actTh_0 - actTh)
            print(f'diffTh: {diffTh:.4f}')


  

if __name__ == "__main__":
    # +++++++++REQUIRED+++++++++++++
    # variation of release thickness
    relThVal = [1.0]

    # fetch input directory
    cfgMain = cfgUtils.getGeneralConfig()
    avaDir = cfgMain["MAIN"]["avalancheDir"]
    cfgDebris = cfgUtils.getModuleConfig(com1DFA)

    # call conversion
    actVolumes = getActRelVol(avaDir, cfgDebris,relThVal)
    print(f'actual volumes {actVolumes}')
    geomVolumes = getGeomRelVol(avaDir,cfgDebris,relThVal)
    print(f'geometric volumes {geomVolumes}')
    adaptRelVol(geomVolumes,actVolumes)
